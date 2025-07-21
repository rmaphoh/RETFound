import argparse, datetime, json, numpy as np, os, time
from pathlib import Path
import torch, torch.backends.cudnn as cudnn
from timm.data.mixup import Mixup
import models_vit as models, util.lr_decay as lrd, util.misc as misc
from util.datasets import build_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from huggingface_hub import hf_hub_download
from engine_finetune import train_one_epoch, evaluate
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--model', default='vit_large_patch16', type=str)
    parser.add_argument('--input_size', default=256, type=int)
    parser.add_argument('--drop_path', type=float, default=0.2)
    parser.add_argument('--clip_grad', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=5e-3)
    parser.add_argument('--layer_decay', type=float, default=0.65)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--finetune', default='', type=str)
    parser.add_argument('--task', default='', type=str)
    parser.add_argument('--global_pool', action='store_true', default=True)
    parser.add_argument('--data_path', default='./data/', type=str)
    parser.add_argument('--nb_classes', default=5, type=int)
    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--log_dir', default='./output_logs')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    return parser

def main(args):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_test = build_dataset(is_train='test', args=args)
    if not args.eval:
        dataset_train = build_dataset(is_train='train', args=args)
        dataset_val = build_dataset(is_train='val', args=args)
    else:
        dataset_train, dataset_val = None, None

    if args.distributed:
        # ... (distributed setup omitted for Colab clarity)
        pass
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        if not args.eval:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    log_writer = None # Disabled for Colab

    data_loader_test = torch.utils.data.DataLoader(dataset_test, sampler=sampler_test, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    if not args.eval:
        data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
        data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    model = models.__dict__[args.model](img_size=args.input_size, num_classes=args.nb_classes, drop_path_rate=args.drop_path, global_pool=args.global_pool)

    if args.finetune and not args.eval:
        print(f"Downloading pre-trained weights from Hugging Face: {args.finetune}")
        checkpoint_path = hf_hub_download(repo_id=f'YukunZhou/{args.finetune}', filename=f'{args.finetune}.pth')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')['model']
        msg = model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded pre-trained checkpoint from {args.finetune} with message: {msg}")

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')['model']
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    print(f'Number of model params (M): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1.e6:.2f}')

    if args.eval:
        evaluate(data_loader_test, model, device, args, 0, 'test', args.nb_classes, log_writer)
        return

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None: args.lr = args.blr * eff_batch_size / 256
    print(f"Actual lr: {args.lr:.2e}")
    param_groups = lrd.param_groups_lrd(model, args.weight_decay, no_weight_decay_list=model.no_weight_decay(), layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()
    criterion = torch.nn.CrossEntropyLoss()

    print(f"--- Starting Training for {args.epochs} epochs ---")
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, args=args)
        val_stats, _ = evaluate(data_loader_val, model, device, args, epoch, 'val', args.nb_classes, log_writer=log_writer)
        print(f"EPOCH:{epoch} | Val Acc: {val_stats['acc1']:.1f}%")
        if max_accuracy < val_stats["acc1"]:
            max_accuracy = val_stats["acc1"]
            misc.save_model(args=args, model=model, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, mode='best')
        print(f'Max accuracy: {max_accuracy:.2f}%')

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir: Path(os.path.join(args.output_dir, args.task)).mkdir(parents=True, exist_ok=True)
    main(args)
