#!/usr/bin/env python3

# =========================
import argparse
import datetime
import json
import os
import time
from pathlib import Path
import warnings
import faulthandler

# =========================
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from huggingface_hub import hf_hub_download, login  # login imported as in original

# =========================
import models_vit as models
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_finetune import train_one_epoch, evaluate

# =========================
faulthandler.enable()
warnings.simplefilter(action="ignore", category=FutureWarning)


def get_args_parser():
    parser = argparse.ArgumentParser(
        "MAE fine-tuning / linear probing for image classification", add_help=False
    )

    # ---- Core training
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size per GPU (effective batch size = batch_size * accum_iter * #gpus)")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--accum_iter", default=1, type=int,
                        help="Gradient accumulation steps")

    # ---- Model parameters
    parser.add_argument("--model", default="vit_large_patch16", type=str, metavar="MODEL",
                        help="Model entry in models_vit.py")
    parser.add_argument("--model_arch", default="dinov3_vits16", type=str, metavar="MODEL_ARCH",
                        help="Backbone architecture key (e.g., dinov2_vitl14, convnext_base, etc.)")
    parser.add_argument("--input_size", default=256, type=int, help="Image size")
    parser.add_argument("--drop_path", type=float, default=0.2, metavar="PCT", help="Drop path rate")
    parser.add_argument("--global_pool", action="store_true"); parser.set_defaults(global_pool=True)
    parser.add_argument("--cls_token", action="store_false", dest="global_pool",
                        help="Use class token instead of global pool for classification")

    # ---- Optimizer parameters
    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM", help="Clip grad norm")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="Absolute LR (overrides blr)")
    parser.add_argument("--blr", type=float, default=5e-3, metavar="LR",
                        help="Base LR: lr = blr * total_batch_size / 256")
    parser.add_argument("--layer_decay", type=float, default=0.65, help="Layer-wise LR decay (ViT)")
    parser.add_argument("--min_lr", type=float, default=1e-6, metavar="LR", help="Lower LR bound")
    parser.add_argument("--warmup_epochs", type=int, default=10, metavar="N", help="Warmup epochs")

    # ---- Augmentation
    parser.add_argument("--color_jitter", type=float, default=None, metavar="PCT")
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1", metavar="NAME")
    parser.add_argument("--smoothing", type=float, default=0.1)

    # ---- Random erase
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT")
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)
    parser.add_argument("--resplit", action="store_true", default=False)

    # ---- Mixup/Cutmix
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--cutmix_minmax", type=float, nargs="+", default=None)
    parser.add_argument("--mixup_prob", type=float, default=1.0)
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5)
    parser.add_argument("--mixup_mode", type=str, default="batch")

    # ---- Finetuning & adaptation
    parser.add_argument("--finetune", default="", type=str, help="Checkpoint id/path (see model rules below)")
    parser.add_argument("--task", default="", type=str, help="Task name for logging/output grouping")
    parser.add_argument("--adaptation", default="finetune", choices=["finetune", "lp"],
                        help="Adaptation strategy: finetune=full fine-tune, lp=linear probe (train head only)")

    # ---- Dataset & paths
    parser.add_argument("--data_path", default="./data/", type=str)
    parser.add_argument("--nb_classes", default=8, type=int)
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--log_dir", default="./output_logs")

    # >>> NEW: training data efficiency <<<
    parser.add_argument(
        "--dataratio", type=str, default="1.0",
        help=('Training data ratio(s) for subsampling in build_dataset. '
              'Use a single float in (0,1] (e.g., 0.25) or a comma-separated list '
              '(e.g., "1.0,0.5,0.25") if your build_dataset supports sweeps.')
    )
    parser.add_argument(
        "--stratified", action="store_true",
        help="If set, subsample training data in a class-stratified manner (requires support in build_dataset)."
    )

    # ---- Runtime
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="Resume full state (optimizer, scaler, etc.)")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
    parser.add_argument("--eval", action="store_true", help="Evaluation only")
    parser.add_argument("--dist_eval", action="store_true", default=False,
                        help="Distributed evaluation (faster monitoring during training)")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true"); parser.set_defaults(pin_mem=True)

    # ---- Distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    # ---- Misc
    parser.add_argument("--savemodel", action="store_true", default=True, help="Save best model")
    parser.add_argument("--norm", default="IMAGENET", type=str)
    parser.add_argument("--enhance", action="store_true", default=False)
    parser.add_argument("--datasets_seed", default=2026, type=int)

    return parser


# =========================
# Main
# =========================
def main(args, criterion):
    # ---- Optionally load args from resume (when training)
    if args.resume and not args.eval:
        resume_path = args.resume
        checkpoint = torch.load(args.resume, map_location="cpu")
        print(f"Load checkpoint (args) from: {args.resume}")
        args = checkpoint["args"]
        args.resume = resume_path

    # ---- Distributed setup
    misc.init_distributed_mode(args)

    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"{args}".replace(", ", ",\n"))

    device = torch.device(args.device)

    # ---- Reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # ---- Build model
    if args.model == "RETFound_mae":
        model = models.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    else:
        model = models.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            args=args,
        )

    # ---- Load pre-trained weights (if requested and not eval-only)
    if args.finetune and not args.eval:
        print(f"Preparing to load pre-trained weights: {args.finetune}")

        if args.model in ["Dinov3", "Dinov2"]:
            checkpoint_path = args.finetune  # local path
        elif args.model in ["RETFound_dinov2", "RETFound_mae"]:
            print(f"Downloading pre-trained weights from Hugging Face Hub: {args.finetune}")
            checkpoint_path = hf_hub_download(
                repo_id=f"YukunZhou/{args.finetune}",
                filename=f"{args.finetune}.pth",
            )
        else:
            raise ValueError(
                f"Unsupported model '{args.model}'. "
                f"Expected one of: Dinov3, Dinov2, RETFound_dinov2, RETFound_mae"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print(f"Loaded pre-trained checkpoint from: {checkpoint_path}")

        if args.model in ["Dinov3", "Dinov2"]:
            checkpoint_model = checkpoint
        elif args.model == "RETFound_dinov2":
            checkpoint_model = checkpoint["teacher"]
        else:  # RETFound_mae
            checkpoint_model = checkpoint["model"]

        # -- Key hygiene
        checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}

        # -- Remove classifier if shape mismatched
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # -- Interpolate pos embed (ViT)
        interpolate_pos_embed(model, checkpoint_model)

        # -- Load backbone weights (non-strict)
        _ = model.load_state_dict(checkpoint_model, strict=False)

        # -- Re-init head
        if hasattr(model, "head") and hasattr(model.head, "weight"):
            trunc_normal_(model.head.weight, std=2e-5)

    # ---- Datasets & samplers
    dataset_train = build_dataset(is_train="train", args=args)
    dataset_val   = build_dataset(is_train="val",   args=args)
    dataset_test  = build_dataset(is_train="test",  args=args)

    num_tasks   = misc.get_world_size()
    global_rank = misc.get_rank()

    if not args.eval:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print(f"Sampler_train = {sampler_train}")
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print("Warning: dist eval with dataset not divisible by #procs; results may differ slightly.")
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.dist_eval:
        if len(dataset_test) % num_tasks != 0:
            print("Warning: dist eval test set not divisible by #procs; results may differ slightly.")
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    # ---- Logging
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.task))
    else:
        log_writer = None

    # ---- DataLoaders
    if not args.eval:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=True,
        )
        print(f"len of train_set: {len(data_loader_train) * args.batch_size}")

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=False,
        )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    # ---- Mixup/CutMix
    mixup_fn = None
    mixup_active = (args.mixup > 0) or (args.cutmix > 0.) or (args.cutmix_minmax is not None)
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes
        )

    # ---- Eval-only: resume weights
    if args.resume and args.eval:
        checkpoint = torch.load(args.resume, map_location="cpu")
        print(f"Load checkpoint for eval from: {args.resume}")
        model.load_state_dict(checkpoint["model"])

    model.to(device)
    model_without_ddp = model

    # ---- Adaptation toggle
    if args.adaptation == "lp":
        for name, param in model.named_parameters():
            param.requires_grad = ("head" in name)
        print("[Adaptation] Linear probe: training classifier head only.")
    else:
        print("[Adaptation] Full fine-tuning: training all parameters.")

    # ---- Count trainable params
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable params (M): {n_parameters / 1.e6:.2f}")

    # ---- LR scaling by effective batch size
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"actual lr: {args.lr:.2e}")
    print(f"accumulate grad iterations: {args.accum_iter}")
    print(f"effective batch size: {eff_batch_size}")

    # ---- DDP (if available)
    if args.distributed and torch.cuda.device_count() > 1:
        ddp_kwargs = {}
        if args.adaptation == "lp":
            ddp_kwargs["find_unused_parameters"] = True
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], **ddp_kwargs
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model  # single-GPU

    # ---- Optimizer param groups (after freezing)
    no_weight_decay = (model_without_ddp.no_weight_decay()
                       if hasattr(model_without_ddp, "no_weight_decay") else [])


    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        weight_decay=args.weight_decay,
        no_weight_decay_list=no_weight_decay,
        layer_decay=args.layer_decay,
    )
    for g in param_groups:
        g["params"] = [p for p in g["params"] if p.requires_grad]

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()
    print(f"criterion = {criterion}")

    # ---- Load previous full state (optimizer, scaler, etc.)
    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    # =========================
    # Eval-only Short Circuit
    # =========================
    if args.eval:
        if "checkpoint" in locals() and isinstance(checkpoint, dict) and ("epoch" in checkpoint):
            print(f"Test with the best model at epoch = {checkpoint['epoch']}")
        test_stats, auc_roc = evaluate(
            data_loader_test, model, device, args, epoch=0, mode="test",
            num_class=args.nb_classes, log_writer=log_writer
        )
        return

    # =========================
    # Train Loop
    # =========================
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_score = 0.0
    best_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer, args=args
        )

        val_stats, val_score = evaluate(
            data_loader_val, model, device, args, epoch, mode="val",
            num_class=args.nb_classes, log_writer=log_writer
        )

        if max_score < val_score:
            max_score = val_score
            best_epoch = epoch
            if args.output_dir and args.savemodel:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, mode="best"
                )
        print(f"Best epoch = {best_epoch}, Best score = {max_score:.4f}")

        if log_writer is not None:
            log_writer.add_scalar("loss/val", val_stats["loss"], epoch)
            log_writer.flush()

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                     "epoch": epoch,
                     "n_parameters": n_parameters}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, args.task, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # =========================
    # Final Test (Best Ckpt)
    # =========================
    ckpt_path = os.path.join(args.output_dir, args.task, "checkpoint-best.pth")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    print(f"Test with the best model, epoch = {checkpoint.get('epoch', -1)}:")
    _test_stats, _auc_roc = evaluate(
        data_loader_test, model, device, args, -1, mode="test",
        num_class=args.nb_classes, log_writer=None
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    criterion = torch.nn.CrossEntropyLoss()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, criterion)
