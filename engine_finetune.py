import math, sys
from typing import Iterable, Optional
import torch, torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from timm.utils import accuracy
import util.misc as misc, util.lr_sched as lr_sched

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples, targets = samples.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            loss = criterion(model(samples), targets)
        loss_value = loss.item()
        if not math.isfinite(loss_value): sys.exit(f"Loss is {loss_value}, stopping training")
        loss_scaler(loss, optimizer, parameters=model.parameters())
        optimizer.zero_grad()
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, args, epoch, mode, num_class, log_writer=None):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header, model.eval(), all_preds, all_labels, all_probs = 'Test:', [], [], []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images, target = batch[0].to(device, non_blocking=True), batch[-1].to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
        preds, probs = torch.argmax(output, dim=1), F.softmax(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        all_probs.extend(probs.cpu().detach().numpy())
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(accuracy(output, target, topk=(1,))[0].item(), n=images.shape[0])
    print(f'* Acc@1 {metric_logger.acc1.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}')
    all_labels, all_preds, all_probs = np.array(all_labels), np.array(all_preds), np.array(all_probs)
    print("\n--- Performance Metrics ---")
    if len(np.unique(all_labels)) > 1 and len(np.unique(all_preds)) > 1:
        try:
            print(f"AUROC (Macro): {roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro'):.4f}")
        except Exception as e: print(f"Could not calculate AUROC: {e}")
    else: print("Skipping AUROC: not enough classes in labels or predictions.")
    cm, fp = confusion_matrix(all_labels, all_preds), cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (fp + (cm.sum(axis=1) - np.diag(cm)) + np.diag(cm))
    print(f"Specificity (Macro): {np.mean(tn / (tn + fp)):.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=[f'Class {i}' for i in range(num_class)], digits=4, zero_division=0))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, 'results'
