import torch
import torch.nn.functional as F
import numpy as np


# ============================================================
# Dice Loss
# ============================================================
def dice_loss(pred, target, smooth=1e-6):
    """
    Computes multi-class Dice loss.

    Args:
        pred   : Raw logits from the model [B, C, H, W]
        target : Ground truth labels [B, H, W]
        smooth : Small constant to avoid division by zero

    Returns:
        Scalar dice loss (1 - dice coefficient)
    """

    # Convert logits to probabilities
    pred = F.softmax(pred, dim=1)

    # Number of classes (e.g., 2 for background/drusen)
    num_classes = pred.shape[1]

    # Convert target to one-hot representation
    target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    # Intersection between prediction and ground truth
    inter = (pred * target_oh).sum((2, 3))

    # Sum of prediction and ground truth areas
    union = pred.sum((2, 3)) + target_oh.sum((2, 3))

    # Dice coefficient → converted to loss (1 - dice)
    return 1 - ((2 * inter + smooth) / (union + smooth)).mean()


# ============================================================
# Combined CE + Dice Loss
# ============================================================
def combined_loss_fn(outputs, targets, ce_fn, dice_w=1.0):
    """
    Combines Cross-Entropy loss with Dice loss.

    Args:
        outputs : Model logits [B, C, H, W]
        targets : Ground truth [B, H, W]
        ce_fn   : CrossEntropyLoss function
        dice_w  : Weight for dice loss term

    Returns:
        Weighted sum of CE and Dice loss
    """

    return ce_fn(outputs, targets) + dice_w * dice_loss(outputs, targets)


# ============================================================
# Metric Computation
# ============================================================
def compute_metrics(preds, targets, smooth=1e-6):
    """
    Computes pixel accuracy, Dice, and IoU.

    Args:
        preds   : Binary predictions (numpy array)
        targets : Binary ground truth (numpy array)

    Returns:
        pixel_acc : Pixel-wise accuracy
        dice      : Dice coefficient
        iou       : Intersection over Union
    """

    # Pixel-wise accuracy
    pixel_acc = (preds == targets).mean()

    # Intersection between prediction and GT
    inter = (preds & targets).sum()

    # Dice coefficient
    dice = (2 * inter + smooth) / (preds.sum() + targets.sum() + smooth)

    # IoU computation
    union = preds.sum() + targets.sum() - inter
    iou = (inter + smooth) / (union + smooth)

    return pixel_acc, dice, iou


# ============================================================
# Training Loop
# ============================================================
def train_segmentation(model, loader, loss_fn, optimizer, device):
    """
    One epoch training for segmentation model.

    Args:
        model     : Segmentation network
        loader    : Training dataloader
        loss_fn   : Loss function (CE + Dice)
        optimizer : Optimizer
        device    : cuda/cpu

    Returns:
        Average epoch loss
    """

    model.train()
    total = 0

    for step, (x, y) in enumerate(loader):

        # Move data to device
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward pass
        out = model(x)

        # Compute loss
        loss = loss_fn(out, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total += loss.item() * x.size(0)

        # Progress print
        if step % 10 == 0:
            print(f"  [batch {step}/{len(loader)}] loss: {loss.item():.4f}")

    return total / len(loader.dataset)


# ============================================================
# Validation / Evaluation Loop
# ============================================================
@torch.no_grad()
def evaluate_segmentation(model, loader, loss_fn, device):
    """
    Runs inference on validation/test set and collects predictions.

    Args:
        model   : Trained model
        loader  : Val/test dataloader
        loss_fn : Loss function
        device  : cuda/cpu

    Returns:
        avg_loss : Average loss over dataset
        P        : All predictions (numpy)
        T        : All ground truth (numpy)
    """

    model.eval()

    total = 0
    P, T = [], []

    for x, y in loader:

        x, y = x.to(device), y.to(device)

        # Forward pass
        out = model(x)

        # Loss computation
        loss = loss_fn(out, y)
        total += loss.item() * x.size(0)

        # Store predictions and targets
        P.append(out.argmax(1).cpu())
        T.append(y.cpu())

    return total / len(loader.dataset), torch.cat(P).numpy(), torch.cat(T).numpy()
