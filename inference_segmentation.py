import os
import cv2
import torch
import argparse
import numpy as np
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

from models_segmentation import RETFoundSegmentation

# Select device automatically
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Metrics
# ============================================================
def dice_iou(pred, gt, smooth=1e-6):
    """
    Compute Dice coefficient and IoU between prediction and ground truth.

    Args:
        pred  : Binary prediction mask (numpy)
        gt    : Binary ground truth mask (numpy)
        smooth: Small constant to avoid division by zero

    Returns:
        dice : Dice similarity score
        iou  : Intersection over Union score
    """

    # Convert to boolean for logical operations
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    # Intersection and union areas
    inter = (pred & gt).sum()
    union = (pred | gt).sum()

    # Dice and IoU formulas
    dice = (2 * inter + smooth) / (pred.sum() + gt.sum() + smooth)
    iou  = (inter + smooth) / (union + smooth)

    return dice, iou


# ============================================================
# Model loader
# ============================================================
def load_model(ckpt, img_size=256):
    """
    Load trained RETFound segmentation model.

    Args:
        ckpt     : Path to checkpoint
        img_size : Input size used during training

    Returns:
        model in evaluation mode
    """

    model = RETFoundSegmentation(img_size=img_size).to(DEVICE)

    # Load weights
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)

    model.eval()
    return model


# ============================================================
# Preprocess
# ============================================================
def preprocess(img_path, img_size=256):
    """
    Read and preprocess input image.

    - Read image using OpenCV
    - Convert BGR → RGB
    - Resize and normalize as per RETFound training
    - Convert to tensor

    Returns:
        tensor image for model, original image for visualization
    """

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tf = Compose([
        Resize(img_size, img_size),
        Normalize((0.485, 0.456, 0.406),
                  (0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    aug = tf(image=img)
    return aug["image"].unsqueeze(0), img


# ============================================================
# Overlay visualization
# ============================================================
def make_overlay(img, mask):
    """
    Create green overlay on original image using predicted mask.

    Args:
        img  : Original RGB image
        mask : Binary prediction mask

    Returns:
        Overlay visualization
    """

    # Ensure uint8 type
    mask = mask.astype(np.uint8)

    # Resize mask back to original image size
    mask = cv2.resize(
        mask,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Create green color mask
    color = np.zeros_like(img, dtype=np.uint8)
    color[:, :, 1] = mask * 255      # Green channel

    # Blend with original image
    overlay = cv2.addWeighted(img, 0.7, color, 0.3, 0)
    return overlay


# ============================================================
# Main inference
# ============================================================
def run_inference(args):
    """
    Perform inference on test set.

    - Load model
    - Iterate over test images
    - Predict masks
    - Compute Dice & IoU
    - Save overlay visualizations
    """

    # ----- Resolve dataset paths from single root -----
    test_img_dir  = os.path.join(args.data_path, "test", "images")
    test_mask_dir = os.path.join(args.data_path, "test", "masks")

    # Validate paths
    if not os.path.isdir(test_img_dir):
        raise FileNotFoundError(f"Images folder not found: {test_img_dir}")

    if not os.path.isdir(test_mask_dir):
        raise FileNotFoundError(f"Masks folder not found: {test_mask_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load trained model
    model = load_model(args.ckpt, args.img_size)

    dice_scores = []
    iou_scores  = []

    print("\n===== Running Inference =====\n")

    # Iterate through test images
    for name in sorted(os.listdir(test_img_dir)):

        img_path = os.path.join(test_img_dir, name)

        # Corresponding ground truth name
        stem = os.path.splitext(name)[0]
        gt_name = stem + "_mask.png"
        gt_path = os.path.join(test_mask_dir, gt_name)

        if not os.path.isfile(gt_path):
            continue

        # ----- Predict -----
        tensor, orig = preprocess(img_path, args.img_size)

        with torch.no_grad():
            out = model(tensor.to(DEVICE))
            pred = out.argmax(1).squeeze().cpu().numpy()

        # ----- Load GT -----
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt = (gt > 0).astype("uint8")

        # Align GT to prediction resolution
        gt = cv2.resize(
            gt,
            (pred.shape[1], pred.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # ----- Metrics -----
        dice, iou = dice_iou(pred, gt)
        dice_scores.append(dice)
        iou_scores.append(iou)

        # ----- Save overlay only -----
        over = make_overlay(orig, pred)
        cv2.imwrite(os.path.join(args.out_dir, stem + "_overlay.png"), over)

        # Print per-image result
        print(f"{name:25s}  Dice: {dice:.4f}   IoU: {iou:.4f}")

    # ---------------------------------------------------------
    # Final aggregated report
    print("\n===== FINAL REPORT =====")
    print("Mean Dice:", round(np.mean(dice_scores), 4))
    print("Mean IoU :", round(np.mean(iou_scores), 4))


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":

    # Command line argument parser
    parser = argparse.ArgumentParser(description="RETFound Segmentation Inference")

    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained model checkpoint")

    parser.add_argument("--data_path", type=str, required=True,
                        help="Root dataset folder containing test/images and test/masks")

    parser.add_argument("--out_dir", type=str, default="segmentation_output/inference",
                        help="Output directory for overlays")

    parser.add_argument("--img_size", type=int, default=256,
                        help="Input resize dimension")

    args = parser.parse_args()

    # Start inference
    run_inference(args)
