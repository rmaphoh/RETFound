import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download
from util.pos_embed import interpolate_pos_embed

from models_segmentation import RETFoundSegmentation
from engine_segmentation import (
    train_segmentation,
    evaluate_segmentation,
    combined_loss_fn,
    compute_metrics,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Dataset Definition
# ============================================================
class OCTDrusenDataset(Dataset):
    """
    Dataset for OCT drusen segmentation.

    Expected structure:
        root/
          ├── images/
          └── masks/  (same filename + _mask.png)
    """

    def __init__(self, root, transform=None):

        self.image_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.transform = transform

        # Collect valid image–mask pairs
        self.samples = []
        for img_name in sorted(os.listdir(self.image_dir)):
            stem = os.path.splitext(img_name)[0]
            mask_name = stem + "_mask.png"
            mask_path = os.path.join(self.mask_dir, mask_name)

            if os.path.isfile(mask_path):
                self.samples.append((img_name, mask_name))

        print(f"Loaded {len(self.samples)} valid samples from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load image and mask and apply transforms.
        """

        img_name, mask_name = self.samples[idx]

        # ----- Image -----
        img = cv2.imread(os.path.join(self.image_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ----- Mask -----
        mask = cv2.imread(
            os.path.join(self.mask_dir, mask_name),
            cv2.IMREAD_GRAYSCALE
        )

        mask = (mask > 0).astype("uint8")

        # ----- Transform -----
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        return img, torch.tensor(mask, dtype=torch.long)


# ============================================================
# Main Training Script
# ============================================================
def main():

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------
    parser = argparse.ArgumentParser("RETFound Segmentation")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--drop_path", type=float, default=0.2)

    parser.add_argument("--finetune", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./segmentation_output")

    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--ce_weight", type=str, default="0.3,0.7")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------
    transform = Compose([
        Resize(args.img_size, args.img_size),
        Normalize((0.485, 0.456, 0.406),
                  (0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # --------------------------------------------------------
    # Datasets & Loaders  (ONLY TRAIN + VAL)
    # --------------------------------------------------------
    train_ds = OCTDrusenDataset(os.path.join(args.data_path, "train"), transform)
    val_ds   = OCTDrusenDataset(os.path.join(args.data_path, "val"), transform)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=4)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = RETFoundSegmentation(
        args.img_size,
        args.patch_size,
        num_classes=2,
        drop_path=args.drop_path
    ).to(device)

    # --------------------------------------------------------
    # Load Pretrained RETFound Encoder (optional)
    # --------------------------------------------------------
    if args.finetune:

        if os.path.isfile(args.finetune):
            ckpt_path = args.finetune
        else:
            ckpt_path = hf_hub_download(
                repo_id=f"YukunZhou/{args.finetune}",
                filename="pytorch_model.bin"
            )

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = state["model"] if "model" in state else state

        # Remove classification head
        for k in ["head.weight", "head.bias"]:
            if k in state:
                del state[k]

        interpolate_pos_embed(model.encoder, state)
        model.encoder.load_state_dict(state, strict=False)

        print("Pretrained RETFound weights loaded.")

    # --------------------------------------------------------
    # Loss & Optimizer
    # --------------------------------------------------------
    ce_weights = torch.tensor(
        [float(x) for x in args.ce_weight.split(",")]
    ).to(device)

    ce_loss = nn.CrossEntropyLoss(weight=ce_weights)

    def loss_fn(out, tgt):
        return combined_loss_fn(out, tgt, ce_loss, args.dice_weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    print("\n[DEBUG] Starting training loop...\n")

    # --------------------------------------------------------
    # Training Loop (NO TEST DATA)
    # --------------------------------------------------------
    best = float("inf")

    for e in range(args.epochs):

        print(f"[DEBUG] Entered epoch {e+1}")

        train_loss = train_segmentation(
            model, train_loader, loss_fn, optimizer, device
        )

        val_loss, P, T = evaluate_segmentation(
            model, val_loader, loss_fn, device
        )

        acc, dice, iou = compute_metrics(P, T)

        print(
            f"Epoch {e+1}: "
            f"Train={train_loss:.4f} | "
            f"Val={val_loss:.4f} | "
            f"Dice={dice:.4f} | "
            f"IoU={iou:.4f}"
        )

        # Save best model based on validation loss
        if val_loss < best:
            best = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, "best.pth")
            )


if __name__ == "__main__":
    main()
