import os
import argparse
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import eyepy as ep
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download
from util.pos_embed import interpolate_pos_embed

# Minimal logging configuration
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ------------------------------
# Multi-eye Dataset for .eye files in a folder
# ------------------------------
class MultiEyeSegmentationDataset(Dataset):
    def __init__(self, folder, transform=None):
        """
        folder: Path to a folder containing .eye files.
        transform: Albumentations transform applied to both image and mask.
        """
        # List all .eye files in the folder
        self.eye_files = sorted(
            [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".eye")]
        )
        self.transform = transform
        # Build a list of (eye_file, bscan_index) tuples for all volumes
        self.samples = []
        for eye_file in self.eye_files:
            volume = ep.EyeVolume.load(eye_file)
            num_scans = volume.shape[0]
            for idx in range(num_scans):
                self.samples.append((eye_file, idx))
        # Cache loaded volumes to avoid repeated disk I/O
        self.cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        eye_file, idx = self.samples[index]
        if eye_file not in self.cache:
            self.cache[eye_file] = ep.EyeVolume.load(eye_file)
        volume = self.cache[eye_file]
        image = volume[idx].data  # assume shape: (H, W)
        image = np.stack([image] * 3, axis=-1)  # convert grayscale to 3 channels
        # Convert mask from bool to uint8 to avoid OpenCV errors
        mask = volume.volume_maps["drusen"].data[idx].astype(np.uint8)  # shape: (H, W)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            mask = mask.long()
        else:
            image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1) / 255.0
            mask = torch.tensor(mask, dtype=torch.long)
        return image, mask


# ------------------------------
# Segmentation Model Components
# ------------------------------
from models_vit import RETFound_mae  # Ensure this import is correct


class SegmentationHead(nn.Module):
    def __init__(self, hidden_dim, num_classes, img_size, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.h = img_size // patch_size
        self.w = img_size // patch_size
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # x: [B, num_tokens, hidden_dim]
        B, N, C = x.shape
        x = x.reshape(B, self.h, self.w, C)
        x = x.permute(0, 3, 1, 2)  # [B, C, h, w]
        x = F.interpolate(
            x, scale_factor=self.patch_size, mode="bilinear", align_corners=False
        )
        x = self.conv(x)
        return x


class RETFoundSegmentation(nn.Module):
    def __init__(
        self, img_size=512, patch_size=16, hidden_dim=1024, num_classes=2, drop_path=0.2
    ):
        super().__init__()
        self.encoder = RETFound_mae(
            img_size=img_size,
            num_classes=num_classes,
            drop_path_rate=drop_path,
            global_pool=False,
        )
        self.seg_head = SegmentationHead(hidden_dim, num_classes, img_size, patch_size)

    def forward(self, x):
        B = x.shape[0]
        x_tokens = self.encoder.patch_embed(x)  # [B, num_patches, hidden_dim]
        cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
        x_tokens = torch.cat((cls_tokens, x_tokens), dim=1)
        x_tokens = x_tokens + self.encoder.pos_embed
        x_tokens = self.encoder.pos_drop(x_tokens)
        for blk in self.encoder.blocks:
            x_tokens = blk(x_tokens)
        x_tokens = self.encoder.norm(x_tokens)
        tokens = x_tokens[:, 1:]  # exclude cls token
        seg_map = self.seg_head(tokens)
        return seg_map


# ------------------------------
# Loss Functions for Imbalanced Segmentation
# ------------------------------
def dice_loss(pred, target, smooth=1e-6):
    """
    Computes Dice loss.
    pred: logits [B, num_classes, H, W]
    target: ground truth [B, H, W] (long)
    """
    pred_soft = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    target_onehot = (
        F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    )
    intersection = (pred_soft * target_onehot).sum(dim=(2, 3))
    union = pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def combined_loss_fn(outputs, targets, ce_loss_fn, dice_weight=1.0):
    ce_loss = ce_loss_fn(outputs, targets)
    d_loss = dice_loss(outputs, targets)
    return ce_loss + dice_weight * d_loss


# ------------------------------
# Metrics Function
# ------------------------------
def compute_metrics(preds, targets, smooth=1e-6):
    pixel_acc = np.mean(preds == targets)
    intersection = np.sum(preds * targets)
    dice = (2.0 * intersection + smooth) / (np.sum(preds) + np.sum(targets) + smooth)
    union = np.sum(preds) + np.sum(targets) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return pixel_acc, dice, iou


# ------------------------------
# Training and Evaluation Loops
# ------------------------------
def train_segmentation(model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


def evaluate_segmentation(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())
    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    return epoch_loss, all_preds, all_targets


# ------------------------------
# Main Training Script
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="RETFound Segmentation Fine-tuning")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Base dataset directory with subfolders: train, val, test",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--img_size", type=int, default=512, help="Input image size (square)"
    )
    parser.add_argument(
        "--patch_size", type=int, default=16, help="Patch size used by the encoder"
    )
    parser.add_argument("--drop_path", type=float, default=0.2, help="Drop path rate")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./segmentation_output",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--finetune",
        type=str,
        default="",
        help="Path to pretrained RETFound checkpoint (or repo name for download)",
    )
    parser.add_argument(
        "--dice_weight", type=float, default=1.0, help="Weight for dice loss term"
    )
    parser.add_argument(
        "--ce_weight",
        type=str,
        default="0.3,0.7",
        help="Comma-separated weights for cross entropy loss (background, drusen)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Minimal transform: resize and normalize
    transform = Compose(
        [
            Resize(args.img_size, args.img_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # Create datasets for train, val, and test
    train_folder = os.path.join(args.data_path, "train")
    val_folder = os.path.join(args.data_path, "val")
    test_folder = os.path.join(args.data_path, "test")
    from torch.utils.data import DataLoader

    train_dataset = MultiEyeSegmentationDataset(train_folder, transform=transform)
    val_dataset = MultiEyeSegmentationDataset(val_folder, transform=transform)
    test_dataset = MultiEyeSegmentationDataset(test_folder, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = RETFoundSegmentation(
        img_size=args.img_size,
        patch_size=args.patch_size,
        hidden_dim=1024,
        num_classes=2,
        drop_path=args.drop_path,
    )
    model.to(device)

    # ----- Load Pretrained RETFound Foundation Weights -----
    if args.finetune:
        if os.path.exists(args.finetune):
            checkpoint_path = args.finetune
            print(
                f"Loading pretrained weights from local checkpoint: {checkpoint_path}"
            )
        else:
            print(f"Downloading pretrained weights from: {args.finetune}")
            checkpoint_path = hf_hub_download(
                repo_id=f"YukunZhou/{args.finetune}",
                filename=f"{args.finetune}.pth",
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model" in checkpoint:
            pretrained_dict = checkpoint["model"]
        else:
            pretrained_dict = checkpoint
        for k in ["head.weight", "head.bias"]:
            if k in pretrained_dict:
                print(f"Removing key {k} from pretrained checkpoint")
                del pretrained_dict[k]
        interpolate_pos_embed(model.encoder, pretrained_dict)
        model.encoder.load_state_dict(pretrained_dict, strict=False)
        print("Pretrained RETFound encoder weights loaded.")

    # Create weighted CrossEntropyLoss for class weighting
    ce_weights = [float(x) for x in args.ce_weight.split(",")]
    ce_weights_tensor = torch.tensor(ce_weights, device=device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=ce_weights_tensor)

    def loss_fn(outputs, targets):
        return combined_loss_fn(
            outputs, targets, ce_loss_fn, dice_weight=args.dice_weight
        )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss = train_segmentation(model, train_loader, loss_fn, optimizer, device)
        val_loss, all_preds, all_targets = evaluate_segmentation(
            model, val_loader, loss_fn, device
        )
        pixel_acc, dice, iou = compute_metrics(all_preds, all_targets)
        print(
            f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
        print(f"Metrics: Pixel Acc: {pixel_acc:.4f}, Dice: {dice:.4f}, IoU: {iou:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = os.path.join(
                args.output_dir, f"checkpoint_epoch{epoch+1}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Evaluate on test set
    test_loss, test_preds, test_targets = evaluate_segmentation(
        model, test_loader, loss_fn, device
    )
    pixel_acc, dice, iou = compute_metrics(test_preds, test_targets)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Metrics: Pixel Acc: {pixel_acc:.4f}, Dice: {dice:.4f}, IoU: {iou:.4f}")


if __name__ == "__main__":
    main()

# example usage:
# !python segmentation_finetune.py --data_path ./data/ --epochs 50 --batch --data_path "Data" --finetune "" --epochs 50 --batch_size 1 --lr 1e-4 --img_size 256 --patch_size 16 --drop_path 0.2 --ce_weight "0.3,0.7" --dice_weight 1.0 --output_dir "./segmentation_output/2"
