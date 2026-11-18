import torch
import torch.nn as nn
import torch.nn.functional as F
from models_vit import RETFound_mae


# ============================================================
# Decoder / Segmentation Head
# ============================================================
class SegmentationHead(nn.Module):
    """
    Lightweight decoder that converts ViT patch embeddings
    into full-resolution segmentation map.

    Steps:
    - Reshape sequence → 2D feature map
    - Upsample to original image size
    - Apply small CNN to produce class logits
    """

    def __init__(self, hidden_dim, num_classes, img_size, patch_size):
        super().__init__()

        # Patch geometry from ViT
        self.patch_size = patch_size
        self.h = img_size // patch_size
        self.w = img_size // patch_size

        # Simple convolutional decoder
        self.conv = nn.Sequential(
            # Reduce channel dimension
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),

            # Final layer → number of classes
            nn.Conv2d(hidden_dim // 2, num_classes, 1),
        )

    def forward(self, x):
        """
        Args:
            x: ViT token embeddings [B, N, C]
               (without CLS token)

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """

        B, N, C = x.shape

        # Reshape sequence back to 2D feature map
        x = x.reshape(B, self.h, self.w, C).permute(0, 3, 1, 2)

        # Upsample from patch grid → image resolution
        x = F.interpolate(
            x,
            scale_factor=self.patch_size,
            mode="bilinear",
            align_corners=False
        )

        # Apply conv decoder to get class logits
        return self.conv(x)


# ============================================================
# Full RETFound + Decoder Model
# ============================================================
class RETFoundSegmentation(nn.Module):
    """
    Segmentation model built on top of RETFound MAE encoder.

    Architecture:
        RETFound ViT Encoder  →  SegmentationHead Decoder
    """

    def __init__(
        self,
        img_size=512,
        patch_size=16,
        hidden_dim=1024,
        num_classes=2,
        drop_path=0.2
    ):
        super().__init__()

        # ----------------------------------------------------
        # Encoder: pretrained RETFound ViT (MAE)
        # ----------------------------------------------------
        self.encoder = RETFound_mae(
            img_size=img_size,
            num_classes=num_classes,
            drop_path_rate=drop_path,
            global_pool=False  # keep token sequence
        )

        # ----------------------------------------------------
        # Decoder head for pixel prediction
        # ----------------------------------------------------
        self.seg_head = SegmentationHead(
            hidden_dim,
            num_classes,
            img_size,
            patch_size
        )

    def forward(self, x):
        """
        Forward pass:
        1. Patch embedding
        2. Add CLS token
        3. Positional embedding
        4. Transformer blocks
        5. Decoder head
        """

        B = x.size(0)

        # ----- Patch embedding -----
        x = self.encoder.patch_embed(x)

        # ----- Add CLS token -----
        cls = self.encoder.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)

        # ----- Positional encoding -----
        x = x + self.encoder.pos_embed
        x = self.encoder.pos_drop(x)

        # ----- Transformer encoder blocks -----
        for blk in self.encoder.blocks:
            x = blk(x)

        # ----- Final normalization -----
        x = self.encoder.norm(x)

        # ----- Remove CLS token & decode -----
        return self.seg_head(x[:, 1:])
