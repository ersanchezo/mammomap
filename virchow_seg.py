"""
virchow_seg.py — Virchow2 + Cascaded ATM segmentation
Encoder : Virchow2 (Paige AI, ViT-H/14, 1280-dim) with LoRA adapters
Decoder : CascadedATMDecoder  (3-stage ATM heads)
Classes : 4  (tumor · stroma · lymphocyte · necrosis)

Fixes vs. original inference script:
  - Added missing `import math` (used in CascadedATMDecoder.forward)
  - Weight paths adapted from Google Drive → Modal Volume (/weights/)
  - process_image accepts bytes instead of a file path
  - Output converted to PathVision regions dict for the API response
  - Class map is explicit — verify it matches your training labels
"""

import io
import math          # ← was missing in original script
import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2
from timm.layers import SwiGLUPacked
from peft import LoraConfig, get_peft_model, PeftModel
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    NormalizeIntensityd, EnsureTyped,
)
from monai.inferers import sliding_window_inference
from PIL import Image

# ─── Class label map (1-indexed, verify against your training config) ──────────
# Original script does: preds = argmax + 1  →  classes are 1-4
CLASS_LABELS: dict[int, str] = {
    1: "tumor",
    2: "stroma",
    3: "lymphocyte",
    4: "necrosis",
}

# Weight paths inside the Modal Volume (mounted at /weights)
ADAPTER_DIR   = "/weights/seg_adapter"
DECODER_PATH  = "/weights/seg_decoder.pth"

# ─── Model config (must match training) ───────────────────────────────────────
SEG_CONFIG = {
    "img_size":      (448, 448),
    "num_classes":   4,
    "lora_rank":     16,
    "sw_batch_size": 2,
    "overlap":       0.5,
    "hook_indices":  [15, 23, 31],
}

# ─── ATMHead ─────────────────────────────────────────────────────────────────
class ATMHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_classes: int = 150,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        use_self_attn: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim   = embed_dim
        self.use_self_attn = use_self_attn

        self.query_embed = nn.Parameter(torch.zeros(1, num_classes, embed_dim))

        if use_self_attn:
            self.self_attn  = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.norm1      = nn.LayerNorm(embed_dim)
            self.dropout1   = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2      = nn.LayerNorm(embed_dim)
        self.dropout2   = nn.Dropout(dropout)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim), nn.Dropout(dropout),
        )
        self.norm3    = nn.LayerNorm(embed_dim)
        self.mask_proj = nn.Linear(embed_dim, embed_dim)
        self.cls_head  = nn.Linear(embed_dim, 1)

        nn.init.trunc_normal_(self.query_embed, std=0.02)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, features, prev_queries=None):
        B, C, H, W = features.shape
        memory = features.flatten(2).transpose(1, 2)

        queries = self.query_embed.expand(B, -1, -1) if prev_queries is None else prev_queries

        if self.use_self_attn:
            attn_out, _ = self.self_attn(queries, queries, queries)
            queries = self.norm1(queries + self.dropout1(attn_out))

        attn_out, attn_weights = self.cross_attn(query=queries, key=memory, value=memory)
        queries = self.norm2(queries + self.dropout2(attn_out))
        queries = self.norm3(queries + self.ffn(queries))

        mask_embed  = self.mask_proj(queries)
        mask_logits = torch.einsum("bnc,blc->bnl", mask_embed, memory).view(B, self.num_classes, H, W)
        cls_logits  = self.cls_head(queries).squeeze(-1)

        return mask_logits, cls_logits, queries


# ─── CascadedATMDecoder ───────────────────────────────────────────────────────
class CascadedATMDecoder(nn.Module):
    def __init__(
        self,
        backbone_dim: int = 1280,
        embed_dim: int    = 512,
        num_classes: int  = 150,
        num_stages: int   = 3,
        num_heads: int    = 8,
        use_self_attn: bool = True,
        dropout: float    = 0.0,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.embed_dim  = embed_dim

        self.feature_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(backbone_dim, embed_dim), nn.LayerNorm(embed_dim))
            for _ in range(num_stages)
        ])
        self.atm_heads = nn.ModuleList([
            ATMHead(embed_dim=embed_dim, num_classes=num_classes,
                    num_heads=num_heads, use_self_attn=use_self_attn, dropout=dropout)
            for _ in range(num_stages)
        ])

    def forward(self, feature_list):
        queries     = None
        aux_outputs = []

        for stage_idx, (features, proj, atm_head) in enumerate(
                zip(feature_list, self.feature_projs, self.atm_heads)):
            B, L, C = features.shape
            H = W = int(math.sqrt(L))          # ← math import required here

            features_proj    = proj(features)
            features_spatial = features_proj.transpose(1, 2).reshape(B, self.embed_dim, H, W)

            mask_logits, cls_logits, queries = atm_head(features_spatial, queries)

            if self.training and stage_idx < self.num_stages - 1:
                aux_outputs.append((mask_logits, cls_logits))

        return mask_logits, cls_logits, aux_outputs


# ─── VirchowSegViT ────────────────────────────────────────────────────────────
class VirchowSegViT(nn.Module):
    def __init__(self, config: dict = SEG_CONFIG):
        super().__init__()
        self.patch_size   = 14
        self.num_classes  = config["num_classes"]
        self.hook_indices = config["hook_indices"]
        self.hidden_states: dict = {}

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        self.backbone.set_grad_checkpointing(True)

        # ── LoRA ──────────────────────────────────────────────────────────────
        target_regex = r".*blocks\.(15|23|31)\.(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)"
        peft_config = LoraConfig(
            r=config["lora_rank"],
            lora_alpha=config["lora_rank"] * 2,
            target_modules=target_regex,
            lora_dropout=0.1,
            bias="lora_only",
        )
        self.backbone = get_peft_model(self.backbone, peft_config)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.decoder = CascadedATMDecoder(
            backbone_dim=1280, embed_dim=512,
            num_classes=config["num_classes"],
            num_stages=3, num_heads=8, use_self_attn=True,
        )
        self._register_hooks()

    def _register_hooks(self):
        def get_hook(layer_idx):
            def hook(module, input, output):
                self.hidden_states[layer_idx] = output
            return hook
        for idx in self.hook_indices:
            if idx < len(self.backbone.blocks):
                self.backbone.blocks[idx].register_forward_hook(get_hook(idx))
            else:
                raise ValueError(f"Hook index {idx} exceeds backbone depth {len(self.backbone.blocks)}")

    def forward(self, x):
        B, C, H_img, W_img = x.shape
        self.hidden_states = {}
        _ = self.backbone.forward_features(x)

        h_feat = H_img // self.patch_size
        w_feat = W_img // self.patch_size
        num_spatial = h_feat * w_feat

        outputs = []
        for idx in self.hook_indices:
            feat = self.hidden_states[idx]
            feat = feat[:, -num_spatial:, :]   # strip CLS + register tokens
            outputs.append(feat)

        mask_logits, cls_logits, _ = self.decoder(outputs)
        logits = F.interpolate(mask_logits, size=(H_img, W_img),
                               mode="bilinear", align_corners=False)
        return logits


# ─── MONAI preprocessing transform ───────────────────────────────────────────
_transform = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),
    NormalizeIntensityd(
        keys=["image"],
        subtrahend=[0.485, 0.456, 0.406],
        divisor=[0.229, 0.224, 0.225],
        channel_wise=True,
    ),
    EnsureTyped(keys=["image"], dtype=torch.float32),
])


# ─── Model loading ────────────────────────────────────────────────────────────
_model_cache: VirchowSegViT | None = None

def load_virchow_seg_model(device: str = "cuda") -> VirchowSegViT:
    """
    Load Virchow2 + ATM model. Results are cached in module scope so the
    second call within the same Modal container is instant.

    Expects weights at:
      /weights/seg_adapter/   — LoRA adapter directory
      /weights/seg_decoder.pth — ATM decoder state dict
    (populated by upload_weights.py once, then persisted in the Modal Volume)
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    # Virchow2 is a gated HF model — HF_TOKEN must be in Modal secrets
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)

    print("Building VirchowSegViT…")
    model = VirchowSegViT(SEG_CONFIG).to(device)

    print(f"Loading LoRA adapter from {ADAPTER_DIR}")
    if hasattr(model.backbone, "load_adapter"):
        model.backbone.load_adapter(ADAPTER_DIR, adapter_name="default")
        model.backbone.set_adapter("default")
    else:
        model.backbone = PeftModel.from_pretrained(model.backbone, ADAPTER_DIR)

    print(f"Loading ATM decoder from {DECODER_PATH}")
    state_dict = torch.load(DECODER_PATH, map_location=device)
    model.decoder.load_state_dict(state_dict)

    model.eval()
    _model_cache = model
    print("Model ready.")
    return model


# ─── Inference ────────────────────────────────────────────────────────────────
def run_virchow_seg(model: VirchowSegViT, image_bytes: bytes, device: str = "cuda") -> dict:
    """
    Run Virchow2+ATM segmentation on image bytes.
    Returns a dict compatible with PathVision's segmentation result format.
    """
    # Write bytes to a temp file for MONAI LoadImaged
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        data = _transform({"image": tmp_path})
        input_tensor = data["image"].unsqueeze(0).to(device)
    finally:
        os.unlink(tmp_path)

    H_orig, W_orig = input_tensor.shape[-2], input_tensor.shape[-1]

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = sliding_window_inference(
                inputs        = input_tensor,
                roi_size      = SEG_CONFIG["img_size"],
                sw_batch_size = SEG_CONFIG["sw_batch_size"],
                predictor     = model,
                overlap       = SEG_CONFIG["overlap"],
                mode          = "gaussian",
            )
            # Original transpose fix from inference script preserved
            preds = (
                torch.argmax(logits, dim=1)
                .squeeze(0).cpu().numpy().astype(np.uint8).T
            )
            preds = preds + 1   # shift to 1-indexed classes

    return _preds_to_regions(preds, W_orig, H_orig)


def _preds_to_regions(preds: np.ndarray, W: int, H: int) -> dict:
    """
    Convert a (W, H) class mask to PathVision's regions + metrics dict.
    preds is 1-indexed: 1=tumor, 2=stroma, 3=lymphocyte, 4=necrosis
    (verify CLASS_LABELS matches your training config)
    """
    regions = []
    class_areas: dict[str, float] = {v: 0.0 for v in CLASS_LABELS.values()}
    total_px = W * H

    for cls_idx, cls_name in CLASS_LABELS.items():
        binary = ((preds == cls_idx).astype(np.uint8) * 255)
        class_areas[cls_name] = round(float(binary.sum() / 255 / total_px * 100), 2)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            x_bb, y_bb, bw, bh = cv2.boundingRect(cnt)
            regions.append({
                "type":  cls_name,
                "cx":    round(M["m10"] / M["m00"] / W * 100, 2),
                "cy":    round(M["m01"] / M["m00"] / H * 100, 2),
                "rx":    round(bw / W * 45, 2),
                "ry":    round(bh / H * 45, 2),
                "angle": 0.0,
                "area_pct": round(area / total_px * 100, 2),
            })

    return {
        "regions": regions,
        "metrics": {
            "tumor_purity":   class_areas["tumor"],
            "til_score":      class_areas["lymphocyte"],
            "stroma_ratio":   class_areas["stroma"],
            "necrosis_ratio": class_areas["necrosis"],
        },
        "model": "Virchow2 + CascadedATM",
    }
