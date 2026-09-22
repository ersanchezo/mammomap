#Inference Script
# ==========================================
# 1. IMPORTS
# ==========================================
import os, glob, math  # Added math here!
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    NormalizeIntensityd, EnsureTyped
)
from monai.inferers import sliding_window_inference
from timm.layers import SwiGLUPacked

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class ATMHead(nn.Module):
    def __init__(self, embed_dim: int = 512, num_classes: int = 150, num_heads: int = 8, mlp_ratio: float = 4.0, use_self_attn: bool = True, dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_self_attn = use_self_attn

        self.query_embed = nn.Parameter(torch.zeros(1, num_classes, embed_dim))

        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(embed_dim)
            self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

        self.mask_proj = nn.Linear(embed_dim, embed_dim)
        self.cls_head = nn.Linear(embed_dim, 1)

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

        if prev_queries is None:
            queries = self.query_embed.expand(B, -1, -1)
        else:
            queries = prev_queries

        if self.use_self_attn:
            attn_out, _ = self.self_attn(queries, queries, queries)
            queries = queries + self.dropout1(attn_out)
            queries = self.norm1(queries)

        attn_out, attn_weights = self.cross_attn(query=queries, key=memory, value=memory)
        queries = queries + self.dropout2(attn_out)
        queries = self.norm2(queries)

        ffn_out = self.ffn(queries)
        queries = queries + ffn_out
        queries = self.norm3(queries)

        mask_embed = self.mask_proj(queries)
        mask_logits = torch.einsum("bnc,blc->bnl", mask_embed, memory)
        mask_logits = mask_logits.view(B, self.num_classes, H, W)
        cls_logits = self.cls_head(queries).squeeze(-1)

        return mask_logits, cls_logits, queries


class CascadedATMDecoder(nn.Module):
    def __init__(self, backbone_dim: int = 1280, embed_dim: int = 512, num_classes: int = 150, num_stages: int = 3, num_heads: int = 8, use_self_attn: bool = True, dropout: float = 0.0):
        super().__init__()
        self.num_stages = num_stages
        self.embed_dim = embed_dim

        self.feature_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            )
            for _ in range(num_stages)
        ])

        self.atm_heads = nn.ModuleList([
            ATMHead(
                embed_dim=embed_dim,
                num_classes=num_classes,
                num_heads=num_heads,
                use_self_attn=use_self_attn,
                dropout=dropout
            )
            for _ in range(num_stages)
        ])

    def forward(self, feature_list):
        queries = None
        aux_outputs = []

        for stage_idx, (features, proj, atm_head) in enumerate(zip(feature_list, self.feature_projs, self.atm_heads)):
            B, L, C = features.shape
            H = W = int(math.sqrt(L))

            features_proj = proj(features)
            features_spatial = features_proj.transpose(1, 2).reshape(B, self.embed_dim, H, W)

            mask_logits, cls_logits, queries = atm_head(features_spatial, queries)

            if self.training and stage_idx < self.num_stages - 1:
                aux_outputs.append((mask_logits, cls_logits))

        return mask_logits, cls_logits, aux_outputs

class VirchowSegViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = 14
        self.num_classes = config["num_classes"]
        self.hook_indices = [15, 23, 31]
        self.hidden_states = {}

        self.backbone = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )
        self.backbone.set_grad_checkpointing(True)

        target_regex = r".*blocks\.(15|23|31)\.(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)"
        peft_config = LoraConfig(
            r=config["lora_rank"],
            lora_alpha=config["lora_rank"]*2,
            target_modules=target_regex,
            lora_dropout=0.1,
            bias="lora_only"
        )
        self.backbone = get_peft_model(self.backbone, peft_config)

        self.decoder = CascadedATMDecoder(
            backbone_dim=1280, embed_dim=512, num_classes=config["num_classes"], num_stages=3, num_heads=8, use_self_attn=True
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

        outputs = []
        h_feat = H_img // self.patch_size
        w_feat = W_img // self.patch_size
        num_spatial = h_feat * w_feat

        for idx in self.hook_indices:
            feat = self.hidden_states[idx]
            feat = feat[:, -num_spatial:, :]
            outputs.append(feat)

        mask_logits, cls_logits, _ = self.decoder(outputs)
        logits = F.interpolate(mask_logits, size=(H_img, W_img), mode="bilinear", align_corners=False)
        return logits

# ==========================================
# 3. CONFIGURATION (Updated for Modal)
# ==========================================
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "img_size": (448, 448),
    "num_classes": 4,
    "lora_rank": 16,
    "backbone_name": "paige-ai/virchow2",
    "sw_batch_size": 2,
    "overlap": 0.5
}

# Paths pointing directly to your mounted Modal volume
MODEL_BASE = "/weights"
ADAPTER_DIR = os.path.join(MODEL_BASE, "checkpoint_epoch_18_adapter")
DECODER_PATH = os.path.join(MODEL_BASE, "checkpoint_epoch_18_decoder.pth")

# ==========================================
# 4. LOAD MODEL
# ==========================================
def load_model():
    device = CONFIG["device"]
    model = VirchowSegViT(CONFIG).to(device)

    print(f"Loading LoRA adapter from {ADAPTER_DIR}...")
    if hasattr(model.backbone, 'load_adapter'):
        model.backbone.load_adapter(ADAPTER_DIR, adapter_name="default")
        model.backbone.set_adapter("default")
    else:
        model.backbone = PeftModel.from_pretrained(model.backbone, ADAPTER_DIR)

    print(f"Loading decoder from {DECODER_PATH}...")
    state_dict = torch.load(DECODER_PATH, map_location=device)
    model.decoder.load_state_dict(state_dict)

    model.eval()
    return model

# ==========================================
# 5. PREPROCESSING & INFERENCE
# ==========================================
transform = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),
    NormalizeIntensityd(keys=["image"], subtrahend=[0.485, 0.456, 0.406], divisor=[0.229, 0.224, 0.225], channel_wise=True),
    EnsureTyped(keys=["image"], dtype=torch.float32),
])

def process_image(model, img_path):
    data = transform({"image": str(img_path)})
    input_tensor = data["image"].unsqueeze(0).to(CONFIG["device"])

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=CONFIG["device"] == "cuda"):
            logits = sliding_window_inference(
                inputs=input_tensor,
                roi_size=CONFIG["img_size"],
                sw_batch_size=CONFIG["sw_batch_size"],
                predictor=model,
                overlap=CONFIG["overlap"],
                mode="gaussian"
            )
            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8).T
            preds = preds + 1
    return preds