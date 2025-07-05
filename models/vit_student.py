import torch
import torch.nn as nn
import timm

class V18Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model("convnext_tiny", pretrained=True, in_chans=1, num_classes=0, features_only=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.meta_head = nn.Sequential(nn.Linear(cfg["META_DIM"], 64), nn.ReLU(), nn.Linear(64, 64))
        self.head = nn.Sequential(
            nn.Linear(self.backbone[-1].num_chs + 64, 512),
            nn.ReLU(),
            nn.Linear(512, cfg["NUM_BINS"] * 2)
        )
        self.cfg = cfg

    def forward(self, x_airs, x_fgs=None, x_meta=None):
        x = self.backbone[0](x_airs)
        for blk in self.backbone[1:]:
            x = blk(x)
        x = self.pool(x).flatten(1)
        meta = self.meta_head(x_meta)
        fused = torch.cat([x, meta], dim=1)
        out = self.head(fused).view(-1, self.cfg["NUM_BINS"], 2)
        return out