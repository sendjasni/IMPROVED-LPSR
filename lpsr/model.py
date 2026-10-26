# lpsr/model.py
"""Multi-frame plate super-resolution network (MFSR, x2 or x4), self-contained copy for the v0.2 release."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class CALayer(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.body = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // r, 1), nn.ReLU(True),
                                  nn.Conv2d(c // r, c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.body(x)


class RCAB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(True), nn.Conv2d(c, c, 3, 1, 1), CALayer(c))

    def forward(self, x):
        return x + self.body(x)


class ResidualGroup(nn.Module):
    def __init__(self, c, n_blocks):
        super().__init__()
        self.body = nn.Sequential(*[RCAB(c) for _ in range(n_blocks)], nn.Conv2d(c, c, 3, 1, 1))

    def forward(self, x):
        return x + self.body(x)


class DeformAlign(nn.Module):
    def __init__(self, c, groups=8):
        super().__init__()
        self.offset = nn.Sequential(nn.Conv2d(2 * c, c, 3, 1, 1), nn.LeakyReLU(0.1, True),
                                    nn.Conv2d(c, c, 3, 1, 1), nn.LeakyReLU(0.1, True),
                                    nn.Conv2d(c, 2 * 9 * groups, 3, 1, 1))
        self.dcn = DeformConv2d(c, c, 3, padding=1)

    def forward(self, f, f_ref):
        off = self.offset(torch.cat([f, f_ref], 1))
        return F.leaky_relu(self.dcn(f.float(), off.float()), 0.1).to(f.dtype)


class MFSR(nn.Module):
    """seq (B, N, 3, H, W) in [0,1], frame 0 = reference -> (B, 3, scale*H, scale*W)."""

    def __init__(self, scale=2, c=64, n_enc=3, n_groups=3, n_blocks=6):
        super().__init__()
        self.scale = scale
        self.enc = nn.Sequential(nn.Conv2d(3, c, 3, 1, 1), *[RCAB(c) for _ in range(n_enc)])
        self.align = DeformAlign(c)
        self.emb_ref, self.emb = nn.Conv2d(c, c, 3, 1, 1), nn.Conv2d(c, c, 3, 1, 1)
        self.fuse = nn.Conv2d(2 * c, c, 1)
        self.body = nn.Sequential(*[ResidualGroup(c, n_blocks) for _ in range(n_groups)], nn.Conv2d(c, c, 3, 1, 1))
        up = []
        for _ in range({2: 1, 4: 2}[scale]):
            up += [nn.Conv2d(c, 4 * c, 3, 1, 1), nn.PixelShuffle(2)]
        self.tail = nn.Sequential(*up, nn.Conv2d(c, 3, 3, 1, 1))

    def forward(self, seq):
        B, N, C, H, W = seq.shape
        f = self.enc(seq.flatten(0, 1)).view(B, N, -1, H, W)
        f_ref = f[:, 0]
        aligned = self.align(f.flatten(0, 1), f_ref.repeat_interleave(N, 0)).view(B, N, -1, H, W)
        e_ref = self.emb_ref(f_ref).unsqueeze(1)
        e = self.emb(aligned.flatten(0, 1)).view(B, N, -1, H, W)
        w = torch.softmax((e * e_ref).sum(2, keepdim=True), dim=1)
        fused = self.fuse(torch.cat([(w * aligned).sum(1), f_ref], 1))
        out = self.tail(fused + self.body(fused))
        return F.interpolate(seq[:, 0], scale_factor=self.scale, mode='bicubic', align_corners=False) + out
