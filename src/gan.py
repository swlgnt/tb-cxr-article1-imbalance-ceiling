"""GAN dasar (DCGAN, loss non-saturating) untuk sintesis citra kelas TB.

Pilihan desain (tulis di bagian Metode):
  * Arsitektur DCGAN (Radford et al., 2016): Generator upsampling + BatchNorm + ReLU,
    Discriminator konvolusi stride-2 + LeakyReLU(0.2). Upsampling nearest + konvolusi 3x3
    dipakai sebagai pengganti transposed convolution untuk mengurangi artefak checkerboard
    (Odena et al., 2016) yang bisa menjadi "jalan pintas" bagi classifier.
  * Loss non-saturating BCE (Goodfellow et al., 2014). Loss Wasserstein SENGAJA tidak dipakai
    karena merupakan komponen CGAN teroptimasi (artikel Scopus) -> ablation tetap bersih.
  * DiffAugment (Zhao et al., 2020) mencegah discriminator overfit pada data sedikit.
  * EMA bobot generator (Karras et al., 2018) untuk sampel yang lebih stabil.
  DiffAugment dan EMA adalah protokol pelatihan bersama untuk semua varian GAN berikutnya
  (CGAN, CGAN teroptimasi) sehingga perbandingan antar-generator adil.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _n_blocks(img_size: int) -> int:
    n = int(math.log2(img_size)) - 2  # 4x4 -> img_size
    if 2 ** (n + 2) != img_size or n < 2:
        raise ValueError(f"gan.img_size harus pangkat 2 dan >= 16 (diberikan {img_size})")
    return n


class Generator(nn.Module):
    def __init__(self, z_dim: int = 128, img_size: int = 256, base_ch: int = 32, max_ch: int = 512,
                 upsample: str = "nearest"):
        super().__init__()
        n = _n_blocks(img_size)
        chs = [min(base_ch * 2 ** (n - i), max_ch) for i in range(n + 1)]  # dari 4x4 ke img_size
        self.z_dim = z_dim
        self.fc = nn.Sequential(nn.Linear(z_dim, chs[0] * 16, bias=False), nn.BatchNorm1d(chs[0] * 16), nn.ReLU(True))
        self.c0 = chs[0]
        blocks = []
        for i in range(n):
            if upsample == "transpose":
                up = nn.ConvTranspose2d(chs[i], chs[i + 1], 4, 2, 1, bias=False)
            else:
                up = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                   nn.Conv2d(chs[i], chs[i + 1], 3, 1, 1, bias=False))
            blocks += [up, nn.BatchNorm2d(chs[i + 1]), nn.ReLU(True)]
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Sequential(nn.Conv2d(chs[-1], 1, 3, 1, 1), nn.Tanh())

    def forward(self, z):
        x = self.fc(z).view(z.size(0), self.c0, 4, 4)
        return self.out(self.blocks(x))


class Discriminator(nn.Module):
    def __init__(self, img_size: int = 256, base_ch: int = 32, max_ch: int = 512):
        super().__init__()
        n = _n_blocks(img_size)
        chs = [min(base_ch * 2 ** i, max_ch) for i in range(n + 1)]
        layers = [nn.Conv2d(1, chs[0], 3, 1, 1), nn.LeakyReLU(0.2, True)]
        for i in range(n):
            layers += [nn.Conv2d(chs[i], chs[i + 1], 4, 2, 1, bias=False)]
            if i > 0:  # DCGAN: tanpa BN pada blok pertama
                layers += [nn.BatchNorm2d(chs[i + 1])]
            layers += [nn.LeakyReLU(0.2, True)]
        self.features = nn.Sequential(*layers)
        self.out = nn.Conv2d(chs[-1], 1, 4, 1, 0)  # 4x4 -> 1 logit

    def forward(self, x):
        return self.out(self.features(x)).view(-1)


def init_dcgan_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def gen_batch_size(img_size: int, override: int | None = None) -> int:
    """Ukuran potongan saat MEMBANGKITKAN citra (sampel, evaluasi FID, ekspor). Memori aktivasi
    naik kuadratik dengan resolusi: 64 citra pada 256 px setara ~16 citra pada 512 px."""
    if override:
        return int(override)
    return int(max(2, min(64, 64 * (256 / img_size) ** 2)))


# ----------------------------------------------------------------------------- DiffAugment
def _brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * 0.4


def _contrast(x):
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 0.6 + 0.7) + mean


def _translation(x, ratio=0.125):
    b, _, h, w = x.shape
    sh, sw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    tx = torch.randint(-sh, sh + 1, (b, 1, 1), device=x.device)
    ty = torch.randint(-sw, sw + 1, (b, 1, 1), device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(b, device=x.device), torch.arange(h, device=x.device),
                                torch.arange(w, device=x.device), indexing="ij")
    gx = torch.clamp(gx + tx + 1, 0, h + 1)
    gy = torch.clamp(gy + ty + 1, 0, w + 1)
    xp = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return xp.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)


def _cutout(x, ratio=0.25):
    b, _, h, w = x.shape
    ch, cw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    ox = torch.randint(0, h + (1 - ch % 2), (b, 1, 1), device=x.device)
    oy = torch.randint(0, w + (1 - cw % 2), (b, 1, 1), device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(b, device=x.device), torch.arange(ch, device=x.device),
                                torch.arange(cw, device=x.device), indexing="ij")
    gx = torch.clamp(gx + ox - ch // 2, 0, h - 1)
    gy = torch.clamp(gy + oy - cw // 2, 0, w - 1)
    mask = torch.ones(b, h, w, dtype=x.dtype, device=x.device)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)


_POLICIES = {"brightness": _brightness, "contrast": _contrast, "translation": _translation, "cutout": _cutout}


def diff_augment(x, policy: str):
    """Augmentasi terdiferensiasi; diterapkan IDENTIK pada citra asli & palsu sebelum D."""
    if not policy:
        return x
    for p in policy.split(","):
        x = _POLICIES[p.strip()](x)
    return x


# ----------------------------------------------------------------------------- EMA
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)
