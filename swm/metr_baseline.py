"""METR baseline (paper protocol R=10, S=100).

Mirrors `metr.optim_utils` but fixes two issues found during reproduction:
  * the official `circle_mask` flips the y-axis (`y = y[::-1]`), which shifts
    the disk off the FFT DC pixel and cancels most of the signal after the
    `.real` truncation -- we use the un-flipped, Hermitian-centred disk;
  * annulus masks must be boolean before torch indexing (the official code
    subtracts booleans into int64 and indexes with 0/1 arrays).

Encoding protocol:
  * concentric disks are assigned from R down to 1, so annulus (r-1, r] carries
    bit r (bit 1 at the innermost disk);
  * message is written to channel w_channel=3, other channels receive +1 on the
    same support (as in the official implementation);
  * decoding averages the real FFT value over each annulus and takes its sign.
"""

from __future__ import annotations

import numpy as np
import torch


def circle_mask(size: int = 64, r: int = 10) -> np.ndarray:
    """Hermitian-centred circle mask (center at the fftshifted DC pixel)."""
    x0 = y0 = size // 2
    y, x = np.ogrid[:size, :size]
    return ((x - x0) ** 2 + (y - y0) ** 2) <= r**2


def disk_support(radius: int) -> np.ndarray:
    return circle_mask(size=64, r=radius)


def annulus_mask(radius: int) -> np.ndarray:
    if radius == 1:
        return disk_support(radius)
    ann = disk_support(radius).astype(np.int8) - disk_support(radius - 1).astype(np.int8)
    return ann.astype(bool)


def embed_metr(
    z_T: torch.Tensor,
    bits: np.ndarray,
    radius: int = 10,
    scaler: float = 100.0,
    channel: int = 3,
) -> torch.Tensor:
    """bits: length `radius`, 1/0. Official concentric-ring embedding."""
    assert len(bits) == radius
    z_fft = torch.fft.fftshift(torch.fft.fft2(z_T.float()), dim=(-1, -2))
    z_fft = z_fft.clone()
    n_ch = z_fft.shape[1]
    for i in range(radius, 0, -1):
        mask = torch.tensor(disk_support(i), device=z_fft.device, dtype=torch.bool)
        for c in range(n_ch):
            if c == channel:
                val = float(scaler if bits[i - 1] == 1 else -scaler)
            else:
                val = 1.0
            z_fft[:, c, mask] = val
    z_w = torch.fft.ifft2(torch.fft.ifftshift(z_fft, dim=(-1, -2))).real
    return z_w.to(dtype=z_T.dtype)


def decode_metr(
    z_hat: torch.Tensor, radius: int = 10, channel: int = 3
) -> list[int]:
    """Predict radius bits from the inverted latent (official sign per annulus)."""
    z_fft = torch.fft.fftshift(torch.fft.fft2(z_hat.float()), dim=(-1, -2))
    pred = []
    for i in range(radius, 0, -1):
        mask = annulus_mask(i)
        value = z_fft[0, channel, mask].real.mean()
        pred.append(1 if value > 0 else 0)
    return pred[::-1]
