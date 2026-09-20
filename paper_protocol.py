"""Shared attack superset + metrics for the public-baseline comparison.

One attack operator set and one set of metric definitions, shared by our method
and by every open-source baseline we port into the harness:

    Gaussian Shading (CVPR'24)   bit payload
    SFWMark HSTR/HSQR (ICCV'25)  verification + identification (official code)
    RingID (ECCV'24)             handled by its own official script
    MaXsive (MM'25)              handled by its own official script

The attack vocabulary follows the union of the SFWMark evaluation list, the
Tree-Ring/RingID/Gaussian-Shading signal attacks and the rotation family that
this paper is about.  A case is a '+'-joined token list applied left to right,
e.g. "rot75+noise0.05".
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

CASE_GROUPS: Dict[str, List[str]] = {
    "signal": ["bright6.0", "contrast0.5", "jpeg25", "blur5", "noise0.05",
               "bm3d0.1"],
    "signal_extra": ["jpeg75", "blur4", "noise0.1", "median5", "resize0.9",
                     "resize1.25"],
    "rotation": ["rot15", "rot30", "rot45", "rot60", "rot75", "rot90"],
    "crop": ["cc0.5", "cc0.7", "rc0.7"],
    "composed": ["rot+noise0.05", "rot75+noise0.05", "rot75+noise0.1",
                 "rot75+jpeg75", "rot45+resize1.25", "rot30+blur4"],
}

ALL_CASES: List[str] = ["clean"]
for _g in CASE_GROUPS.values():
    for _c in _g:
        if _c not in ALL_CASES:
            ALL_CASES.append(_c)


def expand_cases(spec: str) -> List[str]:
    out: List[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part in CASE_GROUPS:
            chosen = CASE_GROUPS[part]
        elif ":" in part:
            grp, _, items = part.partition(":")
            pool = CASE_GROUPS.get(grp, ALL_CASES)
            chosen = [c for c in pool if c in items.split("+")]
        else:
            chosen = [part]
        for c in chosen:
            if c not in out:
                out.append(c)
    return out


@dataclass
class AttackCtx:
    device: str = "cuda"
    resolution: int = 512
    regen: Optional[Callable[[Image.Image, int], Image.Image]] = None
    _vaeb: object = field(default=None, repr=False)
    _vaec: object = field(default=None, repr=False)

    def _codec(self, which: str):
        key = "_vaeb" if which == "vaeb" else "_vaec"
        model = getattr(self, key)
        if model is None:
            from compressai.zoo import bmshj2018_hyperprior, cheng2020_anchor

            factory = bmshj2018_hyperprior if which == "vaeb" else cheng2020_anchor
            model = factory(quality=3, pretrained=True).to(self.device).eval()
            setattr(self, key, model)
        return model

    def apply_codec(self, img: Image.Image, which: str) -> Image.Image:
        import torch
        import torchvision.transforms as T

        model = self._codec(which)
        tf = T.Compose([T.Resize((self.resolution, self.resolution)), T.ToTensor()])
        with torch.no_grad():
            x = tf(img).unsqueeze(0).to(self.device)
            enc = model.compress(x)
            dec = model.decompress(enc["strings"], enc["shape"])
        return T.ToPILImage()(dec["x_hat"].squeeze().clamp(0, 1).cpu())


def _rotate(img, deg):
    return img.rotate(deg, resample=Image.BILINEAR, fillcolor=(0, 0, 0))


def _noise(img, sigma, rng):
    arr = np.asarray(img).astype(np.float64)
    noise = rng.normal(0.0, sigma, arr.shape) * 255.0
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def _jpeg(img, quality):
    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert(img.mode)


def _blur(img, radius):
    from PIL import ImageFilter

    return img.filter(ImageFilter.GaussianBlur(radius))


def _median(img, k):
    from PIL import ImageFilter

    return img.filter(ImageFilter.MedianFilter(size=k))


def _brightness(img, factor):
    from PIL import ImageEnhance

    return ImageEnhance.Brightness(img).enhance(factor)


def _contrast(img, factor):
    from PIL import ImageEnhance

    return ImageEnhance.Contrast(img).enhance(factor)


def _resize(img, ratio):
    w, h = img.size
    return img.resize((max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))),
                      Image.LANCZOS)


def _center_crop(img, area_ratio, fill=0):
    """SFWMark 'CC': centre crop, then zero-pad back to the original size."""
    w, h = img.size
    n = int(w * math.sqrt(area_ratio))
    left, top = (w - n) // 2, (h - n) // 2
    crop = img.crop((left, top, left + n, top + n))
    canvas = Image.new(img.mode, (w, h), fill)
    canvas.paste(crop, (left, top))
    return canvas


def _random_crop_original_position(img, area_ratio, rng, fill=0):
    """SFWMark 'RC': random crop pasted back at its original position."""
    w, h = img.size
    n = int(w * math.sqrt(area_ratio))
    left = random.Random(int(rng.randint(0, 2 ** 31 - 1))).randint(0, max(0, w - n))
    top = random.Random(int(rng.randint(0, 2 ** 31 - 1))).randint(0, max(0, h - n))
    crop = img.crop((left, top, left + n, top + n))
    canvas = Image.new(img.mode, (w, h), fill)
    canvas.paste(crop, (left, top))
    return canvas


def _bm3d(img, sigma):
    from bm3d import bm3d_rgb

    arr = np.asarray(img).astype(np.float64) / 255.0
    out = np.clip(bm3d_rgb(arr, sigma), 0, 1)
    return Image.fromarray((out * 255.0).astype(np.uint8))


def apply_token(img, token, theta, rng, ctx: AttackCtx):
    if token in ("clean", ""):
        return img
    if token == "rot":
        return _rotate(img, theta)
    m = re.fullmatch(r"rot(\d+(?:\.\d+)?)", token)
    if m:
        return _rotate(img, float(m.group(1)))
    if token == "noise":
        return _noise(img, 0.05, rng)
    m = re.fullmatch(r"noise(\d+(?:\.\d+)?)", token)
    if m:
        return _noise(img, float(m.group(1)), rng)
    m = re.fullmatch(r"jpeg(\d+)", token)
    if m:
        return _jpeg(img, int(m.group(1)))
    m = re.fullmatch(r"blur(\d+)", token)
    if m:
        return _blur(img, int(m.group(1)))
    m = re.fullmatch(r"median(\d+)", token)
    if m:
        return _median(img, int(m.group(1)))
    m = re.fullmatch(r"(?:bright|brightness)(\d+(?:\.\d+)?)", token)
    if m:
        return _brightness(img, float(m.group(1)))
    m = re.fullmatch(r"contrast(\d+(?:\.\d+)?)", token)
    if m:
        return _contrast(img, float(m.group(1)))
    m = re.fullmatch(r"resize(\d+(?:\.\d+)?)", token)
    if m:
        raw = float(m.group(1))
        return _resize(img, raw / 100.0 if raw >= 10 else raw)
    m = re.fullmatch(r"cc(\d+(?:\.\d+)?)", token)
    if m:
        return _center_crop(img, float(m.group(1)))
    m = re.fullmatch(r"rc(\d+(?:\.\d+)?)", token)
    if m:
        return _random_crop_original_position(img, float(m.group(1)), rng)
    m = re.fullmatch(r"bm3d(\d+(?:\.\d+)?)", token)
    if m:
        return _bm3d(img, float(m.group(1)))
    m = re.fullmatch(r"vaeb(\d+)", token)
    if m:
        return ctx.apply_codec(img, "vaeb")
    m = re.fullmatch(r"vaec(\d+)", token)
    if m:
        return ctx.apply_codec(img, "vaec")
    m = re.fullmatch(r"regen(\d+)", token)
    if m:
        if ctx.regen is None:
            raise RuntimeError("regen token needs AttackCtx.regen")
        return ctx.regen(img, int(m.group(1)))
    raise ValueError(f"unknown attack token: {token!r}")


def apply_case(img, case: str, theta: float, rng, ctx: AttackCtx):
    for token in case.split("+"):
        img = apply_token(img, token, theta, rng, ctx)
    return img


def case_seed(base_seed: int, case: str) -> int:
    cid = sum(ord(c) * (k + 1) for k, c in enumerate(case))
    return base_seed * 1000 + cid


def detection_metrics(pos: Sequence[float], neg: Sequence[float],
                      fprs: Sequence[float] = (0.01, 1e-3, 1e-6)) -> Dict[str, float]:
    """Presence detection from positive/negative scores (higher = watermarked)."""
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc as _auc

    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    s = np.concatenate([neg, pos])
    fpr, tpr, _ = roc_curve(y, s)
    out = {"auc": float(_auc(fpr, tpr)),
           "max_acc": float(1.0 - np.min((fpr + (1.0 - tpr)) / 2.0)),
           "pos_mean": float(pos.mean()), "neg_mean": float(neg.mean()),
           "n_pos": int(len(pos)), "n_neg": int(len(neg))}
    for f in fprs:
        idx = np.where(fpr <= f)[0]
        out[f"tpr_at_fpr_{f:g}"] = float(tpr[idx[-1]]) if len(idx) else 0.0
    return out


def bit_metrics(true_bits, pred_bits) -> Dict[str, float]:
    t = np.asarray(true_bits).reshape(-1)
    p = np.asarray(pred_bits).reshape(-1)
    ok = (t == p)
    return {"bit_acc": float(ok.mean()), "perfect": float(ok.all()),
            "n_bits": int(t.size), "ber": float(1.0 - ok.mean())}


# --------------------------------------------------------------------------- #
# P0 additions (additive; the attack vocabulary above is unchanged)
#
# The pilot reported point estimates from a single N=50 batch.  The P0 gate
# needs (i) a detection threshold fitted on images the test statistic never
# touches, (ii) an angle error defined modulo the carrier's 180-deg symmetry,
# and (iii) a single protocol version string stamped into every JSON blob.
# The implementations live in p0_common.py and are re-exported here so that
# the shared protocol stays the one place a reader has to look at.
# --------------------------------------------------------------------------- #
from p0_common import (PROTOCOL_VERSION, SplitPlan, align_error,  # noqa: E402
                       calibrate_threshold, case_rotation_angle, rate_above,
                       signed_error, wilson)

__all__ = [
    "CASE_GROUPS", "ALL_CASES", "expand_cases", "AttackCtx", "apply_token",
    "apply_case", "case_seed", "detection_metrics", "bit_metrics",
    "PROTOCOL_VERSION", "SplitPlan", "align_error", "signed_error", "wilson",
    "calibrate_threshold", "rate_above", "case_rotation_angle",
]
