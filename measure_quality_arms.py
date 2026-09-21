"""Paired PSNR / SSIM / LPIPS for a set of image folders against a reference.

Every arm folder must hold the *same* file names as the reference folder, i.e.
the arms are the watermarked generations of the same prompts and the same
starting latents.  That is what makes the numbers paired (and what makes a
per-sample difference meaningful); the script refuses to compare folders whose
file-name sets do not match.

Motivation: the quality arms disagree depending on the metric.  On the COCO-5k
subset used here, against the no-watermark image of the same latent:

    arm                     paired PSNR    paired CLIP delta
    ours (B8 dual, eta=1e4)   ~12.3 dB        -0.0128
    METR R8                   ~ 9.8 dB        -0.0341
    MaXsive                   ~ 9.4 dB        -0.0008
    GS-256                    ~ 9.2 dB        +0.0004

PSNR and CLIP therefore rank the methods in *opposite* orders, so neither one
alone answers "does this look acceptable".  LPIPS is the metric that correlates
best with the visual comparison, which is why it is reported here next to the
two extremes.

    python measure_quality_arms.py --ref runs/quality_design/images/no_w \
        --arms ours=runs/quality_design/images/searched,\
maxsive=runs/quality_maxsive/images/maxsive,\
gs=runs/quality_gs/images/gs,\
metr_r8=runs/quality_design/images/metr_r8 \
        --lpips --out_json results/quality_arms.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def list_images(d: str) -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*"))):
        if os.path.splitext(p)[1].lower() in (".png", ".jpg", ".jpeg", ".webp"):
            out[os.path.basename(p)] = p
    return out


def metrics_for_pair(a_path: str, b_path: str, lpips_fn=None, device="cpu"):
    from PIL import Image

    a = Image.open(a_path).convert("RGB")
    b = Image.open(b_path).convert("RGB")
    x = np.asarray(a).astype(np.float64) / 255.0
    y = np.asarray(b).astype(np.float64) / 255.0
    mse = float(np.mean((x - y) ** 2))
    out = {"mse": mse, "psnr": float(10.0 * np.log10(1.0 / max(mse, 1e-12))),
           "ssim": float("nan"), "lpips": float("nan")}
    try:
        from skimage.metrics import structural_similarity

        out["ssim"] = float(structural_similarity(x, y, channel_axis=-1,
                                                  data_range=1.0))
    except Exception:
        pass
    if lpips_fn is not None:
        import torch
        from torchvision.transforms.functional import to_tensor

        dev = next(lpips_fn.parameters()).device
        with torch.no_grad():
            out["lpips"] = float(lpips_fn(
                to_tensor(a).unsqueeze(0).to(dev) * 2 - 1,
                to_tensor(b).unsqueeze(0).to(dev) * 2 - 1).item())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference folder (no watermark)")
    ap.add_argument("--arms", required=True,
                    help="comma list of name=folder entries")
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    ref = list_images(args.ref)
    if not ref:
        raise SystemExit(f"no images in reference folder {args.ref}")

    lpips_fn = None
    if args.lpips:
        import lpips
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        lpips_fn = lpips.LPIPS(net="alex").to(dev).eval()

    rows = []
    for spec in [s for s in args.arms.split(",") if s]:
        name, _, folder = spec.partition("=")
        arm = list_images(folder)
        common = sorted(set(ref) & set(arm))
        if not common:
            print(f"[skip] {name}: no shared file names with the reference")
            continue
        if len(common) != len(arm) or len(common) != len(ref):
            print(f"[warn] {name}: comparing the {len(common)} shared files only "
                  f"(ref has {len(ref)}, arm has {len(arm)})")
        vals = [metrics_for_pair(ref[k], arm[k], lpips_fn=lpips_fn)
                for k in common]
        agg = {f: float(np.nanmean([v[f] for v in vals])) for f in
               ("psnr", "ssim", "lpips")}
        agg.update({"arm": name, "n": len(common), "folder": folder})
        rows.append(agg)

    rows.sort(key=lambda r: -(r["psnr"] if np.isfinite(r["psnr"]) else -99))
    print(f"\nreference: {args.ref}  ({len(ref)} images)\n")
    print(f"{'arm':16s} {'n':>5s} {'PSNR up':>9s} {'SSIM up':>9s} {'LPIPS dn':>9s}")
    for r in rows:
        print(f"{r['arm']:16s} {r['n']:5d} {r['psnr']:9.2f} "
              f"{r['ssim']:9.4f} {r['lpips']:9.4f}")
    print("\n(PSNR/SSIM: higher is better.  LPIPS: lower is better.  "
          "All values are paired per image against the reference.)")

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)),
                    exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"reference": args.ref, "rows": rows}, f, indent=2)
        print(f"-> {args.out_json}")


if __name__ == "__main__":
    main()
