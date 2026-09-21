"""Paired visual-quality inspection for SectorSync (real generated images).

The paired numbers (PSNR 12.10 dB, SSIM 0.394, LPIPS 0.569 at the main
operating point B=8, eta=1e4) say that the watermarked image differs from the
un-watermarked image generated from the *same* latent, while the paired CLIP
delta (-0.0148) only says that both still match the prompt semantically.  The
two statements are compatible, and neither answers the question a reader
actually asks: **what does the difference look like?**

This script produces exactly that, from the real pipeline:

    for each sample i:  same prompt, same z_no
        no watermark            -> images/no_wm/sample{i}.png
        B8 eta=5e3 / 1e4 / 2e4  -> images/<cell>/sample{i}.png
    + amplified difference maps against the no-watermark image of the same latent
    + a centre-crop zoom (differences are usually more visible in local texture)
    + a contact sheet and a markdown report with the paired metrics

Nothing here is an attack experiment: no rotation, no inversion.  It is purely
"how much does the picture change", so it needs only generation (one pass per
configuration) and is cheap.

    python run_visual_quality.py --N 4 --etas 0,5e3,1e4,2e4 --clip \
        --out_dir runs/visual_quality

The sheet/zoom assembly is pure PIL and is unit-testable without torch:

    python run_visual_quality.py --selftest
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


# --------------------------------------------------------------------------- #
# pure-PIL layout helpers (no torch; unit-testable)
# --------------------------------------------------------------------------- #
def load_font(size: int = 18):
    """Best-effort TTF, falling back to PIL's bitmap font."""
    from PIL import ImageFont

    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "C:/Windows/Fonts/arial.ttf",
                 "C:/Windows/Fonts/segoeui.ttf"):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def label_strip(width: int, text: str, height: int = 26, font=None,
                bg: Tuple[int, int, int] = (245, 245, 245),
                fg: Tuple[int, int, int] = (20, 20, 20)) -> Image.Image:
    """A one-line caption bar used above each column."""
    font = font or load_font(16)
    strip = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(strip)
    d.text((6, 4), text, fill=fg, font=font)
    return strip


def diff_map(a: Image.Image, b: Image.Image, gain: float = 8.0,
             signed: bool = False) -> Image.Image:
    """Amplified difference image.

    ``signed=False`` -> ``|a - b| * gain`` in grayscale (where did it change);
    ``signed=True``  -> ``(a - b) * gain + 128`` (which direction).
    """
    x = np.asarray(a.convert("RGB")).astype(np.float64)
    y = np.asarray(b.convert("RGB")).astype(np.float64)
    d = x - y
    if signed:
        out = np.clip(d * gain + 128.0, 0, 255)
    else:
        m = np.abs(d).mean(axis=2, keepdims=True)
        out = np.clip(np.repeat(m, 3, axis=2) * gain, 0, 255)
    return Image.fromarray(out.astype(np.uint8))


def center_crop_zoom(img: Image.Image, size: int = 160,
                     zoom: int = 3) -> Image.Image:
    """Centre crop at ``size`` pixels, upscaled by ``zoom`` (nearest)."""
    w, h = img.size
    size = min(size, w, h)
    left, top = (w - size) // 2, (h - size) // 2
    crop = img.convert("RGB").crop((left, top, left + size, top + size))
    return crop.resize((size * zoom, size * zoom), Image.NEAREST)


def contact_sheet(rows: Sequence[Tuple[str, Sequence[Tuple[str, Image.Image]]]],
                  cell: int = 256, pad: int = 8, title: str = "",
                  gain: float = 8.0) -> Image.Image:
    """Assemble the report sheet.

    ``rows`` is a list of ``(row_label, [(column_label, image), ...])``; the
    first column of each row is expected to be the no-watermark reference, and
    the remaining columns get an amplified difference strip underneath them.

    Each sample occupies four bands, in this order, so no caption ever overlaps
    an image:

        | column labels                     |
        | images                            |
        | "diff vs <ref> x<gain>" labels    |
        | difference maps (columns 1..n-1)  |
    """
    font = load_font(16)
    title_font = load_font(20)
    n_cols = max(len(cols) for _, cols in rows)
    strip_h = 26
    block_h = strip_h * 2 + cell * 2         # 2 label strips + 2 image bands
    W = pad + n_cols * (cell + pad)
    H = pad + (60 if title else 0) + len(rows) * (block_h + pad) + pad
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    y0 = pad
    if title:
        d.text((pad, pad + 10), title, fill=(0, 0, 0), font=title_font)
        y0 += 60
    for r, (row_label, cols) in enumerate(rows):
        y = y0 + r * (block_h + pad)
        ref = cols[0][1]
        # band 1: column labels
        for c, (col_label, img) in enumerate(cols):
            x = pad + c * (cell + pad)
            sheet.paste(label_strip(cell, f"{row_label} | {col_label}",
                                    font=font), (x, y))
        # band 2: images
        for c, (col_label, img) in enumerate(cols):
            x = pad + c * (cell + pad)
            sheet.paste(img.convert("RGB").resize((cell, cell), Image.LANCZOS),
                        (x, y + strip_h))
        # band 3 + 4: difference labels and maps
        yd = y + strip_h + cell
        sheet.paste(label_strip(cell, f"diff vs «{cols[0][0]}» x{gain:g}",
                                font=font, bg=(232, 234, 238)), (pad, yd))
        for c in range(1, len(cols)):
            x = pad + c * (cell + pad)
            sheet.paste(label_strip(cell, f"|ref - cell| x{gain:g}",
                                    font=font, bg=(232, 234, 238)), (x, yd))
            if c > 0:
                dm = diff_map(cols[c][1], ref, gain=gain)
                sheet.paste(dm.resize((cell, cell), Image.NEAREST),
                            (x, yd + strip_h))
    return sheet


def _pair_metrics(a: Image.Image, b: Image.Image,
                  lpips_fn=None, clip_ref=None, prompt: str = "",
                  clip=None) -> Dict[str, float]:
    """Paired distortion of ``b`` against reference ``a`` (+ optional CLIP)."""
    x = np.asarray(a.convert("RGB")).astype(np.float64) / 255.0
    y = np.asarray(b.convert("RGB")).astype(np.float64) / 255.0
    mse = float(np.mean((x - y) ** 2))
    out = {"mse": mse,
           "psnr": float(10.0 * np.log10(1.0 / max(mse, 1e-12))),
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
    if clip is not None and clip_ref is not None and prompt:
        from pipeline.optim_utils import measure_similarity

        out["clip"] = float(measure_similarity([b], prompt, clip_ref[0], clip))
    return out


def selftest(out_dir: str = "") -> None:
    """Build a sheet from synthetic images so the layout can be checked without a GPU."""
    rng = np.random.RandomState(0)
    base = Image.fromarray(
        (rng.rand(512, 512, 3) * 60 + 120).astype(np.uint8))
    rows = []
    for i in range(2):
        cols = [("no watermark", base)]
        for eta in (5e3, 1e4, 2e4):
            noisy = np.asarray(base).astype(np.float64) + \
                rng.normal(0, eta / 2e6 * 255, (512, 512, 3))
            cols.append((f"eta={eta:g}", Image.fromarray(
                np.clip(noisy, 0, 255).astype(np.uint8))))
        rows.append((f"sample {i}", cols))
    sheet = contact_sheet(rows, title="SELFTEST (synthetic, not a result)")
    out_dir = out_dir or os.path.join("runs", "visual_quality")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "selftest_sheet.png")
    sheet.save(path)
    print(f"-> {path}  ({sheet.size[0]}x{sheet.size[1]})")


# --------------------------------------------------------------------------- #
# the real thing
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="results/design_searched_realgeom.npz")
    ap.add_argument("--etas", default="0,5e3,1e4,2e4",
                    help="0 means 'no watermark' (the reference column)")
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--cell", type=int, default=256,
                    help="cell size in the contact sheet")
    ap.add_argument("--gain", type=float, default=8.0,
                    help="amplification for the difference maps")
    ap.add_argument("--zoom_size", type=int, default=160)
    ap.add_argument("--zoom", type=int, default=3)
    ap.add_argument("--clip", action="store_true",
                    help="also score CLIP (needs the ViT-g-14 checkpoint)")
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--clip_model",
                    default="/root/autodl-tmp/models/ViT-g-14/open_clip_pytorch_model.bin")
    ap.add_argument("--model_id", default="/root/autodl-tmp/models/sd21base")
    ap.add_argument("--dataset",
                    default="coco:/root/autodl-tmp/data/coco5k/meta_data.json")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out_dir", default="runs/visual_quality")
    ap.add_argument("--report_path", default="results/visual_quality.md")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest(args.out_dir)
        return

    from p0_common import (OursCore, bits_from_rng, make_config, provenance,
                           resolve_device, safe_print, write_json)
    from pipeline.optim_utils import get_dataset, set_random_seed
    from run_exp import generate

    etas = [float(e) for e in args.etas.split(",") if e]
    device = resolve_device(args.device)
    cfg = make_config(args, device)
    from p0_common import build_pipe

    pipe = build_pipe(cfg, device)
    dataset, prompt_key = get_dataset(cfg.dataset)
    core = OursCore(args.design, device=device, energy_per_point=1e4)

    clip_model = clip_ref = None
    if args.clip:
        from configs import SectorConfig
        from run_exp import load_reference

        # identical call shape to run_capacity_quality_pareto.py
        clip_ref = load_reference(SectorConfig(
            model_id=args.model_id, reference_model="ViT-g-14",
            reference_model_pretrain=args.clip_model, device=device), device)
        clip_model = clip_ref[0]
    lpips_fn = None
    if args.lpips:
        try:
            import lpips

            lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
        except Exception as exc:
            safe_print(f"[warn] LPIPS unavailable ({exc})")

    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    rows: List[Tuple[str, List[Tuple[str, Image.Image]]]] = []
    metrics: Dict[str, Dict] = {}
    t0 = time.time()
    rng_w = np.random.RandomState(cfg.w_seed)

    for i in range(args.start, args.start + args.N):
        seed = i + cfg.gen_seed
        prompt = dataset[i][prompt_key]
        set_random_seed(seed)
        z_no = pipe.get_random_latents()
        bits = bits_from_rng(rng_w, core.n_bits)
        cols: List[Tuple[str, Image.Image]] = []
        ref_img = None
        for eta in etas:
            if eta <= 0:
                img = generate(pipe, prompt, z_no, cfg, device)
                tag = "no_wm"
            else:
                core.dl.energy_per_point = float(eta)
                z_w = core.embed(z_no, bits)
                img = generate(pipe, prompt, z_w, cfg, device)
                tag = f"eta{eta:g}"
            d = os.path.join(args.out_dir, "images", tag)
            os.makedirs(d, exist_ok=True)
            img.save(os.path.join(d, f"sample{i:03d}.png"))
            if ref_img is None:
                ref_img = img
                cols.append(("no watermark", img))
                continue
            m = _pair_metrics(ref_img, img, lpips_fn=lpips_fn,
                              clip=clip_model,
                              clip_ref=clip_ref, prompt=prompt)
            metrics[f"sample{i:03d}|{tag}"] = m
            cols.append((f"{tag}  PSNR {m['psnr']:.1f} dB", img))
        rows.append((f"sample {os.path.basename(str(i))}", cols))
        # per-sample full-size comparison + zoom
        full = contact_sheet([(f"sample {i}", cols)], cell=512, title="")
        full.save(os.path.join(args.out_dir, f"sample{i:03d}_full.png"))
        zrows = [(f"sample {i}", [(lab, center_crop_zoom(im, args.zoom_size,
                                                         args.zoom))
                                  for lab, im in cols])]
        contact_sheet(zrows, cell=args.zoom_size * args.zoom,
                      title=f"centre crop x{args.zoom}").save(
            os.path.join(args.out_dir, f"sample{i:03d}_zoom.png"))
        safe_print(f"[ok] sample {i}")

    sheet = contact_sheet(rows, cell=args.cell, gain=args.gain,
                          title=f"SectorSync visual quality — "
                                f"{os.path.basename(args.design)}, N={args.N}")
    os.makedirs(args.out_dir, exist_ok=True)
    sheet_path = os.path.join(args.out_dir, "sheet_all.png")
    sheet.save(sheet_path)

    # ---- markdown report ------------------------------------------------
    lines = ["# 可视质量检查（配对图像，真实生成结果）", "",
             f"载波 `{os.path.basename(args.design)}`；N={args.N}；"
             f"每张图的四个版本共用**同一个 prompt 与同一个起始潜变量**"
             f"（`η=0` 即无水印参考）。差分图为 `|ref − cell| × {args.gain:g}`。",
             "",
             "> 这些是**未受攻击**的生成图，只回答「水印本身让画面变了多少」。",
             "", f"![contact sheet]({os.path.relpath(sheet_path, os.path.dirname(args.report_path))})",
             "", "## 配对失真（相对同一 latent 的无水印图）", "",
             "| sample \\| cell | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP ↑ |",
             "|---|---:|---:|---:|---:|"]
    for k, m in metrics.items():
        clipv = m.get("clip")
        lines.append(f"| `{k}` | {m['psnr']:.2f} | "
                     f"{m['ssim']:.4f} | {m['lpips']:.4f} | "
                     f"{(f'{clipv:.4f}' if clipv is not None else '--')} |")
    if metrics:
        arr = {f: np.asarray([m[f] for m in metrics.values()], dtype=float)
               for f in ("psnr", "ssim", "lpips")}
        lines += ["", "均值：" + "、".join(
            f"{f.upper()} {np.nanmean(v):.4f}" for f, v in arr.items()) + "。", ""]
    lines += ["## 逐样本大图 / 细节放大", ""]
    for i in range(args.start, args.start + args.N):
        lines += [f"### sample {i}", "",
                  f"![full]({os.path.relpath(os.path.join(args.out_dir, f'sample{i:03d}_full.png'), os.path.dirname(args.report_path))})",
                  "",
                  f"![zoom]({os.path.relpath(os.path.join(args.out_dir, f'sample{i:03d}_zoom.png'), os.path.dirname(args.report_path))})",
                  ""]
    os.makedirs(os.path.dirname(os.path.abspath(args.report_path)), exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    write_json(os.path.join(args.out_dir, "summary.json"),
               {"config": vars(args), "provenance": provenance(),
                "metrics": metrics, "elapsed_sec": time.time() - t0})
    safe_print(f"\n-> {sheet_path}\n-> {args.report_path}")


if __name__ == "__main__":
    main()
