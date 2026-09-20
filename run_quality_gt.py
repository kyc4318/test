"""Generation quality: FID-vs-GT protocol plus paired distortion metrics.

Literature (Tree-Ring / RingID / SFWMark / MaXsive) reports FID between the
generated images and the **MS-COCO ground-truth images**, not against their own
un-watermarked generations.  This script therefore writes one PNG folder per
configuration under runs/quality_gt/images/<cfg>/, together with the CLIP score
(OpenCLIP ViT-g-14) of the generated image against its prompt.

Because FID needs thousands of images to separate two methods, the same run
also reports the small-N quantities that *are* measurable: PSNR / SSIM / LPIPS
between the watermarked and the un-watermarked generation of the same latent,
the paired CLIP delta, and bootstrap confidence intervals for all of them.
Every blob carries provenance (model id, dataset, CLIP checkpoint, git commit).

    python run_quality_gt.py --N 500 --methods no_wm,ours,gs256,gs8,sfw_hsqr
    python run_fid.py runs/quality_gt/images/<cfg> /root/autodl-tmp/data/coco5k/images512
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from configs import SectorConfig
from p0_common import bootstrap_ci, paired_bootstrap_ci, provenance
from pipeline.optim_utils import get_dataset, measure_similarity, set_random_seed
from run_exp import Watermarker, generate, load_pipeline, load_reference
from run_paper_compare import build_method, SFWMethod


def distortion(a, b, lpips_fn=None):
    """Paired distortion of image ``b`` against reference ``a`` (PIL in)."""
    x = np.asarray(a).astype(np.float64) / 255.0
    y = np.asarray(b).astype(np.float64) / 255.0
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
        from torchvision.transforms.functional import to_tensor

        # ``to_tensor`` returns CPU tensors while the LPIPS net lives on
        # ``--device``; without the move LPIPS dies on its shift/scale buffers.
        lpips_dev = next(lpips_fn.parameters()).device
        with torch.no_grad():
            out["lpips"] = float(lpips_fn(
                to_tensor(a).unsqueeze(0).to(lpips_dev) * 2 - 1,
                to_tensor(b).unsqueeze(0).to(lpips_dev) * 2 - 1).item())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="no_wm,ours,gs256,gs8,sfw_hsqr")
    ap.add_argument("--N", type=int, default=500)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=1e4)
    ap.add_argument("--design", default="results/design_searched_realgeom.npz")
    # kept in sync with run_paper_compare.py so build_method() works unchanged
    ap.add_argument("--grid_step", type=float, default=2.0)
    ap.add_argument("--energy_per_point", type=float, default=None,
                    help="defaults to --eta")
    ap.add_argument("--gs_ch", type=int, default=1)
    ap.add_argument("--gs_hw", type=int, default=8)
    ap.add_argument("--fpr", type=float, default=1e-6)
    ap.add_argument("--user_number", type=int, default=10 ** 6)
    ap.add_argument("--sfw_candidates", type=int, default=2048)
    ap.add_argument("--paired_ref", default="no_wm",
                    help="method whose image is the distortion reference")
    ap.add_argument("--lpips", action="store_true",
                    help="also compute LPIPS (needs the lpips package)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--model_id", default="/root/autodl-tmp/models/sd21base")
    ap.add_argument("--dataset",
                    default="coco:/root/autodl-tmp/data/coco5k/meta_data.json")
    ap.add_argument("--clip_model",
                    default="/root/autodl-tmp/models/ViT-g-14/open_clip_pytorch_model.bin")
    ap.add_argument("--out_dir", default="runs/quality_gt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.energy_per_point is None:
        args.energy_per_point = args.eta

    cfg = SectorConfig(model_id=args.model_id, dataset=args.dataset,
                       num_inference_steps=args.steps,
                       test_num_inference_steps=args.steps, device=args.device)
    pipe = load_pipeline(cfg, args.device)
    dataset, prompt_key = get_dataset(cfg.dataset)
    ref = load_reference(SectorConfig(
        model_id=args.model_id, reference_model="ViT-g-14",
        reference_model_pretrain=args.clip_model, device=args.device), args.device)
    wm = Watermarker(cfg)
    rng_w = np.random.RandomState(cfg.w_seed)

    lpips_fn = None
    if args.lpips:
        try:
            import lpips

            lpips_fn = lpips.LPIPS(net="alex").to(args.device).eval()
        except Exception as exc:  # pragma: no cover
            print(f"[warn] LPIPS unavailable ({exc}); skipping")

    methods = [m for m in args.methods.split(",") if m]
    dirs = {m: os.path.join(args.out_dir, "images", m) for m in methods}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # one adapter per method; `no_wm` uses the raw latent
    adapters = {}
    for m in methods:
        if m == "no_wm":
            continue
        adapters[m] = build_method(m, args, args.device)
        if isinstance(adapters[m], SFWMethod):
            adapters[m].build_patterns(pipe)

    rows = []
    t0 = time.time()
    for i in tqdm(range(args.start, args.start + args.N), desc="quality-gt"):
        seed = i + cfg.gen_seed
        prompt = dataset[i][prompt_key]
        set_random_seed(seed)
        z_no = pipe.get_random_latents()
        bits, _ = wm.make_message(rng_w)
        rec = {"index": i, "seed": seed, "prompt": prompt}
        imgs = {}
        for m in methods:
            msg = {"index": i, "seed": seed, "theta": 0.0}
            if m == "no_wm":
                z = z_no
            else:
                lat = adapters[m].latents(z_no, bits, msg)
                z = lat["wm"]
            img = generate(pipe, prompt, z, cfg, args.device)
            imgs[m] = img
            img.save(os.path.join(dirs[m], f"{i:05d}.png"))
            clip = float(measure_similarity([img], prompt, ref[0], ref[1],
                                            ref[2], args.device)[0])
            rec[m] = clip
        # paired distortion against the (un-watermarked) reference generation
        if args.paired_ref in imgs:
            base = imgs[args.paired_ref]
            for m in methods:
                if m == args.paired_ref:
                    continue
                for k, v in distortion(base, imgs[m], lpips_fn).items():
                    rec[f"{m}_{k}"] = v
        rows.append(rec)
        del z_no, imgs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {}
    for m in methods:
        v = np.array([r[m] for r in rows], dtype=np.float64)
        entry = {"clip_mean": float(v.mean()), "clip_std": float(v.std()),
                 "n": len(v), "images": dirs[m],
                 "clip_ci": bootstrap_ci(v, n_boot=args.n_boot)}
        if args.paired_ref in methods and m != args.paired_ref:
            entry["reference"] = args.paired_ref
            entry["clip_delta_vs_ref"] = paired_bootstrap_ci(
                [r[m] for r in rows], [r[args.paired_ref] for r in rows],
                n_boot=args.n_boot)
            for k in ("psnr", "ssim", "lpips", "mse"):
                vals = [r[f"{m}_{k}"] for r in rows if f"{m}_{k}" in r]
                vals = [x for x in vals if np.isfinite(x)]
                if vals:
                    entry[f"{k}_vs_ref"] = bootstrap_ci(vals,
                                                        n_boot=args.n_boot)
        summary[m] = entry
    with open(os.path.join(args.out_dir, "quality_gt.json"), "w") as f:
        json.dump({"config": vars(args), "summary": summary, "rows": rows,
                   "elapsed_sec": time.time() - t0,
                   "provenance": provenance()}, f, indent=2)
    for m in methods:
        s = summary[m]
        line = (f"{m:>10}: CLIP {s['clip_mean']:.4f} "
                f"[{s['clip_ci']['ci'][0]:.4f}, {s['clip_ci']['ci'][1]:.4f}]")
        if "clip_delta_vs_ref" in s:
            d = s["clip_delta_vs_ref"]
            line += (f"  dCLIP vs {args.paired_ref} {d['mean']:+.4f} "
                     f"[{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]")
        if "psnr_vs_ref" in s:
            line += f"  PSNR {s['psnr_vs_ref']['mean']:.2f}"
        if "ssim_vs_ref" in s:
            line += f"  SSIM {s['ssim_vs_ref']['mean']:.4f}"
        if "lpips_vs_ref" in s:
            line += f"  LPIPS {s['lpips_vs_ref']['mean']:.4f}"
        print(line + f"  (N={s['n']}) -> {dirs[m]}")
    print(f"\nelapsed {(time.time()-t0)/60:.1f} min")
    print("\nnow run: python run_fid.py <cfg_dir> "
          "/root/autodl-tmp/data/coco5k/images512")


if __name__ == "__main__":
    main()
