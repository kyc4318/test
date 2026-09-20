"""P0 gate 5: capacity x energy Pareto (quality vs rotation robustness).

A reviewer's natural attack is "your robustness just comes from injecting more
energy".  This script sweeps

    design   in { B=8 canonical, B=16 }        (capacity axis)
    eta      in { 5e3, 1e4, 2e4 }              (energy axis, 1e4 = pilot)

and, for every cell, measures both sides of the trade-off on the *same* images,
prompts, seeds and payload draws:

    quality     PSNR / SSIM / LPIPS (and CLIP if --clip) against the
                un-watermarked generation of the same latent
    robustness  BitAcc / PMR and sync failure for clean / rot45 / rot+noise

The un-watermarked image and the payload bits are generated once per image and
reused by every cell, so cells are paired rather than independently sampled.

    python run_capacity_quality_pareto.py --N 50 \
        --designs results/design_searched_realgeom.npz,results/design_B16_s0.npz \
        --etas 5e3,1e4,2e4 --cases clean,rot45,rot+noise0.05
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from tqdm import tqdm

from paper_protocol import AttackCtx, apply_case, case_seed
from p0_common import (OursCore, ResumeLog, align_error, bits_from_rng,
                       case_rotation_angle, config_hash, is_synced,
                       make_config, perfect_match, provenance, resolve_device,
                       safe_print, search_grid, wilson, write_json)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs",
                    default="results/design_searched_realgeom.npz,"
                            "results/design_B16_s0.npz")
    ap.add_argument("--etas", default="5e3,1e4,2e4")
    ap.add_argument("--cases", default="clean,rot45,rot+noise0.05")
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--grid_step", type=float, default=2.0)
    ap.add_argument("--sync_tol", type=float, default=2.0)
    ap.add_argument("--clip", action="store_true",
                    help="also score CLIP (needs the ViT-g-14 checkpoint)")
    ap.add_argument("--lpips", action="store_true",
                    help="also score LPIPS (needs the lpips package + weights)")
    ap.add_argument("--save_images", default="",
                    help="write generated PNGs here as <cell>/<idx>.png")
    ap.add_argument("--clip_model",
                    default="/root/autodl-tmp/models/ViT-g-14/open_clip_pytorch_model.bin")
    ap.add_argument("--model_id", default="/root/autodl-tmp/models/sd21base")
    ap.add_argument("--dataset",
                    default="coco:/root/autodl-tmp/data/coco5k/meta_data.json")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out_dir", default="runs/pareto")
    ap.add_argument("--report_path", default="results/paper_pareto.md")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def cell_tag(design: str, eta: float) -> str:
    base = os.path.basename(design).replace(".npz", "")
    return f"{base}_eta{eta:g}"


def quality_metrics(img_no, img_w, lpips_fn=None):
    from PIL import Image

    a = np.asarray(img_no).astype(np.float64) / 255.0
    b = np.asarray(img_w).astype(np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    ssim = float("nan")
    try:
        from skimage.metrics import structural_similarity

        ssim = float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))
    except Exception:
        pass
    out = {"mse": mse, "psnr": psnr, "ssim": ssim, "lpips": float("nan")}
    if lpips_fn is not None:
        import torch
        from torchvision.transforms.functional import to_tensor

        # ``to_tensor`` returns CPU tensors while the LPIPS net lives on
        # ``--device``; without the move LPIPS dies on its shift/scale buffers
        # ("Expected all tensors to be on the same device").
        lpips_dev = next(lpips_fn.parameters()).device
        with torch.no_grad():
            x = (to_tensor(img_no).unsqueeze(0).to(lpips_dev) * 2 - 1)
            y = (to_tensor(img_w).unsqueeze(0).to(lpips_dev) * 2 - 1)
            out["lpips"] = float(lpips_fn(x, y).item())
    return out


def main() -> None:
    args = parse_args()
    from run_exp import generate, invert
    from pipeline.optim_utils import get_dataset, set_random_seed

    device = resolve_device(args.device)
    cfg = make_config(args, device)
    from p0_common import build_pipe

    pipe = build_pipe(cfg, device)
    dataset, prompt_key = get_dataset(cfg.dataset)
    emb = pipe.get_text_embedding("")

    designs = [d for d in args.designs.split(",") if d]
    etas = [float(e) for e in args.etas.split(",") if e]
    cases = [c for c in args.cases.split(",") if c]
    grid = search_grid(args.grid_step)

    cores = {}
    for d in designs:
        if not os.path.exists(d):
            raise SystemExit(f"missing design: {d}")
        for eta in etas:
            cores[(d, eta)] = OursCore(d, device=device, energy_per_point=eta)

    ref = None
    if args.clip:
        from run_exp import load_reference
        from configs import SectorConfig
        from pipeline.optim_utils import measure_similarity  # noqa: F401

        ref = load_reference(SectorConfig(
            model_id=args.model_id, reference_model="ViT-g-14",
            reference_model_pretrain=args.clip_model, device=device), device)
    lpips_fn = None
    if args.lpips:
        try:
            import lpips

            lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
        except Exception as exc:  # pragma: no cover
            print(f"[warn] LPIPS unavailable ({exc}); skipping")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(args.save_images, "no_wm"), exist_ok=True)
    run_meta = {
        "script": "run_capacity_quality_pareto.py",
        "designs": designs,
        "design_hashes": {d: config_hash({"design": d}) for d in designs},
        "etas": etas,
        "cases": cases,
        "N": args.N,
        "start": args.start,
        "grid_step": args.grid_step,
        "clip": bool(args.clip),
        "lpips": bool(args.lpips),
        "cells": [cell_tag(d, e) for d in designs for e in etas],
        "provenance": provenance(),
    }
    write_json(os.path.join(args.out_dir, "run_config.json"), run_meta)

    rows_path = os.path.join(args.out_dir, "rows.jsonl")
    rng_w = np.random.RandomState(cfg.w_seed)
    ctx = AttackCtx(device=device)
    t0 = time.time()

    with ResumeLog(rows_path, keys=("uid",)) as log:
        for i in tqdm(range(args.start, args.start + args.N), desc="pareto"):
            seed = i + cfg.gen_seed
            prompt = dataset[i][prompt_key]
            set_random_seed(seed)
            z_no = pipe.get_random_latents()
            bits = bits_from_rng(rng_w, max(c.n_bits for c in cores.values()))
            true_bits = np.where(bits > 0, 1, 0)
            img_no = generate(pipe, prompt, z_no, cfg, device)
            if args.save_images:
                img_no.save(os.path.join(args.save_images, "no_wm",
                                         f"{i:05d}.png"))
            clip_no = None
            if ref is not None:
                from pipeline.optim_utils import measure_similarity

                clip_no = float(measure_similarity([img_no], prompt, ref[0],
                                                   ref[1], ref[2], device)[0])

            for d in designs:
                for eta in etas:
                    core = cores[(d, eta)]
                    tag = cell_tag(d, eta)
                    b = bits[:core.n_bits]
                    tb = true_bits[:core.n_bits]
                    z_w = core.embed(z_no, b)
                    img_w = generate(pipe, prompt, z_w, cfg, device)
                    if args.save_images:
                        outdir = os.path.join(args.save_images, tag)
                        os.makedirs(outdir, exist_ok=True)
                        img_w.save(os.path.join(outdir, f"{i:05d}.png"))

                    qm = quality_metrics(img_no, img_w, lpips_fn)
                    if ref is not None:
                        from pipeline.optim_utils import measure_similarity

                        qm["clip_w"] = float(measure_similarity(
                            [img_w], prompt, ref[0], ref[1], ref[2], device)[0])
                        qm["clip_no"] = clip_no
                        qm["clip_delta"] = qm["clip_w"] - clip_no

                    theta = float(np.random.RandomState(seed + 777)
                                  .uniform(0.0, 180.0))
                    for case in cases:
                        uid = f"{tag}|{i}|{case}"
                        if log.has({"uid": uid}):
                            continue
                        alpha = case_rotation_angle(case, theta)
                        rng_a = np.random.RandomState(case_seed(seed, case))
                        img_a = apply_case(img_w, case, theta, rng_a, ctx)
                        z_hat = invert(pipe, img_a, emb, cfg, device)
                        S = core.landscape(z_hat, grid)
                        est = float(grid[int(np.argmax(S))])
                        ell = core.ell_mean(z_hat, est)
                        pred = (ell > 0).astype(int)
                        ell_or = core.ell_mean(z_hat, alpha)
                        pred_or = (ell_or > 0).astype(int)
                        row = {
                            "uid": uid,
                            "cell": tag,
                            "design": d,
                            "eta": eta,
                            "n_bits": core.n_bits,
                            "image_id": int(i),
                            "prompt_id": int(i),
                            "seed": int(seed),
                            "case": case,
                            "true_angle": float(alpha),
                            "est_angle": est,
                            "est_error_deg": align_error(est, alpha),
                            "synced": is_synced(est, alpha, args.sync_tol),
                            "bit_acc": float((tb == pred).mean()),
                            "perfect": perfect_match(tb, pred),
                            "bit_acc_oracle": float((tb == pred_or).mean()),
                            "perfect_oracle": perfect_match(tb, pred_or),
                            "detection_score": float(S.max()),
                            **qm,
                        }
                        log.append(row)
                    del img_w, z_w
            del img_no, z_no
            if _cuda():
                import torch

                torch.cuda.empty_cache()

    from p0_common import load_rows

    rows = load_rows(rows_path)
    summary = summarise(rows, designs, etas, cases, args.sync_tol)
    write_json(os.path.join(args.out_dir, "summary.json"),
               {"config": run_meta, "summary": summary,
                "elapsed_sec": time.time() - t0})
    report(summary, run_meta, args)
    print(f"\nelapsed {(time.time() - t0) / 60:.1f} min -> {args.out_dir}")


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def summarise(rows, designs, etas, cases, sync_tol):
    out = {}
    for d in designs:
        for eta in etas:
            tag = cell_tag(d, eta)
            sub = [r for r in rows if r["cell"] == tag]
            if not sub:
                continue
            entry = {
                "design": d,
                "eta": eta,
                "n_bits": int(sub[0]["n_bits"]),
                "n": int(len({r["image_id"] for r in sub})),
                "quality": {
                    "psnr": _m([r["psnr"] for r in sub]),
                    "ssim": _m([r["ssim"] for r in sub if np.isfinite(r["ssim"])]),
                    "lpips": _m([r["lpips"] for r in sub if np.isfinite(r["lpips"])]),
                    "clip_w": _m([r["clip_w"] for r in sub if "clip_w" in r]),
                    "clip_delta": _m([r["clip_delta"] for r in sub
                                      if "clip_delta" in r]),
                },
                "cases": {},
            }
            for case in cases:
                cr = [r for r in sub if r["case"] == case]
                if not cr:
                    continue
                n = len(cr)
                k_pmr = int(sum(1 for r in cr if r["perfect"]))
                pmr, plo, phi = wilson(k_pmr, n)
                k_sf = int(sum(1 for r in cr if not r["synced"]))
                sf, slo, shi = wilson(k_sf, n)
                entry["cases"][case] = {
                    "n": n,
                    "bit_acc_mean": _m([r["bit_acc"] for r in cr]),
                    "bit_acc_oracle_mean": _m([r["bit_acc_oracle"] for r in cr]),
                    "pmr": {"mean": pmr, "ci": [plo, phi], "k": k_pmr},
                    "sync_fail": {"mean": sf, "ci": [slo, shi], "k": k_sf},
                    "align_err_mean": _m([r["est_error_deg"] for r in cr]),
                }
            out[tag] = entry
    return out


def _m(v):
    return float(np.mean(v)) if len(v) else None


def report(summary, run_meta, args) -> None:
    lines = ["# P0 容量 × 能量 Pareto", "",
             f"N={run_meta['N']}（每图固定 prompt/latent/载荷，跨格配对），"
             f"搜索网格 {run_meta['grid_step']}°，图像质量相对**同一 latent 的"
             f"无水印生成图**。", ""]
    lines += ["| cell | bits | PSNR | SSIM | LPIPS | clean PMR | rot45 PMR | "
              "rot+noise PMR | rot+noise BitAcc | rot+noise 失锁率 |",
              "|---|---:|---:|---:|---:|---|---|---|---:|---:|"]
    for tag, e in summary.items():
        c_clean = e["cases"].get("clean", {})
        c45 = e["cases"].get("rot45", {})
        cn = e["cases"].get("rot+noise0.05") or next(
            (v for k, v in e["cases"].items() if k.startswith("rot") and
             "noise" in k), {})
        q = e["quality"]
        lines.append(
            f"| {tag} | {e['n_bits']} | {_f(q['psnr'], 2)} | {_f(q['ssim'], 4)} | "
            f"{_f(q['lpips'], 4)} | {_p(c_clean.get('pmr'))} | "
            f"{_p(c45.get('pmr'))} | {_p(cn.get('pmr'))} | "
            f"{_f(cn.get('bit_acc_mean'), 4)} | "
            f"{_f((cn.get('sync_fail') or {}).get('mean'), 4)} |")
    lines += ["", "说明：", "",
              "- 每个 cell 都使用同一批 prompt / latent / 载荷比特，因此质量列与"
              "鲁棒列之间可直接配对比较，而不是各自独立采样。",
              "- 与 GS / SFWMark 的工作点对比：这两类方法没有可比的能量旋钮"
              "（GS 的强度由截断高斯标定，SFWMark 由模板幅度决定），因此 Pareto "
              "曲线只刻画**本文方法内部**的质量—鲁棒权衡；跨方法的工作点比较放"
              "在质量表（§5.7）与主表（§5.3）中完成。",
              "- LPIPS 需要 `lpips` 包与权重，缺失时该列显示 `--`。", ""]
    path = args.report_path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    safe_print("\n".join(lines))
    print(f"-> {path}")


def _f(v, nd=4):
    return "--" if v is None else f"{float(v):.{nd}f}"


def _p(d):
    if not d:
        return "--"
    return f"{d['mean']:.3f}"


if __name__ == "__main__":
    main()
