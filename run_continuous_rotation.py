"""P0 gate 2: continuous, negative and off-grid rotations.

The pilot only tested 15/45/75/90 degrees, all positive and all on the search
grid.  A reviewer will ask two things: does the synchroniser survive
intermediate angles, and does it survive negative ones?  This script sweeps

    alpha in { fixed list | U(-180, 180) per image | regular grid }
    sigma in { 0, 0.02, 0.05, 0.1 }   (Gaussian noise applied after rotation)

and records, per (image, alpha, sigma):

    est_angle, align error (mod 180, the carrier symmetry period),
    synced / not, BitAcc, PMR, oracle BitAcc (bits decoded at the *true* angle),
    S_true, best false peak, relative margin, wall-clock

so that "0% sync failure" and "the failure is peak collapse, not a bad arg-max"
are both directly checkable.

    python run_continuous_rotation.py --N 30 \
        --angles 7.3,22.5,37.3,45,58.7,73.4,102.5,135,168.9,-30,-75,-135
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from tqdm import tqdm

from p0_common import (OursCore, ResumeLog, add_gaussian_noise, align_error,
                       bit_acc, bits_from_rng, bootstrap_ci, is_synced,
                       make_config, perfect_match, provenance, resolve_device,
                       rotate_image, safe_print, search_grid, signed_error,
                       wilson, write_json)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="results/design_searched_realgeom.npz")
    ap.add_argument("--mode", default="fixed",
                    choices=["fixed", "random", "grid"],
                    help="fixed: --angles list; random: uniform in "
                         "[--angle_min,--angle_max); grid: regular sweep")
    ap.add_argument("--angles", default="7.3,22.5,37.3,45,58.7,73.4,"
                                       "102.5,135,168.9,-30,-75,-135")
    ap.add_argument("--n_angles_per_image", type=int, default=8)
    ap.add_argument("--angle_min", type=float, default=-180.0)
    ap.add_argument("--angle_max", type=float, default=180.0)
    ap.add_argument("--angle_step", type=float, default=15.0)
    ap.add_argument("--sigmas", default="0")
    ap.add_argument("--grid_step", type=float, default=1.0,
                    help="decoder-side angle search resolution")
    ap.add_argument("--energy_per_point", type=float, default=1e4)
    ap.add_argument("--sync_tol", type=float, default=2.0)
    ap.add_argument("--false_window", type=float, default=10.0,
                    help="|gamma-alpha| beyond this counts as a false peak")
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--angle_seed", type=int, default=1234)
    ap.add_argument("--tag", default="")
    ap.add_argument("--model_id", default="/root/autodl-tmp/models/sd21base")
    ap.add_argument("--dataset",
                    default="coco:/root/autodl-tmp/data/coco5k/meta_data.json")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out_dir", default="runs/p0_rotation")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def draw_angles(args, i: int) -> np.ndarray:
    if args.mode == "fixed":
        return np.asarray([float(a) for a in args.angles.split(",") if a])
    if args.mode == "grid":
        n = int(round((args.angle_max - args.angle_min) / args.angle_step))
        return args.angle_min + args.angle_step * np.arange(n)
    rng = np.random.RandomState(args.angle_seed + int(i))
    return rng.uniform(args.angle_min, args.angle_max, args.n_angles_per_image)


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

    sigmas = [float(s) for s in args.sigmas.split(",") if s]
    grid = search_grid(args.grid_step)
    core = OursCore(args.design, device=device,
                    energy_per_point=args.energy_per_point)

    tag = args.tag or f"{args.mode}_step{args.grid_step:g}_N{args.N}"
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    run_meta = {
        "script": "run_continuous_rotation.py",
        "design": args.design,
        "n_bits": core.n_bits,
        "mode": args.mode,
        "angles": args.angles,
        "angle_min": args.angle_min,
        "angle_max": args.angle_max,
        "angle_step": args.angle_step,
        "n_angles_per_image": args.n_angles_per_image,
        "angle_seed": args.angle_seed,
        "sigmas": sigmas,
        "grid_step": args.grid_step,
        "grid_n": int(len(grid)),
        "sync_tol": args.sync_tol,
        "false_window": args.false_window,
        "energy_per_point": args.energy_per_point,
        "N": args.N,
        "start": args.start,
        "provenance": provenance(),
    }
    write_json(os.path.join(out_dir, "run_config.json"), run_meta)

    rows_path = os.path.join(out_dir, "rows.jsonl")
    t0 = time.time()
    rng_w = np.random.RandomState(cfg.w_seed)
    with ResumeLog(rows_path, keys=("uid",)) as log:
        for i in tqdm(range(args.start, args.start + args.N), desc="rotation"):
            seed = i + cfg.gen_seed
            prompt = dataset[i][prompt_key]
            set_random_seed(seed)
            z_no = pipe.get_random_latents()
            bits = bits_from_rng(rng_w, core.n_bits)
            true_bits = np.where(bits > 0, 1, 0)
            z_w = core.embed(z_no, bits)
            img_w = generate(pipe, prompt, z_w, cfg, device)

            for alpha in draw_angles(args, i):
                for sigma in sigmas:
                    row = {
                        "uid": f"{i}|{alpha:.4f}|{sigma:g}",
                        "image_id": int(i),
                        "prompt_id": int(i),
                        "seed": int(seed),
                        "case": f"rot{alpha:.4f}" + (f"+noise{sigma:g}"
                                                     if sigma > 0 else ""),
                        "true_angle": float(alpha),
                        "sigma": float(sigma),
                        "on_grid": float(
                            abs(alpha / args.grid_step
                                - round(alpha / args.grid_step)) < 1e-6),
                        "payload_bits": core.n_bits,
                    }
                    if log.has(row):
                        continue

                    img_a = rotate_image(img_w, alpha)
                    if sigma > 0:
                        img_a = add_gaussian_noise(
                            img_a, sigma,
                            abs(int(alpha * 1000)) + int(sigma * 1e4))
                    t_inv = time.time()
                    z_hat = invert(pipe, img_a, emb, cfg, device)
                    inv_sec = time.time() - t_inv

                    t_dec = time.time()
                    S = core.landscape(z_hat, grid)
                    best = int(np.argmax(S))
                    est = float(grid[best])
                    s_true = core.score_at(z_hat, alpha)
                    far = np.asarray([align_error(g, alpha) > args.false_window
                                      for g in grid])
                    s_false = float(S[far].max()) if far.any() else float("nan")
                    ell_est = core.ell_mean(z_hat, est)
                    ell_true = core.ell_mean(z_hat, alpha)
                    dec_sec = time.time() - t_dec

                    topk = np.argsort(-S)[:5]
                    row.update({
                        "est_angle": est,
                        "est_error_deg": align_error(est, alpha),
                        "signed_error_deg": signed_error(est, alpha),
                        "synced": is_synced(est, alpha, args.sync_tol),
                        "detection_score": float(S[best]),
                        "S_true": float(s_true),
                        "S_false": s_false,
                        "delta_rel": (float((s_true - s_false) / s_true)
                                      if s_true > 0 else float("nan")),
                        "scores_topk": [{"angle": float(grid[j]),
                                         "score": float(S[j])} for j in topk],
                        "decoded_bits": (ell_est > 0).astype(int).tolist(),
                        "decoded_bits_oracle": (ell_true > 0).astype(int).tolist(),
                        "bit_acc": bit_acc(true_bits, (ell_est > 0).astype(int)),
                        "perfect": perfect_match(true_bits,
                                                 (ell_est > 0).astype(int)),
                        "bit_acc_oracle": bit_acc(true_bits,
                                                  (ell_true > 0).astype(int)),
                        "perfect_oracle": perfect_match(true_bits,
                                                        (ell_true > 0).astype(int)),
                        "inv_sec": float(inv_sec),
                        "decode_sec": float(dec_sec),
                    })
                    log.append(row)
            del img_w, z_no
            if _cuda():
                import torch

                torch.cuda.empty_cache()

    from p0_common import load_rows

    rows = load_rows(rows_path)
    summary = summarise(rows, args)
    write_json(os.path.join(out_dir, "summary.json"),
               {"config": run_meta, "summary": summary,
                "elapsed_sec": time.time() - t0})
    report(out_dir, run_meta, summary, rows)
    print(f"\nelapsed {(time.time() - t0) / 60:.1f} min -> {out_dir}")


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _agg(rows):
    n = len(rows)
    if n == 0:
        return None
    errs = np.asarray([r["est_error_deg"] for r in rows], dtype=np.float64)
    fails = int(sum(1 for r in rows if not r["synced"]))
    p, lo, hi = wilson(fails, n)
    bit = np.asarray([r["bit_acc"] for r in rows], dtype=np.float64)
    bit_or = np.asarray([r["bit_acc_oracle"] for r in rows], dtype=np.float64)
    pmr_k = int(sum(1 for r in rows if r["perfect"]))
    pmr, plo, phi = wilson(pmr_k, n)
    return {
        "n": n,
        "sync_fail_rate": {"mean": p, "ci": [lo, hi], "k": fails},
        "align_err_mean": float(errs.mean()),
        "align_err_median": float(np.median(errs)),
        "align_err_p90": float(np.percentile(errs, 90)),
        "align_err_p95": float(np.percentile(errs, 95)),
        "align_err_max": float(errs.max()),
        "bit_acc_mean": float(bit.mean()),
        "bit_acc_oracle_mean": float(bit_or.mean()),
        "bit_acc": bootstrap_ci(bit),
        "pmr": {"mean": pmr, "ci": [plo, phi], "k": pmr_k},
        "delta_rel_mean": float(np.nanmean(
            [r["delta_rel"] for r in rows])),
        "mean_inv_sec": float(np.mean([r["inv_sec"] for r in rows])),
        "mean_decode_sec": float(np.mean([r["decode_sec"] for r in rows])),
    }


def summarise(rows, args):
    out = {"overall": _agg(rows)}
    sigmas = sorted({r["sigma"] for r in rows})
    out["by_sigma"] = {f"{s:g}": _agg([r for r in rows if r["sigma"] == s])
                       for s in sigmas}
    out["by_on_grid"] = {
        "off_grid": _agg([r for r in rows if not r["on_grid"]]),
        "on_grid": _agg([r for r in rows if r["on_grid"]]),
    }
    out["by_angle_bin"] = {}
    for lo in (-180, -90, 0, 90):
        hi = lo + 90
        sub = [r for r in rows if lo <= r["true_angle"] < hi]
        if sub:
            out["by_angle_bin"][f"[{lo},{hi})"] = _agg(sub)
    out["by_abs_bin"] = {}
    for lo, hi in ((0, 15), (15, 45), (45, 75), (75, 105), (105, 180)):
        sub = [r for r in rows if lo <= abs(r["true_angle"]) < hi]
        if sub:
            out["by_abs_bin"][f"[{lo},{hi})"] = _agg(sub)
    # per-image worst case, to avoid the average hiding a single bad image
    per_img = {}
    for r in rows:
        per_img.setdefault(r["image_id"], []).append(r)
    worst = [_agg(v) for v in per_img.values()]
    out["per_image"] = {
        "n_images": len(per_img),
        "mean_sync_fail_rate": float(np.mean([w["sync_fail_rate"]["mean"]
                                              for w in worst])),
        "images_with_any_fail": int(sum(
            1 for v in per_img.values() if any(not r["synced"] for r in v))),
        "mean_bit_acc": float(np.mean([w["bit_acc_mean"] for w in worst])),
    }
    return out


def report(out_dir, run_meta, summary, rows) -> None:
    lines = ["# P0 连续 / 负角旋转扫描", "",
             f"模式 `{run_meta['mode']}`，搜索网格 {run_meta['grid_step']}°"
             f"（{run_meta['grid_n']} 个候选），N={run_meta['N']}，"
             f"载波 B={run_meta['n_bits']}，能量 η={run_meta['energy_per_point']:g}。",
             ""]
    lines += ["| 条件 | n | 失锁率 [95%CI] | 角误差 中位/p90 | BitAcc | "
              "PMR [95%CI] | oracle BitAcc | Δrel |",
              "|---|---:|---|---:|---:|---|---:|---:|"]

    def row(label, a):
        if not a:
            return f"| {label} | -- | -- | -- | -- | -- | -- | -- |"
        sf = a["sync_fail_rate"]
        pm = a["pmr"]
        return (f"| {label} | {a['n']} | {sf['mean']:.4f} "
                f"[{sf['ci'][0]:.3f}, {sf['ci'][1]:.3f}] | "
                f"{a['align_err_median']:.2f} / {a['align_err_p90']:.2f} | "
                f"{a['bit_acc_mean']:.4f} | {pm['mean']:.3f} "
                f"[{pm['ci'][0]:.3f}, {pm['ci'][1]:.3f}] | "
                f"{a['bit_acc_oracle_mean']:.4f} | {a['delta_rel_mean']:.3f} |")

    lines.append(row("all", summary["overall"]))
    for k, a in summary["by_sigma"].items():
        lines.append(row(f"σ={k}", a))
    for k, a in summary["by_on_grid"].items():
        lines.append(row(k, a))
    for k, a in summary["by_angle_bin"].items():
        lines.append(row(f"α∈{k}", a))
    off_clean = [r for r in rows if (not r["on_grid"]) and r["sigma"] == 0]
    if off_clean:
        lines.append(row("off-grid & σ=0", _agg(off_clean)))

    pi = summary["per_image"]
    lines += ["", f"逐图统计：{pi['n_images']} 张图中 "
              f"{pi['images_with_any_fail']} 张至少有一次失锁；"
              f"平均逐图失锁率 {pi['mean_sync_fail_rate']:.4f}；"
              f"平均逐图 BitAcc {pi['mean_bit_acc']:.4f}。", "",
              "角误差定义为 `min(|Δ| mod 180, 180-|Δ| mod 180)`，"
              "与载波的 Hermitian 对径对称一致；`oracle BitAcc` 是在真实角度上"
              "解码得到的，用来区分“同步失败”与“载荷本身失败”。", ""]
    path = os.path.join(out_dir, "summary.md")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    safe_print("\n".join(lines))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
