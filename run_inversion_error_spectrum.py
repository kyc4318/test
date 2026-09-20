"""P0 gate 4: where does the inversion error go?

The pilot established that under rotation+noise the *true* synchronisation peak
collapses (S_true ~139 -> ~57) while the false peak stays flat.  That is a
statement about the statistic; the paper needs the mechanism.  This script
decomposes, for one DDIM inversion per attacked image,

    E(omega) = R_{-alpha} Z_hat_T(omega) - Z_w(omega)

into

    radial profile        mean |E|^2 per integer radius in the annulus
    angular profile       mean |E|^2 per angular bin (in the de-rotated frame)
    real / imaginary      mean |Re E|^2 vs mean |Im E|^2
    in / out of subspace  ||E_parallel||^2 vs ||E_perp||^2, where the parallel
                          part is the projection onto the carrier span Psi_j

and reports, per condition, the carrier-projected SNR

    SNR = mean_j (b_j ell_j)^2 / mean_j nu_j,

with nu_j the per-chip residual power after removing the estimated symbols.
The `clean` condition is kept as the inversion-error floor, so every other row
can be read as an increment over it: attenuation (parallel) versus leakage into
the orthogonal complement (perpendicular).

    python run_inversion_error_spectrum.py --N 25 \
        --cases clean,rot45,rot75,noise0.05,rot+noise0.05
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from tqdm import tqdm

from paper_protocol import AttackCtx, apply_case, case_seed
from p0_common import (OursCore, ResumeLog, align_error, bits_from_rng,
                       case_rotation_angle, config_hash, make_config,
                       provenance, resolve_device, safe_print, search_grid,
                       write_json)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="results/design_searched_realgeom.npz")
    ap.add_argument("--cases", default="clean,rot45,rot75,noise0.05,rot+noise0.05")
    ap.add_argument("--N", type=int, default=25)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--grid_step", type=float, default=2.0)
    ap.add_argument("--false_window", type=float, default=10.0)
    ap.add_argument("--energy_per_point", type=float, default=1e4)
    ap.add_argument("--angle_bins", type=int, default=18)
    ap.add_argument("--save_npz", type=int, default=0)
    ap.add_argument("--model_id", default="/root/autodl-tmp/models/sd21base")
    ap.add_argument("--dataset",
                    default="coco:/root/autodl-tmp/data/coco5k/meta_data.json")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out_dir", default="runs/p0_spectrum")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def layer_analysis(lay, E, Z_al, bits):
    """Project the error/observation onto one layer's carrier span."""
    mask = lay["mask"]
    M = int(lay["spec"].M)
    Psi = lay["Psi"].detach().cpu().numpy()               # [B, P] complex64
    norms = (np.abs(Psi) ** 2).sum(axis=1)                # [B]
    z = Z_al[mask]                                        # [P]
    e = E[mask]                                           # [P]
    ell = (np.conj(Psi) * z[None, :]).sum(axis=1).real / norms
    coef = (np.conj(Psi) * e[None, :]).sum(axis=1) / norms
    e_par = (coef[:, None] * Psi).sum(axis=0)
    e_tot = float((np.abs(e) ** 2).sum())
    e_par_p = float((np.abs(e_par) ** 2).sum())
    res = z - (ell[:, None] * Psi).sum(axis=0)
    q = np.asarray(lay["q"]) % M
    nu = np.asarray([np.mean(np.abs(res[q == j]) ** 2) if np.any(q == j) else 0.0
                     for j in range(M)])
    sig_power = float(np.mean((np.asarray(bits) * ell) ** 2))
    noise_power = float(np.mean(nu))
    z_ann = np.abs(Z_al[mask]) ** 2
    return {
        "ell": ell,
        "ell_true_mean": float(np.mean(np.asarray(bits) * ell)),
        "ell_abs_mean": float(np.mean(np.abs(ell))),
        "e_total": e_tot,
        "e_parallel": e_par_p,
        "e_perp": e_tot - e_par_p,
        "e_parallel_frac": e_par_p / max(e_tot, 1e-30),
        "e_perp_frac": 1.0 - e_par_p / max(e_tot, 1e-30),
        "snr_carrier": sig_power / max(noise_power, 1e-30),
        "nu_mean": noise_power,
        "obs_power": float(np.mean(z_ann)),
    }


def profile_arrays(E, r_grid, th_grid, mask, radii, angle_bins):
    """Radial and angular ``mean |E|^2`` profiles over the annulus.

    ``E``, ``r_grid`` and ``th_grid`` are 2-D, ``mask`` selects the annulus and
    the profiles are accumulated on the masked 1-D values.  Every 2-D selection
    is therefore reduced with ``[mask]`` before it is used to index ``p``.
    """
    p = np.abs(E[mask]) ** 2
    r_masked = np.floor(r_grid)[mask]
    th_masked = th_grid[mask]
    radial = []
    for r in radii:
        sel = (r_masked == r)
        radial.append(float(p[sel].mean()) if sel.any() else float("nan"))
    ang = []
    edges = np.linspace(0.0, 360.0, angle_bins + 1)
    for k in range(angle_bins):
        sel = ((th_masked >= edges[k]) & (th_masked < edges[k + 1]))
        ang.append(float(p[sel].mean()) if sel.any() else float("nan"))
    return radial, ang


def main() -> None:
    args = parse_args()
    from run_exp import generate, invert
    from pipeline.optim_utils import get_dataset, set_random_seed
    from run_paper_compare import rotate_latent
    import torch

    device = resolve_device(args.device)
    cfg = make_config(args, device)
    from p0_common import build_pipe

    pipe = build_pipe(cfg, device)
    dataset, prompt_key = get_dataset(cfg.dataset)
    emb = pipe.get_text_embedding("")
    cases = [c for c in args.cases.split(",") if c]

    core = OursCore(args.design, device=device,
                    energy_per_point=args.energy_per_point)
    size = core.dl.size
    ch = core.dl.channel
    yy, xx = np.ogrid[:size, :size]
    c0 = size // 2
    r_grid = np.hypot(yy - c0, xx - c0)
    th_grid = np.degrees(np.arctan2(yy - c0, xx - c0)) % 360.0
    ann = np.zeros((size, size), dtype=bool)
    for lay in core.dl.layers:
        ann |= lay["mask"]
    radii = sorted({int(v) for v in np.floor(r_grid[ann]).tolist()})

    os.makedirs(args.out_dir, exist_ok=True)
    run_meta = {
        "script": "run_inversion_error_spectrum.py",
        "design": args.design,
        "design_hash": config_hash({"design": args.design}),
        "n_bits": core.n_bits,
        "cases": cases,
        "N": args.N,
        "start": args.start,
        "grid_step": args.grid_step,
        "false_window": args.false_window,
        "energy_per_point": args.energy_per_point,
        "angle_bins": args.angle_bins,
        "radii": radii,
        "annulus_points": int(ann.sum()),
        "layer_bands": [[float(l["spec"].r_lo), float(l["spec"].r_hi)]
                        for l in core.dl.layers],
        "layer_K": [int(l["spec"].K) for l in core.dl.layers],
        "provenance": provenance(),
    }
    write_json(os.path.join(args.out_dir, "run_config.json"), run_meta)

    grid = search_grid(args.grid_step)
    rows_path = os.path.join(args.out_dir, "rows.jsonl")
    npz_dir = os.path.join(args.out_dir, "npz")
    if args.save_npz:
        os.makedirs(npz_dir, exist_ok=True)
    rng_w = np.random.RandomState(cfg.w_seed)
    ctx = AttackCtx(device=device)
    t0 = time.time()

    with ResumeLog(rows_path, keys=("uid",)) as log:
        for i in tqdm(range(args.start, args.start + args.N), desc="spectrum"):
            seed = i + cfg.gen_seed
            prompt = dataset[i][prompt_key]
            set_random_seed(seed)
            z_no = pipe.get_random_latents()
            bits = bits_from_rng(rng_w, core.n_bits)
            z_w = core.embed(z_no, bits)
            img_w = generate(pipe, prompt, z_w, cfg, device)

            with torch.no_grad():
                Z_w = torch.fft.fftshift(
                    torch.fft.fft2(z_w.float()), dim=(-1, -2))
            Z_w_np = Z_w[0, ch].detach().cpu().numpy()

            theta = float(np.random.RandomState(seed + 777)
                          .uniform(0.0, 180.0))
            for case in cases:
                uid = f"{i}|{case}"
                if log.has({"uid": uid}):
                    continue
                alpha = case_rotation_angle(case, theta)
                rng_a = np.random.RandomState(case_seed(seed, case))
                img_a = apply_case(img_w, case, theta, rng_a, ctx)
                z_hat = invert(pipe, img_a, emb, cfg, device)
                z_al = rotate_latent(z_hat, -alpha)
                with torch.no_grad():
                    Z_al = torch.fft.fftshift(
                        torch.fft.fft2(z_al.float()), dim=(-1, -2))
                Z_al_np = Z_al[0, ch].detach().cpu().numpy()
                E = Z_al_np - Z_w_np

                S = core.landscape(z_hat, grid)
                s_true = core.score_at(z_hat, alpha)
                far = np.asarray([align_error(g, alpha) > args.false_window
                                  for g in grid])
                s_false = float(S[far].max()) if far.any() else float("nan")
                radial, ang = profile_arrays(E, r_grid, th_grid, ann, radii,
                                             args.angle_bins)

                row = {
                    "uid": uid,
                    "image_id": int(i),
                    "prompt_id": int(i),
                    "seed": int(seed),
                    "case": case,
                    "true_angle": float(alpha),
                    "payload_bits": core.n_bits,
                    "e_mean_abs2": float(np.mean(np.abs(E[ann]) ** 2)),
                    "e_re_mean_abs2": float(np.mean(np.real(E[ann]) ** 2)),
                    "e_im_mean_abs2": float(np.mean(np.imag(E[ann]) ** 2)),
                    "z_hat_ann_power": float(np.mean(np.abs(Z_al_np[ann]) ** 2)),
                    "z_w_ann_power": float(np.mean(np.abs(Z_w_np[ann]) ** 2)),
                    "attenuation_db": float(10.0 * np.log10(
                        (np.mean(np.abs(Z_al_np[ann]) ** 2) + 1e-30) /
                        (np.mean(np.abs(Z_w_np[ann]) ** 2) + 1e-30))),
                    "S_true": float(s_true),
                    "S_false": s_false,
                    "delta_rel": (float((s_true - s_false) / s_true)
                                  if s_true > 0 else float("nan")),
                    "est_angle": float(grid[int(np.argmax(S))]),
                    "radial_profile": radial,
                    "angular_profile": ang,
                }
                for li, lay in enumerate(core.dl.layers):
                    a = layer_analysis(lay, E, Z_al_np, bits)
                    a.pop("ell")
                    for k, v in a.items():
                        row[f"layer{li}_{k}"] = v
                log.append(row)

                if args.save_npz:
                    np.savez_compressed(
                        os.path.join(npz_dir, f"{i:04d}_{case.replace('+','_')}.npz"),
                        E=E.astype(np.complex64), Z_w=Z_w_np.astype(np.complex64),
                        Z_hat=Z_al_np.astype(np.complex64),
                        ann=ann, true_angle=alpha, bits=bits)

            del img_w, z_no
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    from p0_common import load_rows

    rows = load_rows(rows_path)
    summary = summarise(rows, cases)
    write_json(os.path.join(args.out_dir, "summary.json"),
               {"config": run_meta, "summary": summary,
                "elapsed_sec": time.time() - t0})
    report(args.out_dir, run_meta, summary)
    print(f"\nelapsed {(time.time() - t0) / 60:.1f} min -> {args.out_dir}")


FIELDS = ["e_mean_abs2", "e_re_mean_abs2", "e_im_mean_abs2", "attenuation_db",
          "S_true", "S_false", "delta_rel", "est_angle", "z_hat_ann_power"]


def summarise(rows, cases):
    out = {}
    base = {}
    for case in cases:
        sub = [r for r in rows if r["case"] == case]
        if not sub:
            continue
        entry = {"n": len(sub)}
        for f in FIELDS:
            v = np.asarray([r[f] for r in sub], dtype=np.float64)
            entry[f] = float(v.mean())
            entry[f"{f}_std"] = float(v.std())
        for f in [k for k in sub[0] if k.startswith("layer")]:
            v = np.asarray([r[f] for r in sub], dtype=np.float64)
            entry[f] = float(v.mean())
        entry["radial_profile"] = np.nanmean(
            np.asarray([r["radial_profile"] for r in sub]), axis=0).tolist()
        entry["angular_profile"] = np.nanmean(
            np.asarray([r["angular_profile"] for r in sub]), axis=0).tolist()
        by_img = {}
        for r in sub:
            by_img.setdefault(r["image_id"], []).append(r)
        entry["per_image_snr"] = float(np.mean([
            np.mean([x["layer0_snr_carrier"] for x in v]) for v in by_img.values()]))
        out[case] = entry
        base[case] = entry
    clean = out.get("clean")
    if clean:
        for case, entry in out.items():
            for f in ("e_mean_abs2", "e_re_mean_abs2", "e_im_mean_abs2",
                      "layer0_snr_carrier", "layer1_snr_carrier"):
                if f in entry and f in clean:
                    entry[f"{f}_minus_clean"] = entry[f] - clean[f]
    return out


def report(out_dir, run_meta, summary) -> None:
    lines = ["# P0 Inversion-error spectrum（oracle 反旋转后）", "",
             f"载波 B={run_meta['n_bits']}，环带半径 {run_meta['radii'][0]}–"
             f"{run_meta['radii'][-1]}（{run_meta['annulus_points']} 个频点），"
             f"两层 K={run_meta['layer_K']}；每格 N={run_meta['N']}。",
             "",
             "$E(\\omega)=R_{-\\alpha}\\hat Z_T(\\omega)-Z_w(\\omega)$，"
             "全部指标都在环带与 channel 3 上计算。", ""]
    lines += ["| 条件 | ‖E‖²均值 | Re/Im 能量 | 衰减 dB | ‖E‖²−clean | "
              "层0 SNR | 层1 SNR | S_true | S_false | Δrel |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case, e in summary.items():
        lines.append(
            f"| {case} | {e['e_mean_abs2']:.3e} | "
            f"{e['e_re_mean_abs2']:.2e} / {e['e_im_mean_abs2']:.2e} | "
            f"{e['attenuation_db']:.2f} | {e.get('e_mean_abs2_minus_clean', float('nan')):.3e} | "
            f"{e.get('layer0_snr_carrier', float('nan')):.3f} | "
            f"{e.get('layer1_snr_carrier', float('nan')):.3f} | "
            f"{e['S_true']:.1f} | {e['S_false']:.1f} | {e['delta_rel']:.3f} |")
    lines += ["", "读法：", "",
              "- `‖E‖² − clean` 是相对**干净的 DDIM 反演误差底**的增量；"
              "若旋转+噪声只是把 clean 的误差放大，这一项会整体抬升。",
              "- `层i SNR` 是 carrier-projected SNR "
              "$\\overline{(b_j\\ell_j)^2}/\\overline{\\nu_j}$；"
              "它与 `S_true` 一起区分“真峰塌陷=信噪比下降”和"
              "“假峰不变=噪声底是各向同性”。",
              "- `Δrel=(S_true-S_false)/S_true` 是把同步判据写成相对 margin 的形式；"
              "Δrel 接近 0 时 argmax 已经不可靠。", ""]
    path = os.path.join(out_dir, "summary.md")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    safe_print("\n".join(lines))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
