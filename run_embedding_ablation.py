"""Step 1 of the v2 roadmap: replace vs host-preserving projection embedding.

The canonical embedding *replaces* the whole annulus with the watermark::

    Z[0, ch][band] = W,   W = b @ Psi   (energy-normalised to eta * N_A)

which discards the host spectrum on ~940 frequency points and squeezes each
layer into its 8 carrier dimensions.  Measured on 100 random 64x64 latents, that
costs a latent RMSE of 0.444 at eta=1e4 while a projection-only embedding costs
0.041 (a ~118x MSE gap), so the embedding operator -- not the energy knob -- is
the plausible root cause of the visible quality loss.

This script keeps ``swm/dual_layer.py`` untouched (frozen canonical core) and
implements the alternative embedding here, on top of the loaded design:

    a_j    = Re<Psi_j, Z> / ||Psi_j||^2          (the host's own projection)
    a'_j   = b_j * max(|a_j|, tau)               (payload bit, optional floor)
    Z'     = Z + sum_j (a'_j - a_j) Psi_j

so the orthogonal complement of the carrier subspace is left exactly as the host
had it, and carriers already carrying the right sign are not touched at all.
``tau`` is the new robustness knob; it only binds once it exceeds the natural
|a_j| scale, which measures at ~37 for a unit Gaussian latent and this design
(so tau = 0/20/40/80 is the informative sweep, not 0.5-2 sigma).

Everything is paired on the same prompt and the same starting latent, and the
script reports quality *and* robustness together -- a projection embedding that
looks great but loses the rotation margin is not a win.

    python run_embedding_ablation.py --N 20 --tau 0,20,40,80 --eta 1e4
    python run_embedding_ablation.py --selfcheck     # CPU, no diffusion
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from p0_common import (OursCore, ResumeLog, align_error360, bit_acc,
                       bits_from_rng, case_rotation_angle, clustered_paired_diff,
                       clustered_rate_ci, is_synced360, make_config,
                       perfect_match, provenance, resolve_device, safe_print,
                       search_grid, write_json)
from paper_protocol import AttackCtx, apply_case, case_seed


# --------------------------------------------------------------------------- #
# the two embeddings (no canonical file is modified)
# --------------------------------------------------------------------------- #
def embed_replace(dl, z_T: torch.Tensor, bits: np.ndarray) -> torch.Tensor:
    """The canonical path, called through the existing method."""
    return dl.embed(z_T, bits)


def embed_project(dl, z_T: torch.Tensor, bits: np.ndarray,
                  tau: float = 0.0) -> torch.Tensor:
    """Host-preserving carrier-projection embedding (see module docstring)."""
    ch, size = dl.channel, dl.size
    Z = torch.fft.fftshift(torch.fft.fft2(z_T.float(), dim=(-1, -2)),
                           dim=(-1, -2)).clone()
    b = torch.as_tensor(np.asarray(bits), dtype=torch.float32,
                        device=z_T.device)
    for lay in dl.layers:
        Psi = lay["Psi"]                       # (B, P) complex64
        norms = lay["norms_t"]                 # (B,)
        mask = lay["mask_t"]                   # (size, size) bool
        zv = Z[0, ch][mask]
        a = (torch.conj(Psi) * zv[None, :]).sum(dim=1).real / norms
        ap = b * torch.clamp(a.abs(), min=float(tau))
        delta = ((ap - a)[:, None] * Psi).sum(dim=0)
        Z[0, ch][mask] = zv + delta
    z_w = torch.fft.ifft2(torch.fft.ifftshift(Z, dim=(-1, -2)),
                          dim=(-1, -2)).real
    return z_w.to(z_T.dtype)


def carrier_split(dl, z_T: torch.Tensor) -> Dict[str, float]:
    """How much of the latent difference lies inside the carrier subspaces.

    For ``replace`` most of the change is inside the carrier span *and* the span
    is rebuilt; for ``project`` **all** of the change must be inside it (the
    orthogonal complement is untouched).  This is the diagnostic that shows the
    two operators differ structurally, not just in magnitude.
    """
    out = {}
    Z0 = torch.fft.fftshift(torch.fft.fft2(z_T.float(), dim=(-1, -2)),
                            dim=(-1, -2))
    for li, lay in enumerate(dl.layers):
        mask = lay["mask_t"]
        d = (Z0[0, dl.channel][mask]).clone()
        out[f"layer{li}_host_energy"] = float((d.abs() ** 2).sum())
    return out


def orthogonal_residual(dl, z_a: torch.Tensor, z_b: torch.Tensor) -> Dict:
    """Decompose ``Z_a - Z_b`` into carrier-parallel and orthogonal parts."""
    ch = dl.channel
    Za = torch.fft.fftshift(torch.fft.fft2(z_a.float(), dim=(-1, -2)),
                            dim=(-1, -2))
    Zb = torch.fft.fftshift(torch.fft.fft2(z_b.float(), dim=(-1, -2)),
                            dim=(-1, -2))
    tot = par = 0.0
    for lay in dl.layers:
        m = lay["mask_t"]
        d = (Za[0, ch][m] - Zb[0, ch][m])
        Psi = lay["Psi"]
        norms = lay["norms_t"]
        coef = (torch.conj(Psi) * d[None, :]).sum(dim=1) / norms
        d_par = (coef[:, None] * Psi).sum(dim=0)
        tot += float((d.abs() ** 2).sum())
        par += float((d_par.abs() ** 2).sum())
    return {"total": tot, "parallel": par,
            "orthogonal": max(tot - par, 0.0),
            "parallel_frac": par / max(tot, 1e-30),
            "orthogonal_frac": max(tot - par, 0.0) / max(tot, 1e-30)}


def selfcheck(design: str, seed: int = 0) -> None:
    """CPU validation: bits recoverable, orthogonal complement untouched."""
    from swm.dual_layer import DualLayerWatermarker

    dl = DualLayerWatermarker(design, size=64, channel=3, device="cpu",
                              energy_per_point=1e4)
    core = OursCore(design, device="cpu", energy_per_point=1e4)
    rng = np.random.RandomState(seed)
    print(f"{'variant':18s} {'latent RMSE':>12s} {'par frac':>9s} "
          f"{'orth frac':>10s} {'clean BitAcc':>13s}")
    for name, fn in [("replace eta=1e4", lambda z, b: embed_replace(dl, z, b)),
                     ("project tau=0", lambda z, b: embed_project(dl, z, b, 0)),
                     ("project tau=40", lambda z, b: embed_project(dl, z, b, 40)),
                     ("project tau=80", lambda z, b: embed_project(dl, z, b, 80))]:
        rmse, pf, of, acc = [], [], [], []
        for _ in range(20):
            z = torch.randn(1, 4, 64, 64)
            bits = np.where(rng.randint(0, 2, 8) > 0, 1.0, -1.0).astype(np.float32)
            zw = fn(z, bits)
            rmse.append(float((zw - z).pow(2).mean().sqrt()))
            spl = orthogonal_residual(dl, zw, z)
            pf.append(spl["parallel_frac"])
            of.append(spl["orthogonal_frac"])
            ell = core.ell_mean(zw, 0.0)
            acc.append(bit_acc(np.where(bits > 0, 1, 0), (ell > 0).astype(int)))
        print(f"{name:18s} {np.mean(rmse):12.4f} {np.mean(pf):9.4f} "
              f"{np.mean(of):10.4f} {np.mean(acc):13.4f}")
    print("\n(replace: most of the change rebuilds the carrier span; "
          "project: orthogonal_frac must be ~0 by construction.)")


# --------------------------------------------------------------------------- #
# experiment
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="results/design_searched_realgeom.npz")
    ap.add_argument("--arms", default="",
                    help="explicit arm list, e.g. replace:1e4,project:0,project:40; "
                         "default = replace at --eta plus project at each --tau")
    ap.add_argument("--eta", type=float, default=1e4)
    ap.add_argument("--tau", default="0,20,40,80")
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--cases", default="clean,rot45,rot75,rot+noise0.05")
    ap.add_argument("--grid_step", type=float, default=2.0)
    ap.add_argument("--sync_tol", type=float, default=2.0)
    ap.add_argument("--lpips", action="store_true", default=True)
    ap.add_argument("--no_lpips", dest="lpips", action="store_false")
    ap.add_argument("--model_id", default="/root/autodl-tmp/models/sd21base")
    ap.add_argument("--dataset",
                    default="coco:/root/autodl-tmp/data/coco5k/meta_data.json")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out_dir", default="runs/embedding_ablation")
    ap.add_argument("--report_path", default="results/embedding_ablation.md")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selfcheck", action="store_true")
    return ap.parse_args()


def parse_arms(args) -> List[Tuple[str, float]]:
    if args.arms:
        out = []
        for spec in args.arms.split(","):
            name, _, val = spec.partition(":")
            out.append((name.strip(), float(val or 0)))
        return out
    arms = [("replace", float(args.eta))]
    for t in args.tau.split(","):
        if t.strip():
            arms.append(("project", float(t)))
    return arms


def main() -> None:
    args = parse_args()
    if args.selfcheck:
        selfcheck(args.design)
        return

    from pipeline.optim_utils import get_dataset, set_random_seed
    from run_exp import generate, invert

    arms = parse_arms(args)
    cases = [c for c in args.cases.split(",") if c]
    device = resolve_device(args.device)
    cfg = make_config(args, device)
    from p0_common import build_pipe

    pipe = build_pipe(cfg, device)
    dataset, prompt_key = get_dataset(cfg.dataset)
    emb = pipe.get_text_embedding("")
    grid = search_grid(args.grid_step)
    core = OursCore(args.design, device=device, energy_per_point=args.eta)
    dl = core.dl
    ctx = AttackCtx(device=device)

    lpips_fn = None
    if args.lpips:
        try:
            import lpips

            lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
        except Exception as exc:
            safe_print(f"[warn] LPIPS unavailable ({exc})")

    os.makedirs(args.out_dir, exist_ok=True)
    rows_path = os.path.join(args.out_dir, "rows.jsonl")
    run_meta = {"script": "run_embedding_ablation.py", "design": args.design,
                "arms": [f"{n}:{v:g}" for n, v in arms], "N": args.N,
                "cases": cases, "grid_step": args.grid_step,
                "sync_tol": args.sync_tol, "provenance": provenance()}
    write_json(os.path.join(args.out_dir, "run_config.json"), run_meta)

    rng_w = np.random.RandomState(cfg.w_seed)
    t0 = time.time()
    with ResumeLog(rows_path, keys=("uid",), fingerprint=run_meta) as log:
        for i in tqdm(range(args.start, args.start + args.N), desc="embed-ablation"):
            seed = i + cfg.gen_seed
            prompt = dataset[i][prompt_key]
            set_random_seed(seed)
            z_no = pipe.get_random_latents()
            bits = bits_from_rng(rng_w, core.n_bits)
            true_bits = np.where(bits > 0, 1, 0)

            # no-watermark reference generation
            img_ref = generate(pipe, prompt, z_no, cfg, device)
            ref_dir = os.path.join(args.out_dir, "images", "no_wm")
            os.makedirs(ref_dir, exist_ok=True)
            img_ref.save(os.path.join(ref_dir, f"sample{i:03d}.png"))

            for name, val in arms:
                uid0 = f"{i}|{name}:{val:g}"
                if log.has({"uid": uid0 + "|clean"}):
                    continue
                if name == "replace":
                    dl.energy_per_point = float(val)
                    z_w = embed_replace(dl, z_no, bits)
                else:
                    z_w = embed_project(dl, z_no, bits, tau=val)
                img_w = generate(pipe, prompt, z_w, cfg, device)
                arm_dir = os.path.join(args.out_dir, "images",
                                       f"{name}_{val:g}")
                os.makedirs(arm_dir, exist_ok=True)
                img_w.save(os.path.join(arm_dir, f"sample{i:03d}.png"))

                q = {"latent_rmse": float((z_w - z_no).pow(2).mean().sqrt())}
                x = np.asarray(img_ref).astype(np.float64) / 255.0
                y = np.asarray(img_w).astype(np.float64) / 255.0
                mse = float(np.mean((x - y) ** 2))
                q["psnr"] = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
                try:
                    from skimage.metrics import structural_similarity

                    q["ssim"] = float(structural_similarity(
                        x, y, channel_axis=-1, data_range=1.0))
                except Exception:
                    q["ssim"] = float("nan")
                if lpips_fn is not None:
                    from torchvision.transforms.functional import to_tensor

                    dev = next(lpips_fn.parameters()).device
                    with torch.no_grad():
                        q["lpips"] = float(lpips_fn(
                            to_tensor(img_ref).unsqueeze(0).to(dev) * 2 - 1,
                            to_tensor(img_w).unsqueeze(0).to(dev) * 2 - 1).item())
                else:
                    q["lpips"] = float("nan")
                q.update(orthogonal_residual(dl, z_w, z_no))

                for case in cases:
                    theta = float(np.random.RandomState(seed + 777)
                                  .uniform(0.0, 180.0))
                    alpha = case_rotation_angle(case, theta)
                    rng_a = np.random.RandomState(case_seed(seed, case))
                    img_a = apply_case(img_w, case, theta, rng_a, ctx)
                    z_hat = invert(pipe, img_a, emb, cfg, device)
                    S = core.landscape(z_hat, grid)
                    est = float(grid[int(np.argmax(S))])
                    ell = core.ell_mean(z_hat, est)
                    ell_or = core.ell_mean(z_hat, alpha)
                    log.append({
                        "uid": uid0 + f"|{case}", "image_id": int(i),
                        "arm": f"{name}:{val:g}", "embed_mode": name,
                        "arm_value": float(val), "case": case,
                        "true_angle": float(alpha), "est_angle": est,
                        "est_error_deg360": align_error360(est, alpha),
                        "synced360": is_synced360(est, alpha, args.sync_tol),
                        "detection_score": float(S.max()),
                        "bit_acc": bit_acc(true_bits, (ell > 0).astype(int)),
                        "perfect": perfect_match(true_bits, (ell > 0).astype(int)),
                        "bit_acc_oracle": bit_acc(true_bits,
                                                  (ell_or > 0).astype(int)),
                        "perfect_oracle": perfect_match(
                            true_bits, (ell_or > 0).astype(int)),
                        **q,
                    })
            del img_ref, z_no
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    from p0_common import load_rows, wilson

    rows = load_rows(rows_path)
    summary: Dict[str, Dict] = {}
    for arm, _ in [(f"{n}:{v:g}", v) for n, v in arms]:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        q = {f: float(np.nanmean([r[f] for r in sub]))
             for f in ("psnr", "ssim", "lpips", "latent_rmse",
                       "parallel_frac", "orthogonal_frac")}
        per_case = {}
        for case in cases:
            s = [r for r in sub if r["case"] == case]
            if not s:
                continue
            f = [0.0 if r["synced360"] else 1.0 for r in s]
            cc = clustered_rate_ci(f, [r["image_id"] for r in s])
            per_case[case] = {
                "n": len(s), "bit_acc": float(np.mean([r["bit_acc"] for r in s])),
                "pmr": float(np.mean([r["perfect"] for r in s])),
                "sync_fail": cc["rate_record"],
                "sync_fail_ci": cc["ci_record"],
                "bit_acc_oracle": float(np.mean(
                    [r["bit_acc_oracle"] for r in s])),
            }
        summary[arm] = {"quality": q, "cases": per_case}
    write_json(os.path.join(args.out_dir, "summary.json"),
               {"config": run_meta, "summary": summary,
                "elapsed_sec": time.time() - t0})

    lines = ["# 第一步：replace vs projection embedding（质量 + 鲁棒性一起看）", "",
             f"载波 `{os.path.basename(args.design)}`（canonical，未修改）；N={args.N}；"
             f"所有格共用同一 prompt 与同一起始潜变量。", "",
             "## 质量（相对同一 latent 的无水印图）", "",
             "| arm | 潜变量 RMSE | 载波内占比 | 正交补占比 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, d in summary.items():
        q = d["quality"]
        lines.append(f"| `{arm}` | {q['latent_rmse']:.4f} | "
                     f"{q['parallel_frac']:.4f} | {q['orthogonal_frac']:.4f} | "
                     f"{q['psnr']:.2f} | {q['ssim']:.4f} | {q['lpips']:.4f} |")
    lines += ["", "## 鲁棒性（BitAcc / PMR / mod360 失锁率）", "",
              "| arm | " + " | ".join(cases) + " |",
              "|---|" + "---|" * len(cases)]
    for arm, d in summary.items():
        cells = []
        for case in cases:
            c = d["cases"].get(case)
            cells.append("--" if not c else
                         f"{c['bit_acc']:.3f} / {c['pmr']:.3f} / {c['sync_fail']:.3f}")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    lines += ["", "单元格 = BitAcc / PMR / 失锁率（mod360，按图像聚类）。", "",
              "判读：只有某个 τ 同时满足「质量明显优于 replace」且"
              "「rot45/rot75 不失锁、rot+noise 不显著更差」时，才值得进入第二步"
              "（改同步 score）；若投影法牺牲了旋转余量，就要在报告里写清楚。", ""]
    os.makedirs(os.path.dirname(os.path.abspath(args.report_path)), exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    safe_print("\n".join(lines))
    print(f"\nelapsed {(time.time() - t0) / 60:.1f} min -> {args.report_path}")


if __name__ == "__main__":
    main()
