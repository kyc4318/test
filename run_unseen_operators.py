"""P0 gate 2b: rotation-operator generalisation (unseen rotation operators).

The carrier search optimises ``rho`` under a *latent-space* rotation
(``TF.rotate``, bilinear, ``fill=0``, centre 31.5) and every end-to-end run so
far applied the attack with ``PIL Image.rotate`` (bilinear, zero fill) on the
512x512 image.  A reviewer will therefore ask whether the reported rotation
robustness is a property of the carrier or of that particular implementation.

This script evaluates a two-axis matrix.

Attack operators (applied to the 512x512 image):
    ``pil_bilinear``       PIL, bilinear, zero fill -- the operator used so far
    ``pil_bicubic``        PIL, bicubic
    ``pil_nearest``        PIL, nearest neighbour
    ``pil_expand_crop``    PIL bilinear, ``expand=True`` then centre-crop back:
                           no zero-filled wedges, plus one extra resampling
    ``cv2_linear_reflect`` OpenCV ``warpAffine``, linear, reflect border
    ``cv2_cubic_constant`` OpenCV ``warpAffine``, cubic, constant(0) border

Decoder operators (applied to the recovered 64x64 latent at ``-angle``):
    ``tv_nearest_c31.5``   torchvision nearest about 31.5 -- **this is the
                           operator the design search and every existing run
                           actually used**: ``TF.rotate`` defaults to
                           ``InterpolationMode.NEAREST``, so
                           ``rotate_latent(z, a) == TF.rotate(z, a, NEAREST)``
                           exactly (checked in ``selfcheck_operators``).
                           The draft's "bilinear resampling" wording for the
                           decoder does not match the code.
    ``tv_bilinear_c31.5``  torchvision bilinear about 31.5 (unseen)
    ``tv_bicubic_c31.5``   bicubic about 31.5 via ``grid_sample`` (unseen)
    ``tv_nearest_c32``     nearest about 32, i.e. the DFT origin
    ``tv_bilinear_c32``    bilinear about 32: the half-pixel mistake of
                           Proposition 3, with a smooth kernel
    ``pil_nearest_c31.5``  PIL nearest on the latent (different library)
    ``pil_bilinear_c31.5`` PIL bilinear on the latent (different library)
    ``pil_bicubic_c31.5``  PIL bicubic on the latent (different library)

For every (attack, decoder) cell the script reports the sync failure rate, the
alignment error and the payload accuracy with Wilson intervals, plus
``S_true``/``S_false``/``delta_rel``.  The cell that uses the operators seen so
far (``pil_bilinear`` x ``tv_bilinear_c31.5``) is the reference; the other
cells measure the generalisation gap.

Pre-registered reading of the outcome:
  (i)   if every attack-operator cell stays within the Wilson interval of the
        reference under the canonical decoder, the rotation robustness is a
        property of the carrier/inversion channel rather than of the operator;
  (ii)  if only the ``*_c32`` decoder cells degrade, the gap is the half-pixel
        geometry of Proposition 3 -- expected, since the method *defines* its
        decoder to rotate about the physical centre;
  (iii) if the cv2 / expand-crop attack cells degrade under the canonical
        decoder, the design search is matched to one interpolation geometry,
        and that overfitting must be reported as a limitation.

Nothing in the canonical core is touched: the pluggable decoder is built by
subclassing ``p0_common.OursCore`` and overriding the two methods that rotate.

    python run_unseen_operators.py --N 15 \
        --attack_ops pil_bilinear,pil_bicubic,cv2_linear_reflect \
        --decode_ops tv_bilinear_c31.5,tv_bicubic_c31.5 \
        --angles 37.3,58.7,102.5,-30 --grid_step 2
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

from p0_common import (OursCore, ResumeLog, add_gaussian_noise, align_error,
                       bit_acc, bits_from_rng, bootstrap_ci, is_synced,
                       load_rows, make_config, perfect_match, provenance,
                       resolve_device, safe_print, search_grid, signed_error,
                       wilson, write_json)

ATTACK_OPS = ("pil_bilinear", "pil_bicubic", "pil_nearest", "pil_expand_crop",
              "cv2_linear_reflect", "cv2_cubic_constant", "cv2_cubic_reflect")
CV2_OPS = tuple(o for o in ATTACK_OPS if o.startswith("cv2_"))
# torchvision only supports nearest/bilinear for tensor input, so the bicubic
# decoder variants go through grid_sample instead (see ``_grid_rotate``).
# ``tv_nearest_c31.5`` is the canonical one: TF.rotate defaults to NEAREST.
DECODE_OPS = ("tv_nearest_c31.5", "tv_bilinear_c31.5", "tv_bicubic_c31.5",
              "tv_nearest_c32", "tv_bilinear_c32", "pil_nearest_c31.5",
              "pil_bilinear_c31.5", "pil_bicubic_c31.5")
REFERENCE = ("pil_bilinear", "tv_nearest_c31.5")


# --------------------------------------------------------------------------- #
# attack-side image rotation operators
# --------------------------------------------------------------------------- #
def _cv2_flags(op: str):
    import cv2

    return {
        "cv2_linear_reflect": (cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101),
        "cv2_cubic_constant": (cv2.INTER_CUBIC, cv2.BORDER_CONSTANT),
        "cv2_cubic_reflect": (cv2.INTER_CUBIC, cv2.BORDER_REFLECT_101),
    }[op]


def attack_rotate(img: Image.Image, deg: float, op: str) -> Image.Image:
    """Rotate a PIL image counter-clockwise by ``deg`` with operator ``op``."""
    deg = float(deg)
    if op == "pil_bilinear":
        return img.rotate(deg, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    if op == "pil_bicubic":
        return img.rotate(deg, resample=Image.BICUBIC, fillcolor=(0, 0, 0))
    if op == "pil_nearest":
        return img.rotate(deg, resample=Image.NEAREST, fillcolor=(0, 0, 0))
    if op == "pil_expand_crop":
        w, h = img.size
        big = img.rotate(deg, resample=Image.BILINEAR, expand=True,
                         fillcolor=(0, 0, 0))
        bw, bh = big.size
        left, top = (bw - w) // 2, (bh - h) // 2
        return big.crop((left, top, left + w, top + h))
    if op in CV2_OPS:
        import cv2

        flags, border = _cv2_flags(op)
        arr = np.asarray(img)
        h, w = arr.shape[:2]
        # PIL rotates about the pixel-index centre (w-1)/2, (h-1)/2; matching it
        # keeps the geometry comparable and isolates interpolation/border.
        M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), deg, 1.0)
        out = cv2.warpAffine(arr, M, (w, h), flags=flags, borderMode=border,
                             borderValue=(0, 0, 0))
        return Image.fromarray(out)
    raise ValueError(f"unknown attack operator: {op!r}")


# --------------------------------------------------------------------------- #
# decoder-side latent anti-rotation operators
# --------------------------------------------------------------------------- #
def _tv():
    from torchvision.transforms import functional as TF

    return TF


def _tv_affine_grid(theta, w: int, h: int):
    """Replicate torchvision's ``_gen_affine_grid`` (transforms/_functional_tensor.py).

    torchvision does *not* use ``F.affine_grid``: it builds the base grid over
    ``[-w/2 + 0.5, w/2 - 0.5]`` (i.e. pixel-index centre ``(w-1)/2`` at zero) and
    then normalises ``theta`` by the half-size.  Feeding ``_get_inverse_affine_matrix``
    straight into ``F.affine_grid`` therefore mis-scales the translation column
    and silently moves the rotation centre, which is why the exact copy below is
    used instead of the library call.
    """
    import torch

    d = 0.5
    base = torch.empty(1, h, w, 3, dtype=theta.dtype, device=theta.device)
    x = torch.linspace(-w * 0.5 + d, w * 0.5 + d - 1, steps=w,
                       device=theta.device)
    y = torch.linspace(-h * 0.5 + d, h * 0.5 + d - 1, steps=h,
                       device=theta.device).unsqueeze_(-1)
    base[..., 0].copy_(x)
    base[..., 1].copy_(y)
    base[..., 2].fill_(1)
    rescaled = theta.transpose(1, 2) / torch.tensor(
        [0.5 * w, 0.5 * h], dtype=theta.dtype, device=theta.device)
    return base.view(1, h * w, 3).bmm(rescaled).view(1, h, w, 2)


# Rotation centres in ``_get_inverse_affine_matrix`` coordinates.  The origin of
# that frame is the *physical* image centre, i.e. pixel index (size-1)/2 = 31.5
# for a 64x64 latent, which is what ``TF.rotate`` uses by default (verified:
# ``_grid_rotate(..., CENTER_PHYSICAL, mode)`` reproduces ``TF.rotate`` exactly).
# The DFT origin sits on pixel index 32, half a pixel away.
CENTER_PHYSICAL = (0.0, 0.0)
CENTER_DFT_ORIGIN = (0.5, 0.5)


def _grid_rotate(z, deg_ccw: float, center, mode: str):
    """Rotate a ``(1,C,H,W)`` tensor counter-clockwise about ``(cx, cy)``.

    ``TF.affine``'s positive angle runs *clockwise* while ``TF.rotate``'s runs
    counter-clockwise (verified on an impulse: ``TF.rotate(+37.3)`` moves
    (row 40, col 50) to (27, 51) whereas ``TF.affine(+37.3)`` moves it to
    (50, 41)).  ``_get_inverse_affine_matrix`` follows the ``TF.affine``
    convention, hence the negated angle.  The rotation is then applied with
    ``grid_sample``, the only path that offers bicubic on tensor input; the
    bilinear path is checked against ``TF.rotate`` at startup.
    """
    import torch
    import torch.nn.functional as F
    from torchvision.transforms.functional import _get_inverse_affine_matrix

    # the latent recovered by DDIM inversion is fp16, while ``grid_sample``
    # requires the grid to share the input dtype; do the resampling in fp32 and
    # cast back (slightly *more* precise than the canonical fp16 path).
    out_dtype = z.dtype
    zf = z.float()
    theta = _get_inverse_affine_matrix([float(center[0]), float(center[1])],
                                       -float(deg_ccw), [0.0, 0.0], 1.0,
                                       [0.0, 0.0])
    theta = torch.as_tensor(theta, dtype=torch.float32,
                            device=zf.device).view(1, 2, 3)
    grid = _tv_affine_grid(theta, w=int(zf.shape[-1]), h=int(zf.shape[-2]))
    out = F.grid_sample(zf, grid, mode=mode, padding_mode="zeros",
                        align_corners=False)
    return out.to(out_dtype)


def selfcheck_operators(verbose: bool = True) -> dict:
    """Check the decoder operators against the canonical ``TF.rotate``.

    Three checks, all of which must pass before any result is interpretable:
      (a) ``rotate_latent`` (the canonical decoder call) equals an explicit
          ``TF.rotate(..., NEAREST)`` -- i.e. the canonical operator is
          nearest, not bilinear;
      (b) ``_grid_rotate(..., centre 31.5, bilinear)`` reproduces
          ``TF.rotate(..., BILINEAR)``;
      (c) ``_grid_rotate(..., CENTER_PHYSICAL, nearest)`` reproduces
          ``TF.rotate(..., NEAREST)``.

    Deviations are reported twice: over the whole tensor and inside the
    *decoding disc* (radius <= 24).  ``torchvision`` blends towards ``fill``
    through an extra mask channel, so the two bilinear implementations differ in
    the rotated corners -- pixels the decoder never reads, since the annulus has
    radius <= 20 and therefore stays >= 12 px away from the border.  The
    assertion must be on the disc.
    """
    import torch

    from run_paper_compare import rotate_latent

    torch.manual_seed(0)
    z = torch.randn(1, 4, 64, 64)
    size = z.shape[-1]
    yy, xx = np.ogrid[:size, :size]
    c0 = (size - 1) / 2.0
    disc = np.hypot(yy - c0, xx - c0) <= 24.0
    disc_t = torch.from_numpy(disc)

    def dev(a, b):
        d = (a - b).abs()
        return float(d.max()), float(d[..., disc_t].max())

    out = {}
    for deg in (-37.3, 20.0, 90.0):
        TF = _tv()
        near = TF.rotate(z, float(deg),
                         interpolation=TF.InterpolationMode.NEAREST, fill=0)
        bilin = TF.rotate(z, float(deg),
                          interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        out[float(deg)] = {
            "rotate_latent_vs_nearest": dev(rotate_latent(z, float(deg)),
                                            near),
            "grid_bilinear_vs_tv_bilinear": dev(
                _grid_rotate(z, float(deg), CENTER_PHYSICAL, "bilinear"),
                bilin),
            "grid_nearest_vs_tv_nearest": dev(
                _grid_rotate(z, float(deg), CENTER_PHYSICAL, "nearest"),
                near),
        }
    if verbose:
        for deg, checks in out.items():
            print(f"  selfcheck deg={deg:+.1f}: " + "  ".join(
                f"{k}=(full {v[0]:.2e}, disc {v[1]:.2e})"
                for k, v in checks.items()))
    return out


def decode_rotate(z, angle: float, op: str):
    """Undo a rotation of ``angle`` on the recovered latent with operator ``op``.

    The canonical decoder is ``TF.rotate(z, -angle, fill=0)``, which -- because
    ``TF.rotate`` defaults to nearest -- is ``tv_nearest_c31.5`` here.
    """
    deg = -float(angle)          # the anti-rotation, counter-clockwise
    TF = _tv()
    if op == "tv_nearest_c31.5":
        return TF.rotate(z, deg, interpolation=TF.InterpolationMode.NEAREST,
                         fill=0)
    if op == "tv_bilinear_c31.5":
        return TF.rotate(z, deg, interpolation=TF.InterpolationMode.BILINEAR,
                         fill=0)
    if op == "tv_bicubic_c31.5":
        return _grid_rotate(z, deg, CENTER_PHYSICAL, "bicubic")
    if op == "tv_nearest_c32":
        return _grid_rotate(z, deg, CENTER_DFT_ORIGIN, "nearest")
    if op == "tv_bilinear_c32":
        return _grid_rotate(z, deg, CENTER_DFT_ORIGIN, "bilinear")
    if op in ("pil_nearest_c31.5", "pil_bilinear_c31.5",
              "pil_bicubic_c31.5"):
        import torch

        resample = {"pil_nearest_c31.5": Image.NEAREST,
                    "pil_bilinear_c31.5": Image.BILINEAR,
                    "pil_bicubic_c31.5": Image.BICUBIC}[op]
        arr = z.detach().cpu().float().numpy()
        out = np.empty_like(arr)
        for c in range(arr.shape[1]):
            im = Image.fromarray(arr[0, c], mode="F")
            # PIL's centre argument is offset by +0.5 px from the pixel index
            # (c = pixel_index + 0.5), so PIL's *default* centre of
            # (size/2, size/2) = (32, 32) is the physical centre 31.5 -- the
            # same pivot TF.rotate uses.  Passing 31.5 here would rotate about
            # pixel 31 and silently add a half-pixel error.
            out[0, c] = np.asarray(im.rotate(deg, resample=resample,
                                             fillcolor=0.0))
        return torch.from_numpy(out).to(z.device, dtype=z.dtype)
    raise ValueError(f"unknown decoder operator: {op!r}")


class PluggableCore(OursCore):
    """``OursCore`` with the latent anti-rotation operator made pluggable.

    Only the three methods that touch the rotation are overridden, so the
    matched-filter decoding and the product fusion stay byte-identical to the
    canonical path.
    """

    def __init__(self, design: str, decode_op: str, **kwargs):
        super().__init__(design, **kwargs)
        self.decode_op = decode_op

    def _anti(self, z_hat, angle: float):
        return decode_rotate(z_hat, angle, self.decode_op)

    def ell_at(self, z_hat, angle: float) -> np.ndarray:
        ells = self.dl.decode_layers(self._anti(z_hat, angle))[0]
        return np.stack([e.detach().cpu().numpy().astype(np.float64)
                         for e in ells])

    def ell_mean(self, z_hat, angle: float) -> np.ndarray:
        return self.ell_at(z_hat, angle).mean(axis=0)

    def score_at(self, z_hat, angle: float) -> float:
        _, scores = self.dl.decode_layers(self._anti(z_hat, angle))
        out = 1.0
        for s in scores:
            out *= float(s)
        return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="results/design_searched_realgeom.npz")
    ap.add_argument("--attack_ops", default="pil_bilinear,pil_bicubic,"
                                           "pil_expand_crop,cv2_linear_reflect,"
                                           "cv2_cubic_constant")
    ap.add_argument("--decode_ops", default="tv_nearest_c31.5,"
                                            "tv_bilinear_c31.5,"
                                            "tv_bicubic_c31.5,"
                                            "tv_bilinear_c32")
    ap.add_argument("--angles", default="37.3,58.7,102.5,-30")
    ap.add_argument("--sigmas", default="0")
    ap.add_argument("--N", type=int, default=15)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--grid_step", type=float, default=2.0,
                    help="decoder-side angle search resolution")
    ap.add_argument("--sync_tol", type=float, default=2.0)
    ap.add_argument("--false_window", type=float, default=10.0)
    ap.add_argument("--energy_per_point", type=float, default=1e4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--model_id", default="/root/autodl-tmp/models/sd21base")
    ap.add_argument("--dataset",
                    default="coco:/root/autodl-tmp/data/coco5k/meta_data.json")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out_dir", default="runs/p0_operators")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    from run_exp import generate, invert
    from pipeline.optim_utils import get_dataset, set_random_seed

    attack_ops = [o for o in args.attack_ops.split(",") if o]
    decode_ops = [o for o in args.decode_ops.split(",") if o]
    angles = [float(a) for a in args.angles.split(",") if a]
    sigmas = [float(s) for s in args.sigmas.split(",") if s]
    for op in attack_ops:
        if op not in ATTACK_OPS:
            raise SystemExit(f"unknown attack operator {op!r}; "
                             f"choose from {ATTACK_OPS}")
    for op in decode_ops:
        if op not in DECODE_OPS:
            raise SystemExit(f"unknown decoder operator {op!r}; "
                             f"choose from {DECODE_OPS}")

    # the centre-32 / bicubic cells are only interpretable if the grid
    # replication reproduces torchvision exactly; fail loudly otherwise
    checks = selfcheck_operators(verbose=False)
    worst = max(v[1] for per_deg in checks.values() for v in per_deg.values())
    if worst > 1e-4:
        raise SystemExit(
            f"decoder-operator selfcheck failed (max deviation inside the "
            f"decoding disc {worst:.3e}); the operator comparison would not "
            "be interpretable")
    print(f"[selfcheck] decoder operators reproduce torchvision "
          f"(max deviation inside the decoding disc {worst:.2e})")

    device = resolve_device(args.device)
    cfg = make_config(args, device)
    from p0_common import build_pipe

    pipe = build_pipe(cfg, device)
    dataset, prompt_key = get_dataset(cfg.dataset)
    emb = pipe.get_text_embedding("")

    grid = search_grid(args.grid_step)
    cores = {op: PluggableCore(args.design, decode_op=op, device=device,
                               energy_per_point=args.energy_per_point)
             for op in decode_ops}
    ref_core = cores[decode_ops[0]]

    tag = args.tag or f"step{args.grid_step:g}_N{args.N}"
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    run_meta = {
        "script": "run_unseen_operators.py",
        "design": args.design,
        "n_bits": ref_core.n_bits,
        "attack_ops": attack_ops,
        "decode_ops": decode_ops,
        "reference_cell": list(REFERENCE),
        "angles": angles,
        "sigmas": sigmas,
        "grid_step": args.grid_step,
        "grid_n": int(len(grid)),
        "sync_tol": args.sync_tol,
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
        for i in tqdm(range(args.start, args.start + args.N),
                      desc="unseen-ops"):
            seed = i + cfg.gen_seed
            prompt = dataset[i][prompt_key]
            set_random_seed(seed)
            z_no = pipe.get_random_latents()
            bits = bits_from_rng(rng_w, ref_core.n_bits)
            true_bits = np.where(bits > 0, 1, 0)
            z_w = ref_core.embed(z_no, bits)
            img_w = generate(pipe, prompt, z_w, cfg, device)

            # one inversion per (angle, sigma, attack op) is shared by every
            # decoder op, so the decoder axis costs only the landscape
            for aop in attack_ops:
                for angle in angles:
                    for sigma in sigmas:
                        pending = [d for d in decode_ops if not log.has(
                            {"uid": f"{i}|{aop}|{angle:.4f}|{sigma:g}|{d}"})]
                        if not pending:
                            continue
                        img_a = attack_rotate(img_w, angle, aop)
                        if sigma > 0:
                            img_a = add_gaussian_noise(
                                img_a, sigma,
                                abs(int(angle * 1000)) + int(sigma * 1e4)
                                + int(seed))
                        t_inv = time.time()
                        z_hat = invert(pipe, img_a, emb, cfg, device)
                        inv_sec = time.time() - t_inv

                        for dop in pending:
                            core = cores[dop]
                            t_dec = time.time()
                            S = core.landscape(z_hat, grid)
                            best = int(np.argmax(S))
                            est = float(grid[best])
                            s_true = core.score_at(z_hat, angle)
                            far = np.asarray([align_error(g, angle)
                                              > args.false_window for g in grid])
                            s_false = float(S[far].max()) if far.any() \
                                else float("nan")
                            ell_est = core.ell_mean(z_hat, est)
                            ell_true = core.ell_mean(z_hat, angle)
                            dec_sec = time.time() - t_dec
                            log.append({
                                "uid": f"{i}|{aop}|{angle:.4f}|{sigma:g}|{dop}",
                                "image_id": int(i),
                                "prompt_id": int(i),
                                "seed": int(seed),
                                "attack_op": aop,
                                "decode_op": dop,
                                "true_angle": float(angle),
                                "sigma": float(sigma),
                                "payload_bits": core.n_bits,
                                "est_angle": est,
                                "est_error_deg": align_error(est, angle),
                                "signed_error_deg": signed_error(est, angle),
                                "synced": is_synced(est, angle, args.sync_tol),
                                "detection_score": float(S[best]),
                                "S_true": float(s_true),
                                "S_false": s_false,
                                "delta_rel": (float((s_true - s_false) / s_true)
                                              if s_true > 0
                                              else float("nan")),
                                "decoded_bits": (ell_est > 0).astype(int).tolist(),
                                "decoded_bits_oracle":
                                    (ell_true > 0).astype(int).tolist(),
                                "bit_acc": bit_acc(true_bits,
                                                   (ell_est > 0).astype(int)),
                                "perfect": perfect_match(
                                    true_bits, (ell_est > 0).astype(int)),
                                "bit_acc_oracle": bit_acc(
                                    true_bits, (ell_true > 0).astype(int)),
                                "inv_sec": float(inv_sec),
                                "decode_sec": float(dec_sec),
                            })
            del img_w, z_no
            if _cuda():
                import torch

                torch.cuda.empty_cache()

    rows = load_rows(rows_path)
    summary = summarise(rows, attack_ops, decode_ops, sigmas, args)
    write_json(os.path.join(out_dir, "summary.json"),
               {"config": run_meta, "summary": summary,
                "elapsed_sec": time.time() - t0})
    report(out_dir, run_meta, summary)
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
        "bit_acc_mean": float(bit.mean()),
        "bit_acc": bootstrap_ci(bit),
        "bit_acc_oracle_mean": float(bit_or.mean()),
        "pmr": {"mean": pmr, "ci": [plo, phi], "k": pmr_k},
        "delta_rel_mean": float(np.nanmean([r["delta_rel"] for r in rows])),
    }


def summarise(rows, attack_ops, decode_ops, sigmas, args):
    out = {"cells": {}}
    for aop in attack_ops:
        for dop in decode_ops:
            sub = [r for r in rows
                   if r["attack_op"] == aop and r["decode_op"] == dop]
            if not sub:
                continue
            out["cells"][f"{aop}|{dop}"] = _agg(sub)
    out["by_attack_op"] = {}
    for aop in attack_ops:
        sub = [r for r in rows if r["attack_op"] == aop]
        if sub:
            out["by_attack_op"][aop] = _agg(sub)
    out["by_decode_op"] = {}
    for dop in decode_ops:
        sub = [r for r in rows if r["decode_op"] == dop]
        if sub:
            out["by_decode_op"][dop] = _agg(sub)
    return out


def report(out_dir, run_meta, summary) -> None:
    attack_ops = run_meta["attack_ops"]
    decode_ops = run_meta["decode_ops"]
    cells = summary["cells"]
    ref_key = "|".join(REFERENCE)
    ref = cells.get(ref_key)

    lines = ["# P0 旋转算子泛化（unseen operators）", "",
             f"载波 `{os.path.basename(run_meta['design'])}`"
             f"（B={run_meta['n_bits']}，η={run_meta['energy_per_point']:g}）；"
             f"N={run_meta['N']}；角度 {run_meta['angles']}；"
             f"σ={run_meta['sigmas']}；搜索网格 {run_meta['grid_step']}°"
             f"（{run_meta['grid_n']} 个候选）。", "",
             "各行是在 **512×512 图像**上施加的旋转算子，各列是解码端在"
             "**64×64 潜变量**上做反旋转所用的算子。单元格 = "
             "失锁率 / BitAcc（PMR）。参考格是此前所有实验实际使用的组合"
             f"：`{ref_key}`。", ""]

    header = "| 攻击算子 \\ 解码算子 | " + " | ".join(
        f"`{d}`" for d in decode_ops) + " |"
    lines += [header, "|---|" + "---|" * len(decode_ops)]
    for aop in attack_ops:
        cells_txt = []
        for dop in decode_ops:
            a = cells.get(f"{aop}|{dop}")
            if a is None:
                cells_txt.append("--")
                continue
            mark = " **(ref)**" if (aop, dop) == REFERENCE else ""
            cells_txt.append(f"{a['sync_fail_rate']['mean']:.3f} / "
                             f"{a['bit_acc_mean']:.4f} "
                             f"({a['pmr']['mean']:.2f}){mark}")
        lines.append(f"| `{aop}` | " + " | ".join(cells_txt) + " |")

    lines += ["", "## 逐格区间（95% Wilson / bootstrap）", "",
              "| 攻击算子 | 解码算子 | n | 失锁率 [95%CI] | 角误差 中位/p90 | "
              "BitAcc [95%CI] | PMR [95%CI] | oracle BitAcc | Δrel |",
              "|---|---|---:|---|---|---|---|---:|---:|"]
    for aop in attack_ops:
        for dop in decode_ops:
            a = cells.get(f"{aop}|{dop}")
            if a is None:
                continue
            sf, pm, ba = a["sync_fail_rate"], a["pmr"], a["bit_acc"]
            lines.append(
                f"| `{aop}` | `{dop}` | {a['n']} | {sf['mean']:.3f} "
                f"[{sf['ci'][0]:.3f}, {sf['ci'][1]:.3f}] | "
                f"{a['align_err_median']:.2f} / {a['align_err_p90']:.2f} | "
                f"{ba['mean']:.4f} [{ba['ci'][0]:.4f}, {ba['ci'][1]:.4f}] | "
                f"{pm['mean']:.3f} [{pm['ci'][0]:.3f}, {pm['ci'][1]:.3f}] | "
                f"{a['bit_acc_oracle_mean']:.4f} | {a['delta_rel_mean']:.3f} |")

    lines += ["", "## 汇总轴", "",
              "| 轴 | 取值 | n | 失锁率 | BitAcc | PMR |",
              "|---|---|---:|---:|---:|---:|"]
    for aop, a in summary["by_attack_op"].items():
        lines.append(f"| 攻击算子 | `{aop}` | {a['n']} | "
                     f"{a['sync_fail_rate']['mean']:.3f} | "
                     f"{a['bit_acc_mean']:.4f} | {a['pmr']['mean']:.3f} |")
    for dop, a in summary["by_decode_op"].items():
        lines.append(f"| 解码算子 | `{dop}` | {a['n']} | "
                     f"{a['sync_fail_rate']['mean']:.3f} | "
                     f"{a['bit_acc_mean']:.4f} | {a['pmr']['mean']:.3f} |")

    if ref is not None:
        drops = []
        for aop in attack_ops:
            a = cells.get(f"{aop}|{REFERENCE[1]}")
            if a is None or aop == REFERENCE[0]:
                continue
            drops.append((aop, a["bit_acc_mean"] - ref["bit_acc_mean"],
                          a["sync_fail_rate"]["mean"]
                          - ref["sync_fail_rate"]["mean"]))
        lines += ["", "## 相对参考格的落差（同一解码算子）", "",
                  "| 攻击算子 | ΔBitAcc | Δ失锁率 |", "|---|---:|---:|"]
        for aop, d_bit, d_fail in drops:
            lines.append(f"| `{aop}` | {d_bit:+.4f} | {d_fail:+.3f} |")
        worst = min(drops, key=lambda t: t[1]) if drops else None
        if worst:
            lines += ["", f"参考格 BitAcc={ref['bit_acc_mean']:.4f}；"
                      f"最差攻击算子 `{worst[0]}` 落差 {worst[1]:+.4f}。"]

    lines += ["", "## 预注册解读规则（跑之前就定好）", "",
              "1. 若各攻击算子在**同一解码算子**下都落在参考格的 Wilson 区间内"
              "→ 旋转鲁棒性是载波/反演通道的性质，不是某个 `rotate()` 实现的性质；",
              "2. 若只有 `*_c32` 解码格变差 → 落差来自命题 3 的半像素几何"
              "（方法把解码算子*定义*为绕物理中心，这是定义而非缺陷）；",
              "3. 若 cv2 / expand-crop 攻击格在 canonical 解码下明显变差"
              "→ 设计搜索对某一种插值几何过拟合，必须作为 limitation 如实写出。",
              "",
              "角误差按 mod 180° 定义（Hermitian 对径对称）；`oracle BitAcc` 是在"
              "真实角度上解码得到的，用于区分同步失败与载荷失败。", ""]

    path = os.path.join(out_dir, "summary.md")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    safe_print("\n".join(lines[:40]))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
