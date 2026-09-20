"""Unified comparison: SectorSync vs open-source baselines, one protocol.

    ours                 SectorSync (dual-layer shift-identifiable carrier +
                         decoder-native rotation search, 1-deg grid)
    ours_nosync          the same carrier with the synchroniser switched off
    gs256 / gs8          Gaussian Shading (CVPR'24), official code
    sfw_hstr / sfw_hsqr  SFWMark (ICCV'25), official code

RingID / MaXsive / Tree-Ring run through their own official scripts; their
numbers are merged in make_paper_tables.py.

    python run_paper_compare.py --method ours --N 50 \
        --cases clean,signal,rotation,composed --roc 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

import numpy as np
import torch
from torchvision.transforms import functional as TF
from tqdm import tqdm

from configs import SectorConfig
from paper_protocol import (AttackCtx, apply_case, bit_metrics, case_seed,
                            detection_metrics, expand_cases)
from pipeline.optim_utils import get_dataset, set_random_seed
from run_exp import Watermarker, generate, invert, load_pipeline


def rotate_latent(z: torch.Tensor, deg: float) -> torch.Tensor:
    """Anti-rotate a recovered latent; TF.rotate on 64x64 spins about index 31.5,
    i.e. the physical image centre -- the geometry the attack applied."""
    return TF.rotate(z, float(deg), fill=0)


def windowed_spectrum(z_hat: torch.Tensor, center_slice) -> torch.Tensor:
    """SFWMark convention: transform the 44x44 centre window and place it back."""
    zb = z_hat if z_hat.dim() == 4 else z_hat.unsqueeze(0)
    Z = torch.zeros_like(zb, dtype=torch.complex64)
    Z[center_slice] = torch.fft.fftshift(
        torch.fft.fft2(zb[center_slice].float(), dim=(-1, -2)), dim=(-1, -2))
    return Z


# --------------------------------------------------------------------------- #
# ours
# --------------------------------------------------------------------------- #
class OursMethod:
    name = "ours"
    task = "payload"
    needs_null = True

    def __init__(self, args, device):
        from swm.dual_layer import DualLayerWatermarker

        self.grid = np.arange(0.0, 360.0, args.grid_step)
        self.dl = DualLayerWatermarker(args.design, size=64, channel=3,
                                       device=device,
                                       energy_per_point=args.energy_per_point)
        # payload length comes from the design itself (B = rows of C), so the
        # runner works unchanged for the B=4/8/12/16 designs.
        self.n_bits = int(self.dl.layers[0]["spec"].C.shape[0])
        self.device = device

    def latents(self, z_no, bits, msg):
        return {"wm": self.dl.embed(z_no, bits), "null": z_no}

    def _scores(self, z_hat) -> np.ndarray:
        out = []
        for a in self.grid:
            _, sc = self.dl.decode_layers(rotate_latent(z_hat, -float(a)))
            p = 1.0
            for s in sc:
                p = p * float(s)
            out.append(p)
        return np.asarray(out, dtype=np.float64)

    def _bits_at(self, z_hat, ang: float) -> np.ndarray:
        ells = [e.detach().cpu().numpy() for e in
                self.dl.decode_layers(rotate_latent(z_hat, -float(ang)))[0]]
        return (np.mean(np.stack(ells), axis=0) > 0).astype(int)

    def decode(self, z_hat, i=None) -> dict:
        S = self._scores(z_hat)
        est = float(self.grid[int(np.argmax(S))])
        return {"bits": self._bits_at(z_hat, est),
                "bits_direct": self._bits_at(z_hat, 0.0),
                "score": float(S.max()), "score_direct": float(S[0]),
                "floor": float(np.median(S)), "est_angle": est}


# --------------------------------------------------------------------------- #
# Gaussian Shading (CVPR'24)
# --------------------------------------------------------------------------- #
class GSMixin:
    def _init_gs(self, args, ch, hw):
        sys.path.insert(0, "/root/work/repos/Gaussian-Shading")
        self.args_gs = (ch, hw, args.fpr, args.user_number)
        self.n_bits = 4 * 64 * 64 // (ch * hw * hw)

    def gs_embed(self, z_no, seed):
        from watermark import Gaussian_Shading_chacha

        set_random_seed(seed)
        gs = Gaussian_Shading_chacha(*self.args_gs)
        return gs, gs.create_watermark_and_return_w()

    def _store(self, i, gs):
        self._gs_store = getattr(self, "_gs_store", {})
        self._gs_store[i] = gs


class GSMethod(GSMixin):
    task = "payload"
    needs_null = True

    def __init__(self, args, device, ch, hw, name):
        self._init_gs(args, ch, hw)
        self.name = name
        self.device = device

    def latents(self, z_no, bits, msg):
        gs, z_w = self.gs_embed(z_no, msg["seed"])
        self._store(msg["index"], gs)
        return {"wm": z_w, "null": z_no}

    def decode(self, z_hat, i) -> dict:
        acc = float(self._gs_store[i].eval_watermark(z_hat))
        return {"score": 2.0 * (acc - 0.5), "bit_acc_native": acc,
                "native_perfect": float(acc >= 1.0)}


# --------------------------------------------------------------------------- #
# SFWMark (ICCV'25)
# --------------------------------------------------------------------------- #
class SFWMethod:
    task = "identify"
    needs_null = True

    def __init__(self, args, device, kind: str):
        sys.path.insert(0, "/root/work/repos/SFWMark/src")
        import utils as sfw

        # Compatibility shim (no change to the upstream repo): the official
        # SFWMark code runs the model in float32, so its 44x44 centre-window
        # FFTs are legal.  Our pipeline is float16, and cuFFT rejects
        # non-power-of-two half-precision transforms -- force float32 here.
        sfw.fft = lambda t: torch.fft.fftshift(torch.fft.fft2(t.float()),
                                               dim=(-1, -2))
        self.sfw = sfw
        self.kind = kind
        self.name = f"sfw_{kind.lower()}"
        self.device = device
        self.w_seed = sfw.w_seed
        self.n_candidates = args.sfw_candidates
        self.n_bits = 42 * 21 * 2 if kind == "HSQR" else 0
        if kind == "HSTR":
            # injection mask (generate.py) vs detection mask (detect.py)
            self.inject_mask = sfw.tree_masks
            self.inject_mask[:, sfw.HETER_WATERMARK_CHANNEL] = \
                sfw.single_channel_heter_watermark_mask
            self.dist_mask = sfw.watermark_region_mask_hstr
        self.patterns = None
        self._stack = None
        self._mask_center = None

    def build_patterns(self, pipe):
        if self.patterns is not None:
            return
        sfw = self.sfw
        seeds = [self.w_seed + j for j in range(self.n_candidates)]
        if self.kind == "HSTR":
            self.patterns = [sfw.make_Fourier_treering_pattern(
                pipe, sfw.shape, s, hs=True, center=True, heter=True)
                for s in seeds]
        else:
            self.patterns = [sfw.make_hsqr_pattern(idx=s) for s in seeds]

    def latents(self, z_no, bits, msg):
        sfw = self.sfw
        i = msg["index"]
        # The official SFW injectors transform the 44x44 centre window, whose
        # size is not a power of two -> cuFFT needs float32 there; the rest of
        # the pipeline stays in the native dtype.
        out_dtype = z_no.dtype
        z_b = (z_no if z_no.dim() == 4 else z_no.unsqueeze(0)).float()
        if self.kind == "HSTR":
            pat = self.patterns[i].to(self.device)
            z_w, _ = sfw.inject_wm(z_b, pat, self.inject_mask, center=True,
                                   cut_real=False, device=self.device)
        else:
            pat = self.patterns[i].to(self.device).unsqueeze(0)
            z_w = sfw.inject_hsqr(z_b, pat, center=True, device=self.device)
        return {"wm": z_w.to(out_dtype), "null": z_no}

    def _distance(self, pat, z_hat):
        sfw = self.sfw
        Z = windowed_spectrum(z_hat, sfw.center_slice)
        if self.kind == "HSTR":
            return float(sfw.get_distance(pat, Z, mask=self.dist_mask,
                                          channel=sfw.RINGID_WATERMARK_CHANNEL,
                                          p=1, center=True, mode="complex",
                                          channel_min=True))
        return float(sfw.get_distance_hsqr(pat, Z,
                                           channel=sfw.HSQR_WATERMARK_CHANNEL,
                                           p=1, center=True))

    def _candidate_stack(self):
        sfw = self.sfw
        if self._stack is None:
            st = torch.stack([p[0].to(self.device) for p in self.patterns])
            if self.kind == "HSTR":
                st = st[:, :, sfw.center_slice[2], sfw.center_slice[3]]
            self._stack = st
        return self._stack

    def identify(self, z_hat):
        """Vectorised exhaustive identification over the key space."""
        sfw = self.sfw
        Z = windowed_spectrum(z_hat, sfw.center_slice)
        r0, r1 = sfw.center_slice[2], sfw.center_slice[3]
        if self.kind == "HSTR":
            ch = sfw.RINGID_WATERMARK_CHANNEL
            zc = Z[0][ch][:, r0, r1]                                # (2,44,44)
            if self._mask_center is None:
                self._mask_center = self.dist_mask[..., r0, r1].clone()
            m = self._mask_center
            diff = torch.abs(self._candidate_stack()[:, ch] - zc.unsqueeze(0))
            l1 = [(diff[:, k] * m[k].unsqueeze(0)).sum(dim=(-1, -2)) / m[k].sum()
                  for k in range(len(ch))]
            d = torch.stack(l1, dim=-1).min(dim=-1).values
        else:
            ch = sfw.HSQR_WATERMARK_CHANNEL[0]
            n, half = 42, 21
            blk = Z[0, ch, 11:11 + n, 33:33 + half]
            got = torch.cat([blk.real.flatten(), blk.imag.flatten()])
            st = self._candidate_stack().float()
            gt = torch.cat([st[:, :, :half].reshape(len(st), -1),
                            st[:, :, half:n].reshape(len(st), -1)], dim=1)
            gt = torch.where(gt.bool(), torch.tensor(45.0, device=gt.device),
                             torch.tensor(-45.0, device=gt.device))
            d = (gt - got.unsqueeze(0)).abs().mean(dim=1)
        return int(torch.argmin(d).item()), float(torch.min(d).item())

    def _module_bit_acc(self, qr_gt, z_hat) -> float:
        sfw = self.sfw
        Z = windowed_spectrum(z_hat, sfw.center_slice)
        n, half = qr_gt.shape[-1], 21
        blk = Z[0, sfw.HSQR_WATERMARK_CHANNEL[0], 11:11 + n, 33:33 + half]
        signs = torch.cat([blk.real.flatten(), blk.imag.flatten()]) > 0
        gt = torch.cat([qr_gt[0, :, :half].flatten().bool(),
                        qr_gt[0, :, half:n].flatten().bool()]).to(signs.device)
        return float((signs == gt).float().mean())

    def decode(self, z_hat, i) -> dict:
        gt = self.patterns[i]
        d_gt = self._distance(gt, z_hat)
        best_j, best_d = self.identify(z_hat)
        out = {"score": -d_gt, "id_correct": float(best_j == i),
               "id_dist_margin": float(d_gt - best_d)}
        if self.kind == "HSQR":
            out["bit_acc"] = self._module_bit_acc(gt, z_hat)
        return out


def build_method(name, args, device):
    if name == "ours":
        return OursMethod(args, device)
    if name == "gs256":
        return GSMethod(args, device, 1, 8, "gs256")
    if name == "gs8":
        return GSMethod(args, device, 2, 32, "gs8")
    if name == "sfw_hstr":
        return SFWMethod(args, device, "HSTR")
    if name == "sfw_hsqr":
        return SFWMethod(args, device, "HSQR")
    raise ValueError(f"unknown method {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--cases", default="clean,signal,rotation,composed")
    ap.add_argument("--roc", type=int, default=1)
    ap.add_argument("--grid_step", type=float, default=2.0)
    ap.add_argument("--energy_per_point", type=float, default=1e4)
    ap.add_argument("--design", default="results/design_searched_realgeom.npz")
    ap.add_argument("--gs_ch", type=int, default=1)
    ap.add_argument("--gs_hw", type=int, default=8)
    ap.add_argument("--sfw_candidates", type=int, default=2048)
    ap.add_argument("--fpr", type=float, default=1e-6)
    ap.add_argument("--user_number", type=int, default=10 ** 6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--model_id", default="/root/autodl-tmp/models/sd21base")
    ap.add_argument("--dataset",
                    default="coco:/root/autodl-tmp/data/coco5k/meta_data.json")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    cases = expand_cases(args.cases)
    out_dir = args.out_dir or f"runs/paper_{args.method}"
    os.makedirs(out_dir, exist_ok=True)

    cfg = SectorConfig(model_id=args.model_id, dataset=args.dataset,
                       num_inference_steps=args.steps,
                       test_num_inference_steps=args.steps, device=device)
    pipe = load_pipeline(cfg, device)
    dataset, prompt_key = get_dataset(cfg.dataset)
    emb = pipe.get_text_embedding("")
    wm = Watermarker(cfg)
    rng_w = np.random.RandomState(cfg.w_seed)
    method = build_method(args.method, args, device)
    if isinstance(method, SFWMethod):
        method.build_patterns(pipe)
    prompt_holder = [""]

    def diffusion_regen(img, noise_step: int):
        """DiffWMAttacker-style regeneration: VAE encode, schedule noise, resample."""
        from pipeline.optim_utils import transform_img

        with torch.no_grad():
            x = transform_img(img).unsqueeze(0).to(torch.float16).to(device)
            lat = pipe.get_image_latents(x, sample=False)
            alpha = pipe.scheduler.alphas_cumprod[noise_step].to(device)
            z_noisy = (alpha.sqrt() * lat
                       + (1 - alpha).sqrt() * torch.randn_like(lat)).to(torch.float16)
            out = pipe(prompt_holder[0], num_images_per_prompt=1, guidance_scale=7.5,
                       num_inference_steps=args.steps, latents=z_noisy)
            return out.images[0]

    ctx = AttackCtx(device=device, regen=diffusion_regen)
    print(f"[{method.name}] task={method.task} cases={cases} N={args.N} "
          f"roc={bool(args.roc)} grid={args.grid_step}deg")

    rows: List[dict] = []
    t0 = time.time()
    for i in tqdm(range(args.start, args.start + args.N), desc=method.name):
        seed = i + cfg.gen_seed
        prompt = dataset[i][prompt_key]
        prompt_holder[0] = prompt
        set_random_seed(seed)
        z_no = pipe.get_random_latents()
        # message length follows the method's payload (8 for the canonical
        # design, 16 for design_B16_s0, ...).  The draw reproduces
        # Watermarker.make_message exactly for B = 8, so the existing runs stay
        # comparable; every image gets an independent random message.
        n_bits = getattr(method, "n_bits", 8) or 8
        bits = (rng_w.randint(0, 2, n_bits).astype(np.float32) * 2.0 - 1.0)
        true_bits = np.where(bits > 0, 1, 0)
        theta = float(np.random.RandomState(seed + 777).uniform(0.0, 180.0))
        msg = {"index": i, "seed": seed, "theta": theta}
        lat = method.latents(z_no, bits, msg)
        if not args.roc:
            lat = {k: v for k, v in lat.items() if k == "wm"}
        img = {k: generate(pipe, prompt, v, cfg, device) for k, v in lat.items()}
        for case in cases:
            rng = np.random.RandomState(case_seed(seed, case))
            rec = {"index": i, "case": case, "theta": theta, "seed": seed,
                   "method": method.name, "true_bits": true_bits.tolist()}
            z_w = invert(pipe, apply_case(img["wm"], case, theta, rng, ctx),
                         emb, cfg, device)
            res = method.decode(z_w, i)
            rec["wm_score"] = float(res["score"])
            for k, v in res.items():
                rec[f"wm_{k}"] = v.tolist() if isinstance(v, np.ndarray) else v
            if "bits" in res:
                bm = bit_metrics(true_bits, res["bits"])
                rec["wm_bit_acc"], rec["wm_perfect"] = bm["bit_acc"], bm["perfect"]
                rec["wm_direct_bit_acc"] = bit_metrics(true_bits,
                                                       res["bits_direct"])["bit_acc"]
            if args.roc and "null" in img:
                z_n = invert(pipe, apply_case(img["null"], case, theta,
                                              np.random.RandomState(
                                                  case_seed(seed, case)), ctx),
                             emb, cfg, device)
                rn = method.decode(z_n, i)
                rec["null_score"] = float(rn["score"])
                for k in ("bit_acc", "bit_acc_native", "id_correct"):
                    if k in rn:
                        rec[f"null_{k}"] = float(rn[k])
            rows.append(rec)
        del img, z_no
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fields = ["wm_bit_acc", "wm_perfect", "wm_score", "null_score",
              "wm_direct_bit_acc", "wm_id_correct", "null_id_correct",
              "wm_bit_acc_native", "wm_native_perfect", "null_bit_acc_native",
              "wm_est_angle",
              "wm_id_dist_margin"]
    summary = {}
    for case in cases:
        sub = [r for r in rows if r["case"] == case]
        agg = {"n": len(sub)}
        for k in fields:
            vals = [r[k] for r in sub if k in r and r[k] is not None]
            if vals:
                agg[k] = float(np.mean(vals))
        if args.roc:
            agg["detection"] = detection_metrics(
                [r["wm_score"] for r in sub], [r["null_score"] for r in sub])
        summary[case] = agg

    path = os.path.join(out_dir, f"{method.name}.json")
    with open(path, "w") as f:
        json.dump({"method": method.name, "task": method.task,
                   "config": vars(args), "cases": cases, "rows": rows,
                   "summary": summary, "elapsed_sec": time.time() - t0}, f,
                  indent=2)
    print(f"\n{'case':>18} {'n':>4} {'bitacc':>8} {'perfect':>8} {'AUC':>6} "
          f"{'TPR@1%':>7} {'idacc':>6}")
    for case in cases:
        s = summary[case]
        det = s.get("detection", {})
        print(f"{case:>18} {s['n']:>4} {s.get('wm_bit_acc', float('nan')):>8.4f} "
              f"{s.get('wm_perfect', float('nan')):>8.3f} "
              f"{det.get('auc', float('nan')):>6.3f} "
              f"{det.get('tpr_at_fpr_0.01', float('nan')):>7.3f} "
              f"{s.get('wm_id_correct', float('nan')):>6.3f}")
    print(f"\nelapsed {(time.time()-t0)/60:.1f} min -> {path}")


if __name__ == "__main__":
    main()
