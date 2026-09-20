"""First-phase experiments: sectorized Tree-Ring multi-bit watermark.

Examples
--------
python run_exp.py --exp 1 --K 16 --N 8 --out_dir runs/exp1
python run_exp.py --exp 2 --N 20 --out_dir runs/exp2
python run_exp.py --exp 3 --K 16 --N 20 --out_dir runs/exp3
python run_exp.py --exp 4 --N 20 --out_dir runs/exp4
python run_exp.py --exp 5 --N 20 --out_dir runs/exp5
python run_exp.py --exp 7 --N 20 --out_dir runs/exp7

The detection protocol follows Tree-Ring/METR: DDIM inversion with the empty
prompt and guidance scale 1, then matched filtering in the FFT domain.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler
from tqdm import tqdm

from configs import EXPERIMENTS, SectorConfig
from pipeline.inverse_stable_diffusion import InversableStableDiffusionPipeline
from pipeline.optim_utils import (
    apply_attack,
    get_dataset,
    measure_similarity,
    set_random_seed,
    transform_img,
)
from swm.carriers import annulus_mask, build_carriers, chip_point_counts, sector_chip_map
from swm.detect import decode
from swm.embed import build_watermark, hermitian_error, inject
from swm.metrics import ber, bit_word_acc, carrier_max_corr, cross_talk_matrix, roc


class Watermarker:
    """Per-channel sectorized watermark encoder/decoder."""

    def __init__(self, cfg: SectorConfig, size: int = 64):
        self.cfg = cfg
        self.size = size
        self.K_ind = cfg.K_ind
        self.ann = annulus_mask(size, cfg.r1, cfg.r2)
        self.chip = sector_chip_map(size, cfg.r1, cfg.r2, cfg.K)
        self.N_ann = float(np.sum(self.ann))
        self.E_f = cfg.total_energy if cfg.total_energy is not None else (
            cfg.energy_per_point * self.N_ann
        )
        self.carriers, self.C, self.N, self.norms, self.channel_ann, self.channel_chip = [], [], [], [], [], []
        for ch_idx, _ in enumerate(cfg.channels):
            # capacity mode: different code per channel; diversity: same code
            c_seed = cfg.w_seed + (ch_idx if cfg.channel_mode == "capacity" else 0)
            car, C, N, norms, ann, chip = build_carriers(
                size, cfg.r1, cfg.r2, cfg.K, cfg.carrier, seed=c_seed, device=cfg.device
            )
            self.carriers.append(car)
            self.C.append(C.to(cfg.device))
            self.N.append(N)
            self.norms.append(norms.to(cfg.device))

    # ------------------------------------------------------------------ encode
    def make_message(self, rng: np.random.RandomState):
        """Return (bits_to_score, full_chip_bits).  Full bits include the
        fixed pilot and the guard bits (not scored)."""
        L = self.cfg.n_bits
        bits = (rng.randint(0, 2, L).astype(np.float32) * 2.0 - 1.0)
        full = np.ones(self.K_ind, dtype=np.float32)
        full[self.cfg.n_pilot : self.cfg.n_pilot + L] = bits
        return bits, full

    def embed(self, z_T: torch.Tensor, msg_per_channel):
        """msg_per_channel: list of full-bit vectors, one per channel."""
        z_w = z_T
        # fixed total energy: split E_f across channels (doc 8.6)
        e_ch = self.E_f / max(len(self.cfg.channels), 1)
        for ch_idx, (ch, bits) in enumerate(zip(self.cfg.channels, msg_per_channel)):
            W = build_watermark(
                self.chip, self.ann, self.C[ch_idx],
                torch.tensor(bits, dtype=torch.float32, device=self.cfg.device),
                alpha=1.0, total_energy=e_ch,
            )
            z_w, _ = inject(z_w, W, ch, self.ann, self.cfg.injection)
        return z_w

    # ------------------------------------------------------------------ decode
    def decode_latents(self, z_hat: torch.Tensor):
        """z_hat: inverted latents [1, C, H, W].  Returns (ell_list, q_list,
        pred_list) where each entry is per channel."""
        Z_hat = torch.fft.fftshift(torch.fft.fft2(z_hat.float()), dim=(-1, -2))
        ells, qs, preds = [], [], []
        for ch_idx, ch in enumerate(self.cfg.channels):
            ell, q, pred = decode(
                Z_hat, self.carriers[ch_idx], self.ann, ch, self.chip, self.K_ind
            )
            ells.append(ell)
            qs.append(q)
            preds.append(pred)
        return ells, qs, preds

    def combine_channels(self, ells, qs, preds):
        """Combine per-channel outputs according to channel_mode."""
        if self.cfg.channel_mode == "diversity":
            ell = np.mean(np.stack(ells), axis=0)
            q = np.mean(np.stack(qs), axis=0)
            pred = (ell > 0).astype(np.int64)
            return ell, q, pred
        # capacity: concatenate chunks
        ell = np.concatenate(ells)
        q = np.concatenate(qs)
        pred = (ell > 0).astype(np.int64)
        return ell, q, pred


def load_pipeline(cfg: SectorConfig, device: str):
    scheduler = DPMSolverMultistepScheduler.from_pretrained(cfg.model_id, subfolder="scheduler")
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        cfg.model_id,
        scheduler=scheduler,
        torch_dtype=cfg.dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_reference(cfg: SectorConfig, device: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.reference_model, pretrained=cfg.reference_model_pretrain, device=device
    )
    # open_clip 2.0.2 exposes the tokenizer as `open_clip.tokenize(texts) -> tensor`
    tokenizer = open_clip.tokenize
    return model, preprocess, tokenizer


def generate(pipe, prompt, latents, cfg: SectorConfig, device):
    return pipe(
        prompt,
        num_images_per_prompt=1,
        guidance_scale=cfg.guidance_scale,
        num_inference_steps=cfg.num_inference_steps,
        height=cfg.image_length,
        width=cfg.image_length,
        latents=latents,
    ).images[0]


def invert(pipe, img, tester_embeddings, cfg: SectorConfig, device):
    img_t = transform_img(img).unsqueeze(0).to(tester_embeddings.dtype).to(device)
    lat = pipe.get_image_latents(img_t, sample=False)
    return pipe.forward_diffusion(
        latents=lat,
        text_embeddings=tester_embeddings,
        guidance_scale=1,
        num_inference_steps=cfg.test_num_inference_steps,
    )


def run_config(cfg: SectorConfig, pipe, ref, dataset, prompt_key, start, end, device,
               save_dir=None):
    """Run one (config, attack) cell and return metrics dict."""
    wm = Watermarker(cfg)
    tester_embeddings = pipe.get_text_embedding("")
    rng = np.random.RandomState(cfg.w_seed)

    if save_dir is not None:
        tag = (f"K{cfg.K}_{cfg.carrier}_ch{'_'.join(map(str, cfg.channels))}"
               f"_{cfg.channel_mode}")
        d_no, d_w = os.path.join(save_dir, tag, "no_w"), os.path.join(save_dir, tag, "w")
        os.makedirs(d_no, exist_ok=True)
        os.makedirs(d_w, exist_ok=True)
    else:
        d_no = d_w = None

    acc = {a: [] for a in cfg._attacks}  # per-attack metric storage
    for a in cfg._attacks:
        acc[a] = dict(
            ber=[], bit_acc=[], word_acc=[], score_w=[], score_nw=[],
            psnr=[], clip_w=[], clip_nw=[], ells=[], true_bits=[],
        )

    for i in tqdm(range(start, end), desc=f"K{cfg.K}-{cfg.carrier}"):
        seed = i + cfg.gen_seed
        prompt = dataset[i][prompt_key]

        set_random_seed(seed)
        z_no = pipe.get_random_latents()
        img_no = generate(pipe, prompt, z_no, cfg, device)

        z_w0 = copy.deepcopy(z_no)
        # per-channel messages
        if cfg.channel_mode == "capacity":
            msgs = [wm.make_message(rng)[1] for _ in cfg.channels]
            true_chunks = [np.where(m[cfg.n_pilot:cfg.n_pilot+cfg.n_bits] > 0, 1, 0) for m in msgs]
            true_bits = np.concatenate(true_chunks)
        else:
            bits, full = wm.make_message(rng)
            msgs = [full for _ in cfg.channels]
            true_bits = np.where(bits > 0, 1, 0)
        z_w = wm.embed(z_w0, msgs)
        img_w = generate(pipe, prompt, z_w, cfg, device)

        # attack-independent quantities: no-watermark inversion, quality, CLIP
        z_hat_no = invert(pipe, img_no, tester_embeddings, cfg, device)
        ell_nw, _, _ = wm.combine_channels(*wm.decode_latents(z_hat_no))
        mse = float((transform_img(img_w) - transform_img(img_no)).pow(2).mean())
        psnr = 10.0 * np.log10(4.0 / max(mse, 1e-12))
        clip_nw = clip_w = None
        if ref is not None:
            sims = measure_similarity([img_no, img_w], prompt, ref[0], ref[1], ref[2], device)
            clip_nw, clip_w = sims[0].item(), sims[1].item()
        if d_no is not None:
            img_no.save(os.path.join(d_no, f"img{i}.png"))
            img_w.save(os.path.join(d_w, f"img{i}.png"))

        for attack in cfg._attacks:
            img_w_aug = apply_attack(img_w, seed, attack)
            z_hat_w = invert(pipe, img_w_aug, tester_embeddings, cfg, device)

            ells, qs, preds = wm.decode_latents(z_hat_w)
            ell, q, pred = wm.combine_channels(ells, qs, preds)

            a = acc[attack]
            a["ber"].append(ber(true_bits, pred))
            ba, wa = bit_word_acc(true_bits, pred)
            a["bit_acc"].append(ba)
            a["word_acc"].append(wa)
            a["score_w"].append(float(np.abs(ell).mean()))
            a["score_nw"].append(float(np.abs(ell_nw).mean()))
            a["ells"].append(ell)
            a["true_bits"].append(true_bits)

            a["psnr"].append(psnr)
            if ref is not None:
                a["clip_nw"].append(clip_nw)
                a["clip_w"].append(clip_w)

    # aggregate
    out = {"config": dict(K=cfg.K, carrier=cfg.carrier, channels=list(cfg.channels),
                          channel_mode=cfg.channel_mode, r1=cfg.r1, r2=cfg.r2,
                          K_ind=cfg.K_ind, n_bits=cfg.n_bits, E_f=wm.E_f,
                          injection=cfg.injection)}
    out["carrier_corr"] = {}
    for ch_idx, ch in enumerate(cfg.channels):
        mx, Gamma = carrier_max_corr(wm.C[ch_idx].cpu().numpy(), wm.N[ch_idx])
        out["carrier_corr"][f"ch{ch}"] = {"max_offdiag_abs": mx, "gamma": Gamma.tolist()}

    for attack in cfg._attacks:
        a = acc[attack]
        auc, tpr1, accv = roc(a["score_w"], a["score_nw"])
        out[attack] = {
            "ber_mean": float(np.mean(a["ber"])),
            "bit_acc_mean": float(np.mean(a["bit_acc"])),
            "word_acc_mean": float(np.mean(a["word_acc"])),
            "auc": auc,
            "tpr@1%fpr": tpr1,
            "acc": accv,
            "score_w_mean": float(np.mean(a["score_w"])),
            "score_nw_mean": float(np.mean(a["score_nw"])),
            "psnr_mean": float(np.mean(a["psnr"])),
            "clip_w_mean": float(np.mean(a["clip_w"])) if a["clip_w"] else None,
            "clip_nw_mean": float(np.mean(a["clip_nw"])) if a["clip_nw"] else None,
            "cross_talk": cross_talk_matrix(a["ells"], a["true_bits"], cfg.K_ind).tolist(),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="1", help="experiment id: 1,2,3,4,5,7")
    ap.add_argument("--K", type=int, default=None, help="override K")
    ap.add_argument("--carrier", default=None, help="override carrier")
    ap.add_argument("--attacks", default=None, help="comma list, override attacks")
    ap.add_argument("--N", type=int, default=10, help="number of images")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-2-1-base")
    ap.add_argument("--dataset", default="Gustavosta/Stable-Diffusion-Prompts")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--save_images", default=None, help="save no_w/w PNGs per config (for FID)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--with_clip", action="store_true")
    ap.add_argument("--reference_model", default=None)
    ap.add_argument("--reference_model_pretrain", default=None)
    args = ap.parse_args()

    spec = EXPERIMENTS[args.exp]
    configs = spec["configs"]
    attacks = spec["attacks"] if args.attacks is None else args.attacks.split(",")
    if args.K is not None:
        configs = [{**c, "K": args.K} for c in configs]
    if args.carrier is not None:
        configs = [{**c, "carrier": args.carrier} for c in configs]

    device = args.device if torch.cuda.is_available() else "cpu"
    if args.device == "cpu":
        device = "cpu"

    cfg_base = SectorConfig(
        model_id=args.model_id,
        dataset=args.dataset,
        device=device,
        with_clip=args.with_clip,
        reference_model=args.reference_model,
        reference_model_pretrain=args.reference_model_pretrain,
    )

    print(f"[exp {args.exp}] {spec['name']} | N={args.N} | attacks={attacks} | device={device}")
    pipe = load_pipeline(cfg_base, device)
    dataset, prompt_key = get_dataset(cfg_base.dataset)
    ref = None
    if cfg_base.with_clip:
        ref = load_reference(cfg_base, device)

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    results = {"experiment": spec["name"], "N": args.N, "attacks": attacks, "cells": []}
    for c in configs:
        cfg = copy.deepcopy(cfg_base)
        for k, v in c.items():
            setattr(cfg, k, v)
        cfg._attacks = attacks
        out = run_config(cfg, pipe, ref, dataset, prompt_key,
                         args.start, args.start + args.N, device,
                         save_dir=args.save_images)
        results["cells"].append(out)
        # compact console summary
        print(
            f"K={cfg.K} carrier={cfg.carrier} ch={cfg.channels} mode={cfg.channel_mode} "
            f"E_f={out['config']['E_f']:.3e}"
        )
        for a in attacks:
            o = out[a]
            print(
                f"  {a:>10s}: BER={o['ber_mean']:.4f} bit_acc={o['bit_acc_mean']:.4f} "
                f"word_acc={o['word_acc_mean']:.4f} AUC={o['auc']:.3f} "
                f"TPR@1%={o['tpr@1%fpr']:.3f} psnr={o['psnr_mean']:.2f}"
            )

    results["elapsed_min"] = (time.time() - t0) / 60.0
    out_file = os.path.join(args.out_dir, f"exp{args.exp}_{spec['name']}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {out_file}  (elapsed {results['elapsed_min']:.1f} min)")


if __name__ == "__main__":
    main()
