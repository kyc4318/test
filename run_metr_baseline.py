"""METR baseline runner on our SD2.1 pipeline (official R=10, S=100).

Usage:
  python run_metr_baseline.py --N 10 --attacks clean,jpeg25,noise0.1,blur4,bright6.0,rot75,crop0.75 \
    --model_id /root/autodl-tmp/models/sd21base \
    --dataset coco:/root/autodl-tmp/data/coco5k/meta_data.json \
    --out_dir runs/metr_baseline
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from configs import SectorConfig
from pipeline.optim_utils import apply_attack, get_dataset, set_random_seed
from run_exp import generate, invert, load_pipeline
from swm.metr_baseline import decode_metr, embed_metr
from swm.metrics import bit_word_acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--radius", type=int, default=10)
    ap.add_argument("--scaler", type=float, default=100.0)
    ap.add_argument("--channel", type=int, default=3)
    ap.add_argument("--attacks", default="clean")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-2-1-base")
    ap.add_argument("--dataset", default="Gustavosta/Stable-Diffusion-Prompts")
    ap.add_argument("--out_dir", default="runs/metr_baseline")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    cfg = SectorConfig(
        model_id=args.model_id,
        dataset=args.dataset,
        num_inference_steps=args.steps,
        test_num_inference_steps=args.steps,
        device=device,
    )
    attacks = args.attacks.split(",")

    pipe = load_pipeline(cfg, device)
    dataset, prompt_key = get_dataset(cfg.dataset)
    tester_embeddings = pipe.get_text_embedding("")

    rows = {a: [] for a in attacks}
    rng = np.random.RandomState(cfg.w_seed)
    for i in tqdm(range(args.start, args.start + args.N), desc="METR"):
        seed = i + cfg.gen_seed
        prompt = dataset[i][prompt_key]
        set_random_seed(seed)
        z_no = pipe.get_random_latents()

        bits = rng.randint(0, 2, args.radius)
        z_w = embed_metr(
            z_no, bits, radius=args.radius, scaler=args.scaler, channel=args.channel
        )
        img_w = generate(pipe, prompt, z_w, cfg, device)

        for attack in attacks:
            aug = apply_attack(img_w, seed, attack)
            z_hat = invert(pipe, aug, tester_embeddings, cfg, device)
            pred = decode_metr(z_hat, radius=args.radius, channel=args.channel)
            ba, wa = bit_word_acc(bits, np.array(pred))
            rows[attack].append(
                {
                    "seed": seed,
                    "bits": bits.tolist(),
                    "pred": pred,
                    "bit_acc": float(ba),
                    "word_acc": float(wa),
                }
            )

    out = {
        "experiment": "METR_baseline",
        "N": args.N,
        "radius": args.radius,
        "scaler": args.scaler,
        "channel": args.channel,
        "steps": args.steps,
        "attacks": attacks,
        "rows": rows,
        "summary": {},
    }
    for a in attacks:
        bas = np.array([r["bit_acc"] for r in rows[a]])
        was = np.array([r["word_acc"] for r in rows[a]])
        out["summary"][a] = {
            "bit_acc_mean": float(bas.mean()),
            "word_acc_mean": float(was.mean()),
            "bit_acc_p10": float(np.percentile(bas, 10)),
            "word_acc1": float(np.mean(was == 1.0)),
        }
        print(
            f"{a:>12s}: bit_acc={out['summary'][a]['bit_acc_mean']:.4f} "
            f"word_acc={out['summary'][a]['word_acc_mean']:.4f} "
            f"word_acc1={out['summary'][a]['word_acc1']:.3f}"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, "metr_baseline.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {out_file}")


if __name__ == "__main__":
    main()
