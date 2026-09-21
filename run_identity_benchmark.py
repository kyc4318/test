"""P0 gate 3: put SectorSync into the key-identification table.

The pilot's identity table only lists SFWMark.  This script runs the *same*
closed-set / open-set protocol for every method that can be registered:

    registry         N registered identities drawn without replacement from a
                     key space of --n_keys
    positives        image watermarked with the identity it is registered under
    negatives-null   unwatermarked image
    negatives-unreg  image watermarked with a key *outside* the registry

and reports

    Id-Acc@1 (closed set), the identification margin
    (best minus runner-up), and a margin threshold calibrated on a disjoint
    calibration split -> TPR@1%FPR for "is this a registered identity?".

Capacity honesty: ours-B can register at most 2^B identities, so with
--n_keys 2048 the 8-bit carrier is only run at 256 keys and the key space is
printed with every table.  `run_paper_compare.SFWMethod` is subclassed rather
than modified, and the canonical ``run_sync_search.py`` / ``swm`` core is
untouched.

    python run_identity_benchmark.py --N 20 --n_keys 2048 \
        --methods ours,sfw_hstr,sfw_hsqr --cases clean,rot45,rot75
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from tqdm import tqdm

from p0_common import (OursCore, ResumeLog, SplitPlan, auc,
                       calibrate_threshold, case_rotation_angle, config_hash,
                       make_config, provenance, rate_above, resolve_device,
                       safe_print, search_grid, wilson, write_json)
from paper_protocol import AttackCtx, apply_case, case_seed

try:  # the script is GPU-only in practice, but keep import-time cost low
    import torch
except Exception:  # pragma: no cover
    torch = None


# --------------------------------------------------------------------------- #
# adapters
# --------------------------------------------------------------------------- #
def codewords(n_bits: int, n_keys: int, seed: int = 0):
    """Enumerate up to ``n_keys`` of the ``2**n_bits`` BPSK payload words.

    Returns (words [n_keys, n_bits] in +-1, ids [n_keys]).  When the key space
    is larger than the registry we take a *seeded* subset, so the id -> word
    map is keyed rather than a fixed prefix.
    """
    space = 1 << n_bits
    rng = np.random.RandomState(seed)
    ids = rng.permutation(space) if n_keys <= space else np.arange(space)
    ids = np.asarray(ids[:min(n_keys, space)], dtype=np.int64)
    words = np.empty((ids.size, n_bits), dtype=np.float32)
    for r, ident in enumerate(ids.tolist()):
        for b in range(n_bits):
            words[r, b] = 1.0 if (ident >> b) & 1 else -1.0
    return words, ids


def make_registry(name: str, n_bits: int, n_keys: int, n_reg: int,
                  key_seed: int):
    """Per-method registry (own key space) drawn without replacement."""
    space = (1 << n_bits) if name == "ours" else int(n_keys)
    rng = np.random.RandomState(key_seed)
    arena = rng.permutation(space)
    registry = arena[:n_reg]
    unregistered = arena[n_reg:n_reg + max(n_reg, 1)]
    return registry.astype(np.int64), unregistered.astype(np.int64), int(space)


class OursIdentity:
    """Closed-set identification on the B-bit payload space."""

    def __init__(self, args, device, registry, unregistered):
        self.name = "ours"
        self.core = OursCore(args.design, device=device,
                             energy_per_point=args.energy_per_point)
        self.n_bits = self.core.n_bits
        self.space = 1 << self.n_bits
        self.words, self.ids = codewords(self.n_bits, self.space,
                                         seed=args.key_seed)
        self.registry = np.asarray(registry, dtype=np.int64)
        self.unregistered = np.asarray(unregistered, dtype=np.int64)
        self.grid = search_grid(args.grid_step)
        self.row_of = {int(v): k for k, v in enumerate(self.ids.tolist())}
        self.reg_rows = [self.row_of[int(k)] for k in self.registry]
        self.W = self.words[self.reg_rows]

    # -- meta ----------------------------------------------------------- #
    def describe(self) -> dict:
        return {"method": self.name, "key_space": int(self.space),
                "n_keys_registered": int(self.registry.size),
                "payload_bits": self.n_bits,
                "decode_ops_per_image": int(self.registry.size),
                "task": "bit payload / closed-set identity"}

    # -- pipeline ------------------------------------------------------- #
    def prepare(self, pipe):
        return

    def encode(self, z_no, key_id):
        self._bits = self.words[self.row_of[int(key_id)]]
        return self.core.embed(z_no, self._bits)

    def decode(self, z_hat, key_id, condition):
        t0 = time.time()
        res = self.core.decode(z_hat, self.grid)
        ell = res["ell"]
        scores = self.W @ ell / float(self.n_bits)
        order = np.argsort(-scores)
        best, second = int(order[0]), int(order[1])
        pred_id = int(self.registry[best])
        signs = np.where(ell > 0, 1.0, -1.0).astype(np.float32)
        hdist = (self.W != signs[None, :]).sum(axis=1)
        bits = (signs > 0).astype(int)
        hard_pred = int(self.registry[int(np.argmin(hdist))])
        return {
            "est_angle": res["est_angle"],
            "detection_score": res["detection_score"],
            "pred_id_soft": pred_id,
            "pred_id_hard": hard_pred,
            "margin": float(scores[best] - scores[second]),
            "decoded_bits": bits.astype(int).tolist(),
            "decode_sec": time.time() - t0,
        }


class SFWIdentity:
    """Thin adapter around run_paper_compare.SFWMethod (subclassed, not edited)."""

    def __init__(self, args, device, registry, unregistered, kind):
        from run_paper_compare import SFWMethod

        self.name = f"sfw_{kind.lower()}"
        self.kind = kind
        self.registry = np.asarray(registry, dtype=np.int64)
        self.unregistered = np.asarray(unregistered, dtype=np.int64)
        self.device = device

        outer = self

        class _RegistrySFW(SFWMethod):
            def build_patterns(self, pipe):
                if self.patterns is not None:
                    return
                sfw = self.sfw
                seeds = [self.w_seed + int(j) for j in outer.registry]
                if self.kind == "HSTR":
                    self.patterns = [sfw.make_Fourier_treering_pattern(
                        pipe, sfw.shape, s, hs=True, center=True, heter=True)
                        for s in seeds]
                else:
                    self.patterns = [sfw.make_hsqr_pattern(idx=s) for s in seeds]

            def distances(self, z_hat):
                """Copy of SFWMethod.identify() that returns the full vector."""
                sfw = self.sfw
                from run_paper_compare import windowed_spectrum

                Z = windowed_spectrum(z_hat, sfw.center_slice)
                r0, r1 = sfw.center_slice[2], sfw.center_slice[3]
                if self.kind == "HSTR":
                    ch = sfw.RINGID_WATERMARK_CHANNEL
                    zc = Z[0][ch][:, r0, r1]
                    if self._mask_center is None:
                        self._mask_center = self.dist_mask[..., r0, r1].clone()
                    m = self._mask_center
                    diff = torch.abs(self._candidate_stack()[:, ch]
                                     - zc.unsqueeze(0))
                    l1 = [(diff[:, k] * m[k].unsqueeze(0)).sum(dim=(-1, -2))
                          / m[k].sum() for k in range(len(ch))]
                    d = torch.stack(l1, dim=-1).min(dim=-1).values
                else:
                    ch = sfw.HSQR_WATERMARK_CHANNEL[0]
                    n, half = 42, 21
                    blk = Z[0, ch, 11:11 + n, 33:33 + half]
                    got = torch.cat([blk.real.flatten(), blk.imag.flatten()])
                    st = self._candidate_stack().float()
                    gt = torch.cat([st[:, :, :half].reshape(len(st), -1),
                                    st[:, :, half:n].reshape(len(st), -1)],
                                   dim=1)
                    gt = torch.where(gt.bool(),
                                     torch.tensor(45.0, device=gt.device),
                                     torch.tensor(-45.0, device=gt.device))
                    d = (gt - got.unsqueeze(0)).abs().mean(dim=1)
                return d.detach().cpu().numpy().astype(np.float64)

        self.impl = _RegistrySFW(args, device, kind)
        self.w_seed = self.impl.w_seed

    def describe(self) -> dict:
        return {"method": self.name,
                "key_space": 2048,
                "n_keys_registered": int(self.registry.size),
                "payload_bits": 0,
                "decode_ops_per_image": int(self.registry.size),
                "task": "key identification"}

    def prepare(self, pipe):
        self.impl.build_patterns(pipe)

    def encode(self, z_no, key_id):
        pos = int(np.where(self.registry == int(key_id))[0][0])
        out = self.impl.latents(z_no, None, {"index": pos, "seed": 0,
                                             "theta": 0.0})
        return out["wm"]

    def decode(self, z_hat, key_id, condition):
        t0 = time.time()
        d = self.impl.distances(z_hat)
        order = np.argsort(d)
        best, second = int(order[0]), int(order[1])
        pred_id = int(self.registry[best])
        return {
            "pred_id_soft": pred_id,
            "pred_id_hard": pred_id,
            "margin": float(d[second] - d[best]),
            "detection_score": float(-d[best]),
            "decode_sec": time.time() - t0,
        }


def build_adapter(name, args, device, registry, unregistered):
    if name == "ours":
        return OursIdentity(args, device, registry, unregistered)
    if name in ("sfw_hstr", "sfw_hsqr"):
        return SFWIdentity(args, device, registry, unregistered,
                           "HSTR" if name.endswith("hstr") else "HSQR")
    raise ValueError(f"unknown method {name!r}")


# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="ours,sfw_hstr,sfw_hsqr")
    ap.add_argument("--design", default="results/design_searched_realgeom.npz")
    ap.add_argument("--n_keys", type=int, default=2048)
    ap.add_argument("--key_seed", type=int, default=7)
    ap.add_argument("--cases", default="clean,rot45,rot75")
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--calib_offset", type=int, default=2000)
    ap.add_argument("--test_offset", type=int, default=0)
    ap.add_argument("--grid_step", type=float, default=2.0)
    ap.add_argument("--energy_per_point", type=float, default=1e4)
    ap.add_argument("--sfw_candidates", type=int, default=2048)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--with_unregistered", type=int, default=1)
    ap.add_argument("--model_id", default="/root/autodl-tmp/models/sd21base")
    ap.add_argument("--dataset",
                    default="coco:/root/autodl-tmp/data/coco5k/meta_data.json")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out_dir", default="runs/identity")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


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
    cases = [c for c in args.cases.split(",") if c]

    plan = SplitPlan(start=args.start, calib_n=args.N, test_n=args.N,
                     calib_offset=args.calib_offset,
                     test_offset=args.test_offset)
    indices = np.concatenate([plan.indices("calib"), plan.indices("test")])
    split_of = {}
    for i in plan.indices("calib").tolist():
        split_of[int(i)] = "calib"
    for i in plan.indices("test").tolist():
        split_of[int(i)] = "test"

    os.makedirs(args.out_dir, exist_ok=True)
    ctx = AttackCtx(device=device)
    run_meta = {
        "script": "run_identity_benchmark.py",
        "design": args.design,
        "design_hash": config_hash({"design": args.design}),
        "n_keys": args.n_keys,
        "key_seed": args.key_seed,
        "cases": cases,
        "split": plan.as_dict(),
        "fpr": args.fpr,
        "grid_step": args.grid_step,
        "energy_per_point": args.energy_per_point,
        "registries": {},
        "provenance": provenance(),
    }

    rows_path = os.path.join(args.out_dir, "rows.jsonl")
    t0 = time.time()
    with ResumeLog(rows_path, keys=("uid",), fingerprint=run_meta) as log:
        for name in [m for m in args.methods.split(",") if m]:
            n_bits = payload_bits_of(name, args, device)
            registry, unregistered, space = make_registry(
                name, n_bits, args.n_keys, args.N, args.key_seed)
            adapter = build_adapter(name, args, device, registry, unregistered)
            adapter.prepare(pipe)
            meta = adapter.describe()
            meta.update({"key_space": int(space),
                         "registry": registry.tolist(),
                         "unregistered": unregistered.tolist()})
            write_json(os.path.join(args.out_dir, f"{name}_meta.json"), meta)
            run_meta["registries"][name] = {
                "key_space": int(space),
                "registry": registry.tolist(),
                "unregistered": unregistered.tolist(),
            }
            for i in tqdm(indices.tolist(), desc=f"identity-{name}"):
                seed = i + cfg.gen_seed
                prompt = dataset[i][prompt_key]
                set_random_seed(seed)
                z_no = pipe.get_random_latents()
                key_id = int(registry[i % registry.size])

                z_w = adapter.encode(z_no, key_id)
                img_w = generate(pipe, prompt, z_w, cfg, device)
                img_null = generate(pipe, prompt, z_no, cfg, device)
                img_un = None
                if args.with_unregistered:
                    z_un = adapter.encode(z_no, int(unregistered[i % unregistered.size]))
                    img_un = generate(pipe, prompt, z_un, cfg, device)

                for case in cases:
                    theta = float(np.random.RandomState(seed + 777)
                                  .uniform(0.0, 180.0))
                    variants = [("pos", img_w, key_id),
                                ("null", img_null, key_id)]
                    if img_un is not None:
                        variants.append(
                            ("unregistered", img_un,
                             int(unregistered[i % unregistered.size])))
                    for cond, img, kid in variants:
                        uid = f"{name}|{i}|{case}|{cond}"
                        if log.has({"uid": uid}):
                            continue
                        rng_a = np.random.RandomState(case_seed(seed, case))
                        img_a = apply_case(img, case, theta, rng_a, ctx)
                        z_hat = invert(pipe, img_a, emb, cfg, device)
                        res = adapter.decode(z_hat, kid, cond)
                        row = {
                            "uid": uid,
                            "method": name,
                            "split": split_of[int(i)],
                            "image_id": int(i),
                            "prompt_id": int(i),
                            "seed": int(seed),
                            "case": case,
                            "condition": cond,
                            "key_id": int(kid),
                            "registered": float(cond != "unregistered"),
                            "payload_bits": adapter.describe()["payload_bits"],
                            "key_space": int(space),
                            "n_keys_registered":
                                adapter.describe()["n_keys_registered"],
                            # ``theta`` is the per-image random draw used by the
                            # bare ``rot`` token; for literal tokens such as
                            # ``rot45`` the angle actually applied is the
                            # literal one, so store that (plus the raw draw).
                            "true_angle": float(case_rotation_angle(case,
                                                                    theta)),
                            "theta": float(theta),
                            "id_correct_soft": float(
                                res["pred_id_soft"] == int(kid)),
                            "id_correct_hard": float(
                                res["pred_id_hard"] == int(kid)),
                            "margin": float(res["margin"]),
                            "detection_score": float(res["detection_score"]),
                            "decode_sec": float(res["decode_sec"]),
                        }
                        if "est_angle" in res:
                            row["est_angle"] = res["est_angle"]
                        log.append(row)
                del img_w, img_null, z_no
                if _cuda():
                    import torch

                    torch.cuda.empty_cache()

    from p0_common import load_rows

    write_json(os.path.join(args.out_dir, "run_config.json"), run_meta)
    rows = load_rows(rows_path)
    summary = summarise(rows, args)
    write_json(os.path.join(args.out_dir, "summary.json"),
               {"config": run_meta, "summary": summary,
                "elapsed_sec": time.time() - t0})
    report(args.out_dir, run_meta, summary)
    print(f"\nelapsed {(time.time() - t0) / 60:.1f} min -> {args.out_dir}")


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def payload_bits_of(name: str, args, device) -> int:
    if name != "ours":
        return 0
    return OursCore(args.design, device=device,
                    energy_per_point=args.energy_per_point).n_bits


def summarise(rows, args):
    out = {}
    methods = sorted({r["method"] for r in rows})
    for m in methods:
        mrows = [r for r in rows if r["method"] == m]
        entry = {
            "n_keys_registered": int(mrows[0]["n_keys_registered"]) if mrows else 0,
            "key_space": int(mrows[0]["key_space"]) if mrows else 0,
            "payload_bits": mrows[0]["payload_bits"] if mrows else None,
            "cases": {},
        }
        for case in sorted({r["case"] for r in mrows}):
            sub = [r for r in mrows if r["case"] == case]
            pos = [r for r in sub if r["condition"] == "pos"]
            null = [r for r in sub if r["condition"] == "null"]
            unreg = [r for r in sub if r["condition"] == "unregistered"]
            pos_calib = [r for r in pos if r["split"] == "calib"]
            pos_test = [r for r in pos if r["split"] == "test"]
            null_calib = [r for r in null if r["split"] == "calib"]
            null_test = [r for r in null if r["split"] == "test"]
            unreg_test = [r for r in unreg if r["split"] == "test"]

            k = int(sum(1 for r in pos_test if r["id_correct_soft"]))
            acc, lo, hi = wilson(k, len(pos_test))
            kh = int(sum(1 for r in pos_test if r["id_correct_hard"]))
            acch, loh, hih = wilson(kh, len(pos_test))
            tau = calibrate_threshold([r["margin"] for r in null_calib],
                                      args.fpr)
            entry["cases"][case] = {
                "n_pos_calib": len(pos_calib),
                "n_pos_test": len(pos_test),
                "id_acc_soft": {"mean": acc, "ci": [lo, hi], "k": k},
                "id_acc_hard": {"mean": acch, "ci": [loh, hih], "k": kh},
                "margin_mean_pos": _mean([r["margin"] for r in pos_test]),
                "margin_mean_null": _mean([r["margin"] for r in null_test]),
                "auc_pos_vs_null": auc([r["margin"] for r in pos_test],
                                       [r["margin"] for r in null_test]),
                "tpr_at_calibrated_fpr": rate_above(
                    [r["margin"] for r in pos_test], tau),
                "fpr_realised_null": rate_above(
                    [r["margin"] for r in null_test], tau),
                "decode_sec_mean": _mean([r["decode_sec"] for r in sub]),
            }
            if unreg_test:
                entry["cases"][case].update({
                    "auc_pos_vs_unregistered": auc(
                        [r["margin"] for r in pos_test],
                        [r["margin"] for r in unreg_test]),
                    "tpr_at_calibrated_fpr_unreg": rate_above(
                        [r["margin"] for r in pos_test], tau),
                    "fpr_realised_unregistered": rate_above(
                        [r["margin"] for r in unreg_test], tau),
                })
        out[m] = entry
    return out


def _mean(v):
    return float(np.mean(v)) if len(v) else None


def report(out_dir, run_meta, summary) -> None:
    lines = ["# P0 身份识别（key 任务）", "",
             f"注册表大小 {run_meta['n_keys']}，每格 N="
             f"{run_meta['split']['test_n']}（校准 {run_meta['split']['calib_n']}），"
             f"FPR 目标 {run_meta['fpr']:g}，阈值取自互不相交的校准划分。",
             ""]
    lines += ["| 方法 | key 空间 | 每图解码量 | 攻击 | Id-Acc@1 [95%CI] "
              "| margin 均值 | AUC(pos vs null) | TPR@1%FPR |",
              "|---|---:|---:|---|---|---:|---:|---:|"]
    for m, e in summary.items():
        for case, c in e["cases"].items():
            a = c["id_acc_soft"]
            lines.append(
                f"| {m} | {e['key_space']} | {e['n_keys_registered']} | {case} | "
                f"{a['mean']:.3f} [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}] | "
                f"{_f(c['margin_mean_pos'])} | {_f(c['auc_pos_vs_null'])} | "
                f"{_f(c['tpr_at_calibrated_fpr'])} |")
    lines += ["", "口径说明：", "",
              "- `ours-B` 的可注册身份上限是 $2^B$，因此 8 bit 载波在 2048 key "
              "空间上只能注册 256 个身份；表中 key 空间与注册量分列，不做跨任务"
              "的“追平/胜出”断言。",
              "- `margin` = 最优候选与次优候选的分数差（ours 为软匹配分数差，"
              "SFWMark 为距离差，方向统一成越大越像注册身份）。",
              "- `unregistered` 负样本是**用注册表之外的 key 加水印**的图像，"
              "它比无水印负样本更难拒绝；两个 FPR 都在细节表里给出。", ""]
    path = os.path.join(out_dir, "summary.md")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    safe_print("\n".join(lines))
    print(f"-> {path}")


def _f(v, nd=4):
    return "--" if v is None else f"{float(v):.{nd}f}"


if __name__ == "__main__":
    main()
