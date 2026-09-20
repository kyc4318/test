"""P0 gate 1: does "max over angles" break the false-positive rate?

The decoder reports ``max_gamma S(gamma)``.  Searching many angles inflates the
*null* maximum even when no watermark is present, so an AUC/TPR computed with
the threshold fitted on the same attacked images is not evidence.  This script
runs the full control set:

    positives    watermarked image, same carrier, same attack
    null         unwatermarked image, same latent / prompt / attack seed
    wrongkey*    watermarked image decoded with a *different* codebook

and then reports

    (a) TPR at a threshold calibrated on an independent calibration split,
    (b) the realised FPR on the held-out test split,
    (c) the same numbers for the "no angle search" score S(0), to show how much
        of the FPR is attributable to the max over angles,
    (d) an angle-leak test: for rotation cases, is the estimated angle of a
        *null* image correlated with the applied rotation?  If yes, the
        synchroniser is reading the interpolation/padding, not the carrier,
    (e) an optional variant with the zero-filled wedges cropped away.

The canonical ``run_sync_search.py`` / ``swm`` core is not modified.

    python run_paper_controls.py --N 50 --cases clean,rot45,rot75,rot+noise0.05 \
        --wrongkey_designs results/design_B8_s1.npz,results/design_B8_s2.npz
"""

from __future__ import annotations

import argparse
import os
import re
import time

import numpy as np
from tqdm import tqdm

from paper_protocol import AttackCtx, apply_token, case_seed
from p0_common import (OursCore, ResumeLog, SplitPlan, add_gaussian_noise,
                       align_error, auc, bit_acc, bits_from_rng,
                       calibrate_threshold, case_rotation_angle, config_hash,
                       is_synced, make_config, perfect_match, provenance,
                       rate_above, resolve_device, rotate_image, safe_print,
                       inscribed_crop_after_rotation, search_grid, signed_error,
                       write_json)


def apply_case_control(img, case: str, theta: float, rng, ctx, resample: str,
                       fill: int, crop_black: bool):
    """``apply_case`` with the rotation operator swapped for a control variant."""
    if (not crop_black) and resample == "bilinear" and fill == 0:
        from paper_protocol import apply_case

        return apply_case(img, case, theta, rng, ctx)
    out = img
    for token in case.split("+"):
        if token == "rot" or re.fullmatch(r"rot-?\d+(?:\.\d+)?", token):
            deg = float(theta) if token == "rot" else float(token[3:])
            out = rotate_image(out, deg, resample=resample, fill=fill)
            if crop_black:
                out = inscribed_crop_after_rotation(out, deg)
        else:
            out = apply_token(out, token, theta, rng, ctx)
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="results/design_searched_realgeom.npz")
    ap.add_argument("--wrongkey_designs", default="",
                    help="comma list of additional .npz codebooks used as "
                         "wrong-key negatives (the key-robustness control)")
    ap.add_argument("--cases", default="clean,rot45,rot75,rot+noise0.05")
    ap.add_argument("--N", type=int, default=50,
                    help="images per split (calibration and test are disjoint)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--calib_offset", type=int, default=0)
    ap.add_argument("--test_offset", type=int, default=1000)
    ap.add_argument("--grid_step", type=float, default=2.0)
    ap.add_argument("--energy_per_point", type=float, default=1e4)
    ap.add_argument("--fprs", default="0.01,0.05")
    ap.add_argument("--sync_tol", type=float, default=2.0)
    ap.add_argument("--rotate_resample", default="bilinear",
                    choices=["bilinear", "bicubic", "nearest"])
    ap.add_argument("--rotate_fill", type=int, default=0)
    ap.add_argument("--crop_black", action="store_true",
                    help="crop the zero wedges after rotating (leak control)")
    ap.add_argument("--extra_noise", type=float, default=0.0,
                    help="apply Gaussian noise after the rotation token as an "
                         "additional realisation (sigma in 0..1 image scale)")
    ap.add_argument("--store_full_scores", action="store_true", default=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--model_id", default="/root/autodl-tmp/models/sd21base")
    ap.add_argument("--dataset",
                    default="coco:/root/autodl-tmp/data/coco5k/meta_data.json")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out_dir", default="runs/p0_controls")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    from run_exp import generate, invert
    from pipeline.optim_utils import get_dataset, set_random_seed

    device = resolve_device(args.device)
    cfg = make_config(args, device)
    pipe = build_pipe_safe(cfg, device)
    dataset, prompt_key = get_dataset(cfg.dataset)
    emb = pipe.get_text_embedding("")

    cases = [c for c in args.cases.split(",") if c]
    fprs = [float(x) for x in args.fprs.split(",") if x]
    plan = SplitPlan(start=args.start, calib_n=args.N, test_n=args.N,
                     calib_offset=args.calib_offset,
                     test_offset=args.test_offset)
    angles = search_grid(args.grid_step)

    core = OursCore(args.design, device=device,
                    energy_per_point=args.energy_per_point)
    wrong = {}
    for path in [p for p in args.wrongkey_designs.split(",") if p]:
        wrong[os.path.basename(path)] = OursCore(
            path, device=device, energy_per_point=args.energy_per_point)
    if wrong and any(w.n_bits != core.n_bits for w in wrong.values()):
        raise SystemExit("wrong-key designs must have the same payload length")

    tag = args.tag or f"step{args.grid_step:g}_N{args.N}"
    if args.crop_black:
        tag += "_cropblack"
    if args.rotate_resample != "bilinear":
        tag += f"_{args.rotate_resample}"
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    ctx = AttackCtx(device=device)
    rng_w = np.random.RandomState(cfg.w_seed)

    run_meta = {
        "script": "run_paper_controls.py",
        "design": args.design,
        "design_hash": config_hash({"design": open(args.design, "rb").read()
                                    if os.path.exists(args.design) else args.design}),
        "wrongkey_designs": list(wrong),
        "n_bits": core.n_bits,
        "grid_step": args.grid_step,
        "grid_n": int(len(angles)),
        "energy_per_point": args.energy_per_point,
        "cases": cases,
        "fprs": fprs,
        "sync_tol": args.sync_tol,
        "rotate_resample": args.rotate_resample,
        "rotate_fill": args.rotate_fill,
        "crop_black": bool(args.crop_black),
        "extra_noise": args.extra_noise,
        "split": plan.as_dict(),
        "provenance": provenance(),
    }
    write_json(os.path.join(out_dir, "run_config.json"), run_meta)

    indices = np.concatenate([plan.indices("calib"), plan.indices("test")])
    split_of = {}
    for i in plan.indices("calib").tolist():
        split_of[int(i)] = "calib"
    for i in plan.indices("test").tolist():
        split_of[int(i)] = "test"

    rows_path = os.path.join(out_dir, "rows.jsonl")
    done_path = os.path.join(out_dir, "done.json")
    t0 = time.time()

    with ResumeLog(rows_path, keys=("uid",), fingerprint=run_meta) as log:
        for i in tqdm(indices.tolist(), desc="p0-controls"):
            seed = i + cfg.gen_seed
            prompt = dataset[i][prompt_key]
            set_random_seed(seed)
            z_no = pipe.get_random_latents()

            bits = bits_from_rng(rng_w, core.n_bits)
            true_bits = np.where(bits > 0, 1, 0)
            z_w = core.embed(z_no, bits)

            img_null = generate(pipe, prompt, z_no, cfg, device)
            img_w = generate(pipe, prompt, z_w, cfg, device)

            for case in cases:
                # same draw rule as run_paper_compare: a bare "rot" token uses a
                # per-image theta, a literal "rot75" uses its own angle, and any
                # composed case that contains "rot" is handled uniformly here
                theta = float(np.random.RandomState(seed + 777).uniform(0.0, 180.0))
                true_angle = case_rotation_angle(case, theta)

                rng_a = np.random.RandomState(case_seed(seed, case))
                img_w_a = apply_case_control(img_w, case, theta, rng_a, ctx,
                                             args.rotate_resample,
                                             args.rotate_fill, args.crop_black)
                rng_b = np.random.RandomState(case_seed(seed, case))
                img_n_a = apply_case_control(img_null, case, theta, rng_b, ctx,
                                             args.rotate_resample,
                                             args.rotate_fill, args.crop_black)
                if args.extra_noise > 0:
                    img_w_a = add_gaussian_noise(img_w_a, args.extra_noise,
                                                 case_seed(seed, case) + 17)
                    img_n_a = add_gaussian_noise(img_n_a, args.extra_noise,
                                                 case_seed(seed, case) + 17)

                t_inv = time.time()
                z_hat_w = invert(pipe, img_w_a, emb, cfg, device)
                z_hat_n = invert(pipe, img_n_a, emb, cfg, device)
                if torch_cuda():
                    import torch

                    torch.cuda.empty_cache()
                inv_sec = time.time() - t_inv

                t_dec = time.time()
                S_w = core.landscape(z_hat_w, angles)
                S_n = core.landscape(z_hat_n, angles)
                dec_sec = time.time() - t_dec
                S_wk = {}
                for name, wc in wrong.items():
                    S_wk[name] = wc.landscape(z_hat_w, angles)

                rec = {
                    "uid": f"{i}|{case}|wm",
                    "split": split_of[int(i)],
                    "image_id": int(i),
                    "prompt_id": int(i),
                    "seed": int(seed),
                    "case": case,
                    "condition": "wm",
                    "key_id": 0,
                    "payload_bits": core.n_bits,
                    "true_bits": true_bits.tolist(),
                    "true_angle": float(true_angle),
                    "theta": float(theta),
                    "est_angle": float(angles[int(np.argmax(S_w))]),
                    "est_error_deg": align_error(float(angles[int(np.argmax(S_w))]),
                                                 true_angle),
                    "signed_error_deg": signed_error(
                        float(angles[int(np.argmax(S_w))]), true_angle),
                    "synced": is_synced(float(angles[int(np.argmax(S_w))]),
                                        true_angle, args.sync_tol),
                    "detection_score": float(S_w.max()),
                    "score_at_zero": float(S_w[0]),
                    "scores": S_w.tolist() if args.store_full_scores else None,
                    "inv_sec": float(inv_sec),
                    "decode_sec": float(dec_sec),
                }
                ell = core.ell_mean(z_hat_w, rec["est_angle"])
                rec["decoded_bits"] = (ell > 0).astype(int).tolist()
                rec["bit_acc"] = bit_acc(true_bits, rec["decoded_bits"])
                rec["perfect"] = perfect_match(true_bits, rec["decoded_bits"])
                if not log.has(rec):
                    log.append(rec)

                for cond, S, z_hat in (("null", S_n, z_hat_n),):
                    est = float(angles[int(np.argmax(S))])
                    r = {
                        "uid": f"{i}|{case}|{cond}",
                        "split": split_of[int(i)],
                        "image_id": int(i),
                        "prompt_id": int(i),
                        "seed": int(seed),
                        "case": case,
                        "condition": cond,
                        "key_id": -1,
                        "payload_bits": core.n_bits,
                        "true_angle": float(true_angle),
                        "theta": float(theta),
                        "est_angle": est,
                        "est_error_deg": align_error(est, true_angle),
                        "signed_error_deg": signed_error(est, true_angle),
                        "synced": is_synced(est, true_angle, args.sync_tol),
                        "detection_score": float(S.max()),
                        "score_at_zero": float(S[0]),
                        "scores": S.tolist() if args.store_full_scores else None,
                        "inv_sec": float(inv_sec),
                        "decode_sec": float(dec_sec),
                    }
                    if not log.has(r):
                        log.append(r)

                for name, S in S_wk.items():
                    r = {
                        "uid": f"{i}|{case}|wrongkey:{name}",
                        "split": split_of[int(i)],
                        "image_id": int(i),
                        "prompt_id": int(i),
                        "seed": int(seed),
                        "case": case,
                        "condition": f"wrongkey:{name}",
                        "key_id": -2,
                        "payload_bits": core.n_bits,
                        "true_angle": float(true_angle),
                        "theta": float(theta),
                        "est_angle": float(angles[int(np.argmax(S))]),
                        "est_error_deg": align_error(
                            float(angles[int(np.argmax(S))]), true_angle),
                        "detection_score": float(S.max()),
                        "score_at_zero": float(S[0]),
                        "scores": S.tolist() if args.store_full_scores else None,
                        "inv_sec": 0.0,
                        "decode_sec": float(dec_sec),
                    }
                    if not log.has(r):
                        log.append(r)

            del img_null, img_w, z_no
            if torch_cuda():
                import torch

                torch.cuda.empty_cache()

    rows = load_rows_safe(rows_path)
    summary = summarise(rows, cases, fprs, args)
    write_json(os.path.join(out_dir, "summary.json"),
               {"config": run_meta, "summary": summary,
                "elapsed_sec": time.time() - t0})
    write_json(done_path, {"n_rows": len(rows), "elapsed_sec": time.time() - t0})
    report(out_dir, run_meta, summary)
    print(f"\nelapsed {(time.time() - t0) / 60:.1f} min -> {out_dir}")


# --------------------------------------------------------------------------- #
# helpers kept local so the script works even without torch at import time
# --------------------------------------------------------------------------- #
def torch_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def build_pipe_safe(cfg, device):
    from p0_common import build_pipe

    return build_pipe(cfg, device)


def load_rows_safe(path):
    from p0_common import load_rows

    return load_rows(path)


def summarise(rows, cases, fprs, args):
    out = {}
    for case in cases:
        sub = [r for r in rows if r["case"] == case]
        if not sub:
            continue
        entry = {"n_wm_calib": 0, "n_wm_test": 0}
        by = {}
        for r in sub:
            by.setdefault((r["condition"], r["split"]), []).append(r)

        def scores(cond, split, field="detection_score"):
            return [r[field] for r in by.get((cond, split), [])]

        def angles_of(cond, split):
            return [r["est_angle"] for r in by.get((cond, split), [])]

        def errs_of(cond, split):
            return [r["est_error_deg"] for r in by.get((cond, split), [])]

        wm_calib = scores("wm", "calib")
        wm_test = scores("wm", "test")
        null_calib = scores("null", "calib")
        null_test = scores("null", "test")
        entry["n_wm_calib"] = len(wm_calib)
        entry["n_wm_test"] = len(wm_test)
        entry["n_null_calib"] = len(null_calib)
        entry["n_null_test"] = len(null_test)
        entry["null_score_mean_calib"] = float(np.mean(null_calib)) if null_calib else None
        entry["null_score_mean_test"] = float(np.mean(null_test)) if null_test else None
        entry["wm_score_mean_test"] = float(np.mean(wm_test)) if wm_test else None
        entry["auc_test"] = auc(wm_test, null_test)
        entry["auc_calib"] = auc(wm_calib, null_calib)
        entry["search_gain_null"] = (
            float(np.mean(null_calib)) / max(float(np.mean(
                [r["score_at_zero"] for r in by.get(("null", "calib"), [])])), 1e-12)
            if null_calib else None)

        entry["calibrated"] = {}
        for f in fprs:
            tau = calibrate_threshold(null_calib, f)
            tau0 = calibrate_threshold(
                [r["score_at_zero"] for r in by.get(("null", "calib"), [])], f)
            entry["calibrated"][f"{f:g}"] = {
                "tau_max_over_angles": tau,
                "tpr_test": rate_above(wm_test, tau),
                "fpr_test_realised": rate_above(null_test, tau),
                "tpr_calib_in_sample": rate_above(wm_calib, tau),
                "tau_no_search": tau0,
                "tpr_test_no_search": rate_above(
                    [r["score_at_zero"] for r in by.get(("wm", "test"), [])], tau0),
                "fpr_test_no_search": rate_above(
                    [r["score_at_zero"] for r in by.get(("null", "test"), [])], tau0),
            }

        # angle leak: does a null image's estimate track the applied rotation?
        leak = {}
        for split in ("calib", "test"):
            e = errs_of("null", split)
            leak[split] = {
                "n": len(e),
                "mean_align_err": float(np.mean(e)) if e else None,
                "sync_rate_null": (float(np.mean(
                    [r["synced"] for r in by.get(("null", split), [])]))
                    if e else None),
            }
        entry["null_angle_leak"] = leak
        e_wm = errs_of("wm", "test")
        entry["wm_sync"] = {
            "n": len(e_wm),
            "mean_align_err": float(np.mean(e_wm)) if e_wm else None,
            "sync_fail_rate": (1.0 - float(np.mean(
                [r["synced"] for r in by.get(("wm", "test"), [])]))
                if e_wm else None),
        }
        entry["wm_bit_acc_mean"] = float(np.mean(
            [r["bit_acc"] for r in by.get(("wm", "test"), [])])) if wm_test else None
        entry["wm_pmr"] = float(np.mean(
            [r["perfect"] for r in by.get(("wm", "test"), [])])) if wm_test else None

        for cond in sorted({r["condition"] for r in sub}):
            if not cond.startswith("wrongkey"):
                continue
            s_calib = scores(cond, "calib")
            s_test = scores(cond, "test")
            entry.setdefault("wrongkey", {})[cond] = {
                "n_test": len(s_test),
                "score_mean_test": float(np.mean(s_test)) if s_test else None,
                "auc_vs_wm_test": auc(wm_test, s_test),
                **{f"fpr_test_at_tau_{f:g}": rate_above(
                    s_test, calibrate_threshold(null_calib, f))
                   for f in fprs},
            }
        out[case] = entry
    return out


def report(out_dir, run_meta, summary) -> None:
    lines = ["# P0 控制实验：角度搜索的假阳性与角度泄漏", "",
             f"协议版本 `{run_meta['provenance'].get('protocol_version')}`；"
             f"载波 `{os.path.basename(run_meta['design'])}`（B={run_meta['n_bits']}）；"
             f"搜索网格 {run_meta['grid_step']}°（{run_meta['grid_n']} 个候选）；"
             f"校准/测试各 {run_meta['split']['calib_n']} 张，索引互不相交。", ""]
    if run_meta["crop_black"]:
        lines.append("> 本表使用 `--crop_black` 算子：旋转 -> 取内接正方形 -> "
                     "**resize 回原尺寸**。注意它同时改变尺度（45° 时约 1.41x），"
                     "因此**不能**当作「去黑边」对照来解读；"
                     "尺度不变量下的去填充对照见 `run_unseen_operators.py` 的 "
                     "`cv2_*_reflect`（反射填充，无黑边、几何与插值可对齐）。")
        lines.append("")
    lines += ["| 攻击 | null 分数(均值) | 搜索增益 | AUC | "
              "**搜索检测器** TPR@1%FPR | 实测 FPR | "
              "**无搜索检测器** TPR@1%FPR（独立标定） | 其实测 FPR | "
              "null 角度泄漏率 | ours 失锁率 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for case, e in summary.items():
        c = e["calibrated"].get("0.01", {})
        leak = e["null_angle_leak"].get("test", {})
        lines.append(
            f"| {case} | {fmt(e['null_score_mean_test'])} | "
            f"{fmt(e['search_gain_null'], 3)} | {fmt(e['auc_test'], 3)} | "
            f"{fmt(c.get('tpr_test'), 3)} | {fmt(c.get('fpr_test_realised'), 3)} | "
            f"{fmt(c.get('tpr_test_no_search'), 3)} | "
            f"{fmt(c.get('fpr_test_no_search'), 3)} | "
            f"{fmt(leak.get('sync_rate_null'), 3)} | "
            f"{fmt(e['wm_sync'].get('sync_fail_rate'), 3)} |")
    lines += ["", "说明：", "",
              "- **搜索增益** = 校准集上 null 的 `max_γ S(γ)` 均值 / `S(0)` 均值。"
              "接近 1 说明最大搜索没有抬高噪声底；显著大于 1 说明必须在解码端"
              "做多重比较校正。",
              "- **两个检测器各自标定阈值**：`tau_max_over_angles` 用校准集的"
              "`max_γ S(γ)`，`tau_no_search` 用校准集的 `S(0)`。两列因此是"
              "「相同名义 FPR 下的两个检测器」，而不是共用一个阈值；"
              "两者的**实测** FPR 都列出，因为它们可能不同（校准集只有 50 张）。",
              "- 测试集从未参与阈值选择，因此两列都可作为独立检验。",
              "- **null 角度泄漏率** 只在旋转类攻击上有意义："
              "无水印图同样被旋转，若其估计角仍与真实旋转角吻合，"
              "说明角度信息来自插值/黑边，而非载波。这是**不依赖任何填充算子"
              "几何假设**的泄漏检验（它只要求攻击算子本身一致）。",
              ""]
    path = os.path.join(out_dir, "summary.md")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    safe_print("\n".join(lines))
    print(f"-> {path}")


def fmt(v, nd=4):
    if v is None:
        return "--"
    return f"{float(v):.{nd}f}"


if __name__ == "__main__":
    main()
