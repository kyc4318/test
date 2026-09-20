# SectorSync

**Payload-native blind rotation synchronization for multi-bit frequency-domain
watermarks in diffusion latents.**

SectorSync embeds an 8-bit payload into an annular region of the SD2.1 latent
spectrum with a *shift-identifiable* sector carrier (two incommensurate angular
grids + Hermitian complex phase + product fusion). Detection performs a **single**
DDIM inversion and then searches the rotation angle on the recovered latent,
using the decoder's own profile likelihood as the synchronization score. No
pilot, template, anchor coefficient, or second inversion is used.

> 中文摘要：这是一个把多比特载荷写进扩散模型潜空间环带的频域水印方案。
> 核心主张是「解码器自身的匹配滤波响应就足以做盲旋转同步」，无需 pilot /
> 模板 / 训练配准网络，且只需一次 DDIM 反演。本仓库包含方法核心、载波设计
> 搜索、统一对比流水线，以及一整套用于回应审稿质疑的**证据闸门**实验。

---

## Status

This is **research code accompanying a manuscript in preparation**. All reported
numbers are **pilot scale** (`N=50` images per cell unless stated otherwise) on a
single `SD2.1-base` configuration. See [Known caveats](#known-caveats) before
quoting any number.

The detailed evidence reports (in Chinese) live in [`reports/`](reports/):
start with [`reports/P0_SESSION_SUMMARY.md`](reports/P0_SESSION_SUMMARY.md).

## The research question

Multi-bit angular encoding destroys the rotation invariance that single-bit ring
watermarks (Tree-Ring, RingID) rely on: after a rotation the sector coordinates
mismatch and a fixed-coordinate decoder reads noise. The obvious fixes are an
extra pilot/anchor, a known message, or a trained registration network.

We ask whether the **payload carrier itself** can carry both the message and the
alignment information, so that synchronization costs no extra support and no
extra inversion.

## Method in brief

| Stage | What happens |
|---|---|
| Carrier | Annulus `10 <= rho <= 20` of the `64x64` latent, channel 3, split radially into two layers (`K=32` inner, `K=36` outer); each layer uses its own angular grid and a Hermitian complex phase mask `Psi_j(w) = C[j, chi(w)] * exp(i phi[chi(w)] * eps(w))`, so the IFFT stays real while the 180-degree structural alias is broken |
| Embedding | `Z[c][A] <- W` with `W = sum_j b_j Psi_j`, energy normalized to `E_f = eta * N_A` (default `eta = 1e4`), replace-style injection |
| Detection | **One** DDIM inversion (empty prompt, guidance 1) -> anti-rotate the recovered latent by a candidate angle `gamma` -> matched filter `ell_j(gamma) = Re<Z, Psi_j> / ||Psi_j||^2` |
| Synchronization | `gamma_hat = argmax_gamma prod_l mean_j \|ell_j^{(l)}(gamma)\|` (product = AND fusion across the two layers) |
| Payload | `b_hat_j = sign(ell_j(gamma_hat))` |

Two angular aliases are handled by two *different* metrics: the **common integer
lattice alias** (`rho_mod180`, suppressed by the incommensurate grids + product
fusion) and the **180-degree structural ambiguity** (`A_pi`, broken only by the
complex phase mask). See `measure_alias_pi.py` and the operators report.

## Headline results (pilot, `N=50`, paired per sample)

Payload decoding, cell = `BitAcc / PMR` (PMR = fraction of images whose **whole**
8-bit message is recovered):

| Attack | Gaussian Shading (256-bit) | GS-8 (capacity-matched) | **SectorSync** |
|---|---|---|---|
| clean | 1.0000 / 1.000 | 1.0000 / 1.000 | **1.0000 / 1.000** |
| JPEG q25 | 0.9964 / 0.720 | 1.0000 / 1.000 | **1.0000 / 1.000** |
| noise sigma=0.1 | 0.9849 / 0.460 | 1.0000 / 1.000 | **1.0000 / 1.000** |
| blur r=5 | 0.9567 / 0.020 | 1.0000 / 1.000 | **1.0000 / 1.000** |
| brightness x6 | 0.9008 / 0.120 | 0.9875 / 0.980 | **1.0000 / 1.000** |
| **rotation 45 deg** | 0.4966 / 0.000 | 0.5150 / 0.020 | **0.9950 / 0.960** |
| **rotation 75 deg** | 0.4963 / 0.000 | 0.4875 / 0.000 | **0.9975 / 0.980** |
| **rotation + noise 0.05** | 0.5045 / 0.000 | 0.4825 / 0.000 | **0.9425 / 0.880** |

The cleanest causal evidence is the **synchronization switch**: same carrier,
same attack, only the decoder switches the angle search on or off. At a matched
nominal 1% FPR the rotation TPR goes `0.020 / 0.260 / 0.100` (search **off**) to
`1.000` (search **on**).

Other verified claims (all with Wilson / bootstrap intervals in `reports/`):

* **1230 pure-rotation samples** (integer, off-grid, random and negative angles):
  5 sync failures (**0.41%**).
* **Unseen rotation operators** (PIL bicubic, OpenCV linear+reflect, OpenCV
  cubic+constant, PIL expand+crop) keep `BitAcc` 0.994-0.996 under the canonical
  decoder. `pil_expand_crop` — which removes the zero-filled wedges *without*
  changing scale — matches the reference cell exactly, so the synchronizer is not
  reading padding.
* **Equal-quality comparison**: at the same PSNR/SSIM/LPIPS/CLIP, the 8-bit design
  beats the 16-bit design by **+0.30 to +0.58 PMR** under rotation.

## Repository layout

```
swm/                         method core
  carriers.py                annulus/sector/Hermitian pairing, code matrices
  embed.py                   fixed-energy injection
  detect.py                  matched filter, soft reliability
  metrics.py                 BER / word accuracy / ROC
  dual_layer.py              *the paper's carrier*: two incommensurate layers,
                             complex phase, product fusion (loads a design npz)
  metr_baseline.py           METR re-implementation (comparison baseline)
pipeline/                    DDIM generation + inversion (vendored, see NOTICE)
configs.py                   SectorConfig
run_exp.py                   pipeline loading, generate(), invert()

search_design_realgeom.py    canonical carrier search (produces the design we ship)
search_design_bits.py        same pipeline parameterized by payload length B
search_shift_identifiable_designs.py   the original/general search
verify_shift_identifiable_design.py    CPU verification of a design npz
measure_alias_pi.py          two-operator A_pi ablation (rho_mod180 vs A_pi)

paper_protocol.py            attack vocabulary + metrics + split/threshold tools
run_paper_compare.py         unified runner: ours / GS-256 / GS-8 / SFWMark
make_paper_tables.py         assembles T1/T2/T3 tables
add_confidence_intervals.py  Wilson / bootstrap intervals
run_quality_gt.py            CLIP + paired PSNR/SSIM/LPIPS (+ FID inputs)
run_fid.py                   FID wrapper with sample-size warnings/provenance

p0_common.py                 shared layer for the evidence gates
run_paper_controls.py        angle-search FPR / wrong-key / angle-leak controls
run_continuous_rotation.py   continuous, negative and off-grid rotations
run_unseen_operators.py      rotation-operator generalisation (2-axis matrix)
run_inversion_error_spectrum.py   where the inversion error goes (radial/angular,
                             in/out of the carrier subspace)
run_identity_benchmark.py    key-registry identification (ours included)
run_capacity_quality_pareto.py    capacity x energy quality/robustness Pareto
run_p0_queue.sh              generic stage queue
run_p0_session.sh            the queue actually used for the reported runs

results/design_searched_realgeom.npz   **canonical carrier** used by every result
results/design_search_realgeom.json    its search metadata
reports/                     the evidence chain + main tables (Chinese)
```

`runs/` (raw per-sample JSONL) is intentionally **not** shipped: it is large and
fully regenerable by the scripts above.

## Setup

```bash
conda create -n sector python=3.10 -y && conda activate sector
pip install -r requirements.txt
```

Assets (paths are the defaults baked into the scripts; override with the CLI
flags `--model_id`, `--dataset`, `--clip_model`):

```bash
# SD 2.1-base (diffusers layout)
modelscope download --model AI-ModelScope/stable-diffusion-2-1-base \
  --local_dir /root/autodl-tmp/models/sd21base

# CLIP ViT-g-14 (open_clip checkpoint), optional, for the quality table
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download laion/CLIP-ViT-g-14-laion2B-s12B-b42K \
  --local-dir /root/autodl-tmp/models/ViT-g-14

# COCO prompts (+ 512x512 references if you want FID)
#   /root/autodl-tmp/data/coco5k/meta_data.json  -> {"annotations":[{"caption": ...}, ...]}
#   /root/autodl-tmp/data/coco5k/images512/
```

`meta_data.json` is a single JSON object with an `annotations` list of caption
records; the runner uses `dataset[i]["caption"]`. **Note:** the instance used for
the reported runs had 1000 captions (and 5000 image files), which is why the
queues pass explicit index offsets — see [Known caveats](#known-caveats).

## Quickstart

**1. CPU only — verify the shipped carrier** (~seconds, no diffusion):

```bash
python verify_shift_identifiable_design.py --design results/design_searched_realgeom.npz
python measure_alias_pi.py --design results/design_searched_realgeom.npz
```

**2. GPU smoke test — the synchronization switch, 2 images** (~1 min):

```bash
python run_paper_controls.py --design results/design_searched_realgeom.npz \
  --N 2 --cases clean,rot45 --grid_step 2 \
  --calib_offset 0 --test_offset 500 --out_dir runs/smoke_controls
```

**3. The main comparison** (needs the external baselines, see below):

```bash
bash run_paper_queue.sh core        # ours / gs256 / gs8 / sfw_hstr / sfw_hsqr
python make_paper_tables.py
python add_confidence_intervals.py
```

**4. The evidence-gate suite** (this is what makes the rotation claim auditable):

```bash
# smoke first, then the real thing
N_CONTROLS=5 bash run_p0_session.sh controls
bash run_p0_session.sh all          # controls -> rotation -> operators -> spectrum -> pareto
```

Every P0 stage is resumable (append-only `rows.jsonl` with uid de-duplication).

## The canonical carrier, and the other designs

`results/design_searched_realgeom.npz` is the carrier behind **every** number in
this repo. It is produced by:

```bash
python search_design_realgeom.py            # CPU, a few minutes
```

The capacity / multi-key variants are parameterized by payload length and seed
and are **not** shipped, because they are exactly reproducible on CPU:

```bash
for B in 4 8 12 16; do
  python search_design_bits.py --bits $B --seed 0 --out_name design_B${B}_s0 --out_dir results
done
for S in 1 2 3; do      # independent keys, used as wrong-key negatives
  python search_design_bits.py --bits 8 --seed $S --out_name design_B8_s${S} --out_dir results
done
```

## External baselines are **not** vendored

`run_paper_compare.py` imports the official implementations from two paths that
are hard-coded for the machine that produced the results:

```python
sys.path.insert(0, "/root/work/repos/Gaussian-Shading")   # run_paper_compare.py:104
sys.path.insert(0, "/root/work/repos/SFWMark/src")        # run_paper_compare.py:148
```

To reproduce the comparison tables, clone the upstream repositories:

| Baseline | Paper | Repo |
|---|---|---|
| Gaussian Shading | CVPR'24 | `YuanyouYin/Gaussian-Shading` |
| SFWMark (HSTR / HSQR) | ICCV'25 | SFWMark official code |
| RingID | ECCV'24 | `showlab/RingID` |
| MaXsive | MM'25 | MaXsive official code |

and either place them at those two paths or edit the two `sys.path.insert` lines
(they are ordinary, non-canonical configuration lines). METR is re-implemented
in `swm/metr_baseline.py`; the METR runner is `run_metr_baseline.py`.

## Evidence gates: what each script answers

| Script | Reviewer question it answers | Verdict (pilot) |
|---|---|---|
| `run_paper_controls.py` | "Is your AUC just a max over angles?" / "does a wrong key pass?" / "is the angle leaking from the padding?" | presence: **passes** (TPR@1%FPR 1.000 on 7/8 cells); wrong-key: **fails** by design — see caveats |
| `run_continuous_rotation.py` | "Do intermediate / negative / off-grid angles break it?" | **passes** (0.41% failure over 1230 pure-rotation samples) |
| `run_unseen_operators.py` | "Did you optimize for one particular `rotate()` implementation?" | **passes** (4 unseen attack operators, 3 unseen decoder kernels) |
| `run_inversion_error_spectrum.py` | "Why does rotation+noise still fail?" | rotation barely attenuates the annulus (-3.71 dB, against a -2.12 dB clean-inversion floor) but grows the residual 6x, and **62-70% of that residual lands inside the carrier subspace** |
| `run_identity_benchmark.py` | "Can it enter a key-identification table?" | closed-set `Id-Acc@1 = 1.000`; but rejecting **unregistered** keys at a presence threshold fails (FPR 0.30-0.70) |
| `run_capacity_quality_pareto.py` | "Is the robustness just a bigger energy knob?" | energy is a real knob **with a steep quality price**; at equal quality the 8-bit design wins by 0.30-0.58 PMR |

## Known caveats

Read these before quoting anything:

1. **Pilot scale.** `N=50`. The empirical FPR resolution is 2%; `TPR@1e-3` /
   `TPR@1e-6` cannot be estimated from this batch. All ratio metrics in
   `reports/` carry 95% Wilson / bootstrap intervals — use those, not the point
   estimates alone.
2. **`mean|ell|` is a *presence* statistic, not a key-attribution statistic.**
   Images watermarked with a wrong (or unregistered) key still score high,
   because the statistic measures annulus energy projected onto carriers that
   span the same annulus. Attribute keys with the **payload** (bit match), and
   calibrate any rejection threshold on wrong-key / unregistered negatives
   rather than on unwatermarked images. Both columns are reported by the
   scripts.
3. **The canonical latent rotation is nearest-neighbour.** `torchvision`'s
   `TF.rotate` defaults to `InterpolationMode.NEAREST`; `run_paper_compare.rotate_latent`
   and the design searches call it with defaults. Bilinear/bicubic decoders
   perform equally well (see the operators report) — but if you describe the
   method as "bilinear resampling", that does not match the code.
4. **The rotation centre matters.** The decoder must rotate about the *physical*
   centre (pixel index `31.5` for a `64x64` latent), which is what `TF.rotate`
   does by default. Rotating about the DFT origin (pixel `32`) costs ~0.24 PMR
   (`run_unseen_operators.py`, decoder op `tv_bilinear_c32`). Note that in
   torchvision's API the physical centre corresponds to `center=None`, while
   passing `center=[32, 32]` is *also* the physical centre — the API shifts by
   `c - size/2` internally.
5. **Index offsets.** `p0_common.SplitPlan` defaults to `test_offset=1000`. On a
   prompt set with 1000 captions that is out of range, which is why
   `run_p0_session.sh` passes `--calib_offset 0 --test_offset 500`. Keep the
   calibration and test index ranges disjoint: that is the whole point of the
   presence gate.
6. **`ResumeLog` deduplicates on `uid`.** Running a small smoke test and a full
   run into the *same* `--out_dir` silently keeps the smoke rows (with the smoke
   configuration) and skips those uids later. Always use separate output
   directories per configuration.
7. **`--crop_black` is not a padding-leak control.** `inscribed_crop_after_rotation`
   rotates, crops to the inscribed square and **resizes back to 512**, i.e. it is
   rotation *plus* a ~1.41x zoom at 45 degrees. Use `pil_expand_crop` (in
   `run_unseen_operators.py`) for a scale-preserving, padding-free control.
8. **Scale / crop are out of scope.** The synchronizer is angular; scaling is a
   radial transform. That is a stated boundary, not an implementation bug.
9. **Quality numbers are paired, not literature FID.** `reports/paper_pareto.md`
   reports paired PSNR/SSIM/LPIPS/paired-CLIP deltas against the un-watermarked
   generation of the *same* latent. They are not comparable with
   "FID vs MS-COCO real images" numbers from the literature.
10. **`run_p0_queue.sh` writes its `*_done.txt` marker even when a stage exits
    non-zero.** Check the `(rc=...)` in the log, not the marker.

## Third-party provenance

See [`NOTICE.md`](NOTICE.md). `pipeline/` is vendored from MIT-licensed
Tree-Ring / METR code; the carrier design is inspired by RingID's discretization
and Hermitian constraints. No license file is shipped yet — **the repository
owner must pick one before making this public.**

## Citation

Manuscript in preparation. Until it appears, please cite this repository:

```bibtex
@misc{sectorsync,
  title  = {SectorSync: payload-native blind rotation synchronization for
            multi-bit frequency-domain watermarks in diffusion latents},
  year   = {2026},
  note   = {Code: <repository URL>}
}
```
