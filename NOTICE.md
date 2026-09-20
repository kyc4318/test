# Third-party provenance

SectorSync's own code is the work of the repository owner. The following
components are derived from, or interact with, third-party research code. They are
listed here so that attribution and licensing stay clear.

## Vendored code (included in this repository)

`pipeline/` is vendored and adapted (flattened layout) from two MIT-licensed
projects:

* **METR** — `deepvk/metr` (MIT)
* **Tree-Ring Watermark** — `YuxinWenRick/tree-ring-watermark` (MIT)

Files affected: `pipeline/inverse_stable_diffusion.py`,
`pipeline/modified_stable_diffusion.py`, `pipeline/optim_utils.py`.
The upstream MIT terms apply to those files; keep their copyright notices intact
when redistributing.

## Design inspiration (no code copied)

* **RingID** — `showlab/RingID` (ECCV'24). The sector discretization and the
  Hermitian conjugation constraint used by our carriers follow the conventions
  described in that work.
* **Gaussian Shading** — `YuanyouYin/Gaussian-Shading` (CVPR'24) and
  **SFWMark** (ICCV'25). These are used **only as external baselines** for the
  comparison tables. Their code is *not* included here; `run_paper_compare.py`
  imports them from a separate checkout (see the README section "External
  baselines are not vendored"). Obtain them from upstream and respect their
  licenses.
* **MaXsive** (ACM MM'25) and **AnchorMark** are discussed in the manuscript; no
  code from either is used.

## Models and data (not included)

* **Stable Diffusion 2.1-base** — CreativeML Open RAIL-M license. Download it
  yourself (see the README); it is not redistributed here.
* **CLIP ViT-g-14** (`laion/CLIP-ViT-g-14-laion2B-s12B-b42K`) — used only as a
  quality metric. Not redistributed.
* **COCO 2014 5k test split** (prompts + 512x512 images) — used as the prompt
  source and, optionally, as the FID reference set. Not redistributed.

## Repository license

**No LICENSE file is shipped yet.** Before publishing this repository publicly,
the owner must choose a license. Two practical options:

* **MIT / Apache-2.0** for the whole repository, keeping the vendored `pipeline/`
  files under their original MIT terms (a short note in `NOTICE.md`, as above,
  is sufficient for MIT-to-MIT compatibility).
* A research-only / non-commercial license, if the intent is to restrict reuse.

Because `pipeline/` contains MIT-licensed third-party code, any repository-wide
license must be compatible with MIT (MIT and Apache-2.0 both are).
