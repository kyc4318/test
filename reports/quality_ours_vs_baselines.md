# 视觉质量：SectorSync 与 MaXsive / GS 的真实图像配对对比

**日期**：2026-09-21（AutoDL RTX 4090D + 本地拼图分析）
**数据**：服务器 `runs/quality_design`、`runs/quality_maxsive`、`runs/quality_gs`
**脚本**：`measure_quality_arms.py`（新）、`run_visual_quality.py`（新）

## 0. 一句话结论

**这不是「你的方法质量特别差」，而是「PSNR/SSIM/LPIPS 与 CLIP 给出相反排序」。**
同一 prompt + 同一起始潜变量的配对比较下：

| 方法 | n | PSNR ↑ | SSIM ↑ | **LPIPS ↓** | CLIP 均值 | **配对 ΔCLIP [95%CI]** |
|---|---:|---:|---:|---:|---:|---|
| **SectorSync（canonical B8, η=1e4）** | 100 | **11.83** | **0.3954** | **0.5803** | 0.3186 | **−0.0128 [−0.0210, −0.0050]**（显著） |
| 旧 K16 单层（η=1e4，内部版本） | 100 | 12.32 | 0.4205 | 0.6556 | 0.2955 | −0.0360 [−0.0467, −0.0258]（显著） |
| METR R8 | 100 | 9.38 | 0.3164 | 0.7463 | 0.2974 | −0.0341 [−0.0459, −0.0227]（显著） |
| **MaXsive**（官方代码） | 50 | 9.01 | 0.1962 | **0.7387** | 0.3309 | **−0.0008 [−0.0102, +0.0084]**（不显著） |
| GS-256（官方代码） | 100 | 8.73 | 0.1996 | 0.7414 | 0.3319 | **+0.0004 [−0.0068, +0.0072]**（不显著） |

**要点**：

1. **三个像素/感知失真指标（PSNR、SSIM、LPIPS）上，SectorSync 都是四个方法里最好的**
   —— 比 MaXsive 高约 **+2.8 dB PSNR**、LPIPS 低约 **0.16**、SSIM 约 **2 倍**。
   所以「你的图比 MaXsive 差」**不成立**（按这三个指标是反过来的）。
2. **唯一对你不利的指标是 CLIP**：MaXsive / GS 的配对 ΔCLIP 与 0 不可区分，
   而 SectorSync 有约 **−0.013** 的显著下降（区间不含 0）。
3. 两个「感知/语义」指标**互相矛盾**：LPIPS（AlexNet 特征）说你的方法明显更好，
   CLIP 说 MaXsive/GS 更中性。单看任何一列都可能得出相反结论，
   因此**必须配合真实图像的肉眼判断**（见 §2）。
4. 一个可能的误读来源：`runs/visual_quality` 那张表的第 4 列是 **η=2e4**，
   那是刻意过量的设置（CLIP 掉到 0.19、LPIPS 0.80，画面确实明显损坏）。
   主配置是第 3 列 **η=1e4**。

**数据可靠性**：各运行的 `no_w`（无水印）图在四个运行之间**逐字节相同**
（md5 校验），因此 prompt 与起始潜变量完全一致，配对是干净的；
MaXsive 走的是官方仓库 `/root/work/repos/MaXsive`、其官方阈值文件。

## 1. 为什么 CLIP 与 LPIPS 会矛盾

两种可能，目前数据无法区分，需要留意：

* **CLIP 的口径更粗**：单图 CLIP 相似度的逐样本波动很大（本批跨 0.27–0.41），
  配对差 −0.013 虽然显著但相对量级只有约 4%；
* **扰动结构不同**：本方法的载波是环带（半径 10–20、共 940 个频点）上的
  **分布式高频纹理**，而模板类方法（MaXsive / GS）注入的是**低频全局图案**。
  前者可能被 LPIPS 这类局部纹理指标判为「更接近原图」，
  却让 CLIP 的全局语义嵌入发生更大偏移；后者相反。

**这条矛盾本身值得写进论文**（作为「质量评价指标的敏感性」讨论），
但**不能**用它声称本方法在视觉上更好或更差——那要由人眼判断。

## 2. 真实图像（已生成并存档）

* `_vision/COMPARE_ours_vs_maxsive_overview.png`
  —— 4 个样本 × [无水印 / ours B8 η1e4 / MaXsive / GS-256 / METR R8]，
  非参考列下方为 `|ref − cell| × 8` 差分图；
* `_vision/COMPARE_ours_vs_maxsive_detail.png`
  —— 差异最大的 2 个样本，512 px 全尺寸三方对比（无水印 / ours / MaXsive）；
* `_vision/COMPARE_ours_vs_maxsive_zoom.png`
  —— 同一区域中心裁剪 ×3，用于看局部纹理；
* `_vision/CANONICAL_sheet_all.png` / `CANONICAL_sample000_zoom.png`
  —— SectorSync 自身的能量阶梯（η=0 / 5e3 / 1e4 / 2e4）。

## 3. 判定标准（看图像时按这个来）

| 观察 | 结论 | 下一步 |
|---|---|---|
| η=1e4 的差分图只有**均匀噪声/淡淡纹理**，MaXsive 列看起来与无水印无明显差异 | CLIP 的 −0.013 属可接受代价 | 主配置保持 η=1e4，把 CLIP 代价如实写入论文 |
| η=1e4 出现**结构化伪影**（环状、网格状、条带），且明显比 MaXsive 差 | 主配置的视觉代价不可接受 | 把主配置降到 η=5e3 并重跑主表；或在 limitation 中承认 |
| 只对 η=2e4 有明显感觉 | 那是过量设置 | 确认 η=1e4 的观感后再定 |

## 4. 复现命令

```bash
# 配对质量指标（含 MaXsive / GS）
python measure_quality_arms.py --ref runs/quality_design/images/no_w \
  --arms ours=runs/quality_design/images/searched,\
walsh=runs/quality_design/images/walsh,\
maxsive=runs/quality_maxsive/images/maxsive,\
gs=runs/quality_gs/images/gs,\
metr_r8=runs/quality_design/images/metr_r8 \
  --lpips --out_json results/quality_arms.json

# 生成自己的配对图与差分图
python run_visual_quality.py --N 4 --etas 0,5e3,1e4,2e4 --lpips \
  --out_dir runs/visual_quality --report_path results/visual_quality.md
```

## 5. 规模与口径限制

* MaXsive 只有 **N=50**（本批只跑了 50），其余为 N=100；
  三者共享索引 0–49，因此对比公平，但 MaXsive 的区间更宽；
* 质量数字都是**相对同一 latent 的无水印图**的配对失真，**不是**文献口径的
  「FID vs MS-COCO 真实图」，不可与文献数字直接比较；
* CLIP 用的是 ViT-g-14（open_clip 官方权重）。
