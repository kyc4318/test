# 规模化评测计划：与文献协议对齐（v1 canonical 方法）

**日期**：2026-09-21｜**方法**：canonical SectorSync（replace embedding，与本仓库首发完全一致）
**状态**：等待项目所有者选择规模后执行

## 0. 结论先行

方法已回退/冻结为 canonical。当前论文最大的短板不再是方法，而是**评测规模与协议**：
现有主表是 **N=50 的 pilot**、**8 个攻击格**、**单一 prompt 源（1000 条 COCO caption）**、
**单一模型（SD2.1）**、**单 seed**。文献协议与此相差 20 倍规模。

本轮已补齐最关键的**前置条件**：把 prompt 池从 1000 条扩到 **5000 条**
（`meta_data_5k.json`，与原 1000 条逐条一致，见 §2）。

## 1. 与文献协议的差距

| 项目 | 文献常见设置 | 本项目现状 | 差距 |
|---|---|---|---|
| prompt 源 | Stable-Diffusion-Prompts 1k–8k / DiffusionDB 0.5–1k / ImageNet 类别 prompt | **COCO-5k 现可 5000 条**（本轮补齐） | ✅ 已够；若要跨源复现需额外下载 |
| 数据集 | MS-COCO 2017（5k 图或 caption） | `coco5k`（1000 caption → 现 5000）+ `images512` 5000 张真实图 | ✅ 够做 FID@5000 |
| 生成模型 | SD 1.4/1.5/2.0/2.1；新模型 FLUX/SDXL/PixArt | SD2.1 ✅；**SD1.5 已补全**（91 文件/15 GB） | 跨模型可做；FLUX/SDXL 未下载 |
| 分辨率 / latent | 512×512 / 4×64×64 | 一致 | ✅ |
| 生成 / 反演 | DDIM 50 步、CFG 7.5、空 prompt 反演 50 步 | 一致 | ✅ |
| 检测样本量 | **1000 有水印 + 1000 无水印** | **N=50**（+100 质量子集） | ❌ 20× |
| seed | 3 个 | 1 个 | ❌ |
| 攻击集 | WAVES 26 类（3 数据源）/ Stirmark ~88 类 | 8 主格 + WAVES 8 类×2 强度（经 MaXsive 官方实现，N=50） | ⚠️ 部分 |
| FID | 5000 张生成图 | 无（仅 N=100 相对 FID） | ❌ |
| 多用户识别 | 1024–4096 用户 / 32–32768 keys | 注册 20 个 payload（B=8 上限 256） | ❌ 需 B12/B16（**已具备** `design_B12_s0` / `design_B16_s0`） |
| 统计口径 | TPR@0.1%FPR 等 | Wilson/bootstrap + conformal 阈值（已做） | ✅ 方法论已就绪 |

## 2. 本轮补齐的前置条件：prompt 池 1000 → 5000

`/root/autodl-tmp/data/coco5k_raw/test_5k_mscoco_2014.csv`（标准 MS-COCO 5k 测试集，
每图 5 条 caption）此前**从未被转换成项目需要的 meta 格式**，所以一直只能用 1000 条。

新增 `make_coco_meta.py` 完成转换，并做了**一致性校验**：

```text
5000 captions -> /root/autodl-tmp/data/coco5k/meta_data_5k.json
  vs meta_data.json: 1000/1000 of its entries reproduced exactly
```

**这 1000/1000 的精确复现意味着**：既有全部结果（索引 0–999）在新文件下依然有效，
同时可用索引扩展到 4999，从而支持

* 配对主表：wm 与 null 用**同一批**索引（逐样本配对）；
* 独立的校准块（例如 1000–1999），不再需要 `test_offset=500` 的临时绕法；
* 文献规模：索引 0–999 做 wm，1000–1999 做独立校准。

## 3. 资源盘点（服务器，2026-09-21）

| 资源 | 状态 |
|---|---|
| GPU | RTX 4090 D 24 GB，空闲 |
| 磁盘 | `/` 19 GB 可用、`/root/autodl-tmp` 22 GB 可用 ⚠️ 大规模跑图需注意 |
| 模型 | SD2.1、**SD1.5（已补全）**、CLIP ViT-g-14 |
| 基线代码 | **Gaussian-Shading / SFWMark / RingID / MaXsive / tree-ring 全部已克隆** |
| 攻击套件 | WAVES 经 MaXsive 官方 `apply_single_distortion` 已接入（8 类×2 强度已跑过 N=50）；**Stirmark 未安装** |

## 4. 实测速率与成本模型（4090D，SD2.1，50 步）

| 环节 | 实测 |
|---|---|
| 生成一张 | ≈1.5 s |
| 反演一张 | ≈1.2 s |
| 解码（180 角、2° 网格） | ≈0.8 s |

因此每个（方法, seed）在 **8 个攻击格**下的成本 ≈
`N×1.5s`（生成）+ `8×N×2.0s`（反演+解码）+ 共享的 null 生成/反演。

## 5. 三个可选规模（请挑一个）

| 方案 | 配置 | 估计耗时 | 产出与口径 |
|---|---|---|---|
| **A 最小** | N=200，8 格，ours + GS-8（同容量） | ≈3 h | 比例指标 Wilson 半宽 ≈±3%；可作"pilot+"，仍非文献规模 |
| **B 推荐** | **N=1000，8 格，ours + GS-8 + GS-256 + SFWMark×2**，1 seed | ≈30–36 h（1.5 天） | **与文献规模对齐的主表**（1000 wm + 1000 null，配对） |
| **C 完整** | B + 2 个额外 seed + WAVES 完整 26 类 + SD1.5 跨模型 + FID@5000 | 再加 2–3 天 | 可投稿强度；需要磁盘扩容（生成图 5000 张/配置） |

多用户识别（1024–4096 users）可独立安排：需要 B≥12 的设计，
`design_B12_s0.npz` / `design_B16_s0.npz` 已在仓库中，脚本 `run_identity_benchmark.py` 现成。

## 6. 执行顺序建议（不论选哪个规模）

1. 先把 **P0 的 controls 闸门**用 5000 条 prompt 重跑一次（校准 0–999 / 测试 1000–1999），
   确认新的划分下闸门仍通过——这是所有后续大表的统计地基，成本约 1.5 h；
2. 再跑所选规模的主表；
3. 最后补 WAVES / 跨模型 / FID / identity。

## 7. 尚未解决、需要你决定的事项

* 是否要引入**其他 prompt 源**（Stable-Diffusion-Prompts / DiffusionDB / ImageNet 类别）；
  当前 COCO-5k 已够文献规模，但换源可回应"单一 prompt 分布"的质疑；
* 是否安装 **Stirmark 3.1**（88 类攻击；WAVES 已覆盖其中大部分）；
* 磁盘：方案 C 需要扩容或改成分批落盘 + 及时删除中间图。
