# P0 身份识别（ours 进入 key 任务）

**状态**：已跑完（RTX 4090D，2026-09-20，rc=0，用时 13.6 min）。
**脚本**：`run_identity_benchmark.py`。
**数据**：`runs/identity_ours_n20/{rows.jsonl,summary.json,ours_meta.json}`（360 行）。

## 0. 一句话结论

**正面**：闭集身份识别 **Id-Acc@1 = 1.000**（clean / rot45 / rot75 三格全部），
「注册身份 vs 无水印」的 AUC 0.990–1.000、TPR@1%FPR 0.950–1.000、实测 FPR 0.000。
**这是本轮唯一一处「旋转下几乎没有损失」的完整性证据**（rot75 仍 1.000）。

**负面**：把负样本换成**未注册 key 的水印图**（更难、也是更该报的那一类）后，
AUC 掉到 0.868–0.917，在 null 标定的阈值上 **FPR 高达 0.30–0.70**。
与 controls 的错密钥结果同源：**分数度量「有没有水印」，不度量「是不是注册身份」**。

## 1. 协议

| 项 | 设置 |
|---|---|
| 载波 | `design_searched_realgeom.npz`（B=8，η=1e4） |
| key 空间 | **256**（= 2^8，即 payload 空间；ours-B 的上限就是 2^B） |
| 注册身份数 | **20**（脚本每张测试图对应一个注册 payload） |
| 未注册负样本 | 20 个注册表之外的 payload |
| 划分 | 校准 = prompt 0–19，测试 = prompt 500–519（互不相交） |
| 攻击 | clean / rot45 / rot75；搜索网格 2° |
| 阈值 | FPR 目标 1%，取自校准划分的顺序统计量 |

> 口径提醒：`key 空间 256` 与 `注册量 20` 是两个不同的数，表中分列。
> 由于 ours 的身份 = payload 精确匹配，**Id-Acc 本质上等于 PMR**（不是独立能力），
> 不可与 SFWMark 的 2048-key Id-Acc 直接换算或宣称「追平/胜出」。

## 2. 结果

| 攻击 | Id-Acc@1 [95%CI] | margin 均值 | AUC(注册 vs **无水印**) | TPR@1%FPR | 实测 FPR(null) | AUC(注册 vs **未注册**) | 实测 FPR(未注册) |
|---|---|---:|---:|---:|---:|---:|---:|
| clean | **1.000** [0.839, 1.000] | 190.6 | 1.000 | 1.000 | 0.000 | 0.917 | **0.500** |
| rot45 | **1.000** [0.839, 1.000] | 35.1 | 0.990 | 0.950 | 0.000 | 0.905 | 0.300 |
| rot75 | **1.000** [0.839, 1.000] | 43.7 | 1.000 | 1.000 | 0.000 | 0.868 | **0.700** |

（n = 20 张测试图/格。Id-Acc 的 CI 宽是因为 n=20；FPR 的经验分辨率是 1/20 = 5%。）

## 3. 读数

1. **身份是可读取的**：20 个注册 payload 在 256 的 key 空间里被全部正确识别，
   且旋转 45°/75° 下不退化（margin 从 190.6 降到 35–44，仍远大于 0）。
2. **但「拒绝未注册 key」不行**：在存在性阈值下，未注册 key 的图像有 30–70% 被
   当成某个注册身份接受。原因是水印本身**合法且可读**，只是 payload 不在注册表里；
   `mean|ℓ|` 类分数无法区分这两种情形。
3. **正确做法**（与 controls 的处方一致）：把「未注册 key」当作标定负样本，
   单独定一个更高的拒绝阈值；`run_identity_benchmark.py` 已经同时给出这两列，
   正式版必须用后者（而不是 null 阈值）来报 TPR/FPR。

## 4. 一次被作废的运行（复现时必看）

第一版把**冒烟（`--N 2`，注册表 2 个 key）**与**正式跑（`--N 20`，注册表 20 个 key）**
写进了同一个 `--out_dir`。`ResumeLog` 按 `uid` 去重，于是图片 0/1/500/501 保留了
冒烟时的 2-key 记录，**同一次「运行」里混进了两种注册表大小**（校验：
`n_keys_registered` 取值同时出现 2 与 20）。
污染目录已改名为 `runs/identity_ours_CONTAMINATED_mix`，本文用的是重跑后的
`runs/identity_ours_n20`（校验：360 行全部 `n_keys_registered = 20`）。

**教训**：冒烟与正式跑必须用不同的 `out_dir`；`ResumeLog` 的续跑是优点，
但也会把旧配置的行静默留下。建议在 `run_config.json` 里加一个「参数指纹」并在
启动时与已有 `rows.jsonl` 的指纹比对，不一致就直接报错。

## 5. 尚未做

* **SFWMark HSTR/HSQR / RingID / MaXsive 的并列对比**没跑（本会话只跑 ours），
  因此本文目前**没有**可用的三方法 identity 表；
* 注册量只有 20（受每图一个 payload 的实现限制），正式版应扩到 128 或 256，
  并把「未注册」负样本的量化阈值一起冻结。

## 6. 复现命令

```bash
cd /root/sector_watermark && conda activate sector
python -u run_identity_benchmark.py --design results/design_searched_realgeom.npz \
  --N 20 --n_keys 128 --methods ours --cases clean,rot45,rot75 \
  --calib_offset 0 --test_offset 500 --out_dir runs/identity_ours_n20
```

（`--n_keys 128` 只决定 key 池大小；实际注册量 = `--N`。
ours-8 的 key 空间上限是 256，因此 `--n_keys 2048` 不可能成立。）
