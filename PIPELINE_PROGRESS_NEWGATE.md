# BF16 GEMM — 冷启动 lane `coldstart_newgate_20260819`

本文件只服务这一条 lane。容器里还有另外两条线的接力物，**不要混写、也不要把它们的成绩当本 lane 的起点**：

| 文件 | 属于哪条线 | 状态 |
|---|---|---|
| `PIPELINE_PROGRESS.md` | QD 线（v98） | 历史，只读 |
| `PIPELINE_PROGRESS_GREEDY.md` | `greedy_coldstart_20260817`，跑到 1.4045 | 历史，只读；本 lane 的对照组 |
| **`PIPELINE_PROGRESS_NEWGATE.md`（本文件）** | `coldstart_newgate_20260819` | **在跑** |

机器每几小时回收一次并从 docker 快照恢复，**本文件是唯一可靠的交接物**。每完成一个阶段就更新它。

---

## 1. 这个实验要回答什么

`greedy_coldstart_20260817` 用了两天半、8 波、约 26 个 engineer 方向，从同一个种子跑到 11 例套件 geomean **1.4045**。事后复盘发现，真正吃掉产出的不是"优化不出来"，而是判据：

- **suite geomean 在数学上接不住单路线胜利。** 11 例非加权几何平均，单条路线快 7% 只把 geomean 推动约 0.6%。一个 verified、correctness 过、policy 干净的三补丁栈（绝对 1.42619，相对 +1.58%）就是这样被 `MIN_IMPROVE=0.02` 拒掉的——同一份 per-case，`route_gate.py` 的判决是 ACCEPT。
- **per-route gate 写好了但从没运行过。** 它唯一的喂食方式是 `args.route_bands`，而规范调用里从来没有这个键；7 个运行轨迹里 grep `[gate r` 零命中。
- **停机计数器用同一把错尺子。** `madeProgress` 也拿 suite geomean 比，于是真实的过 band 胜利被记成 stall，波次带着三分之一没花完的预算退出（wave 6：3 轮，12 里用了 8）。
- **IR/汇编层自我关闭。** 父归档只在 commit 成功时才被赋值，而大部分轮次不提交 → 父归档恒为 null → 11 次干净采集、0 个可读判决。

分支 `gate/per-route-default` 改的就是这四条。**本 lane 是它的配对对照**：同一个种子、同一套 harness、同一个 oracle、同样的 budget 12 和六张卡，唯一的变量是 workflow 代码。所以两条曲线之间的差可以归因到 gate 的改动上。

对照对象（只读，不要动）：
- `exp/state_greedy_coldstart_20260817` —— 旧 lane 的累计状态，终点 1.4045
- `PIPELINE_PROGRESS_GREEDY.md` —— 它的逐轮叙事，6402 行

## 2. 实验定义（硬约束，任何一波都不许改）

**参数不写在这里。** 上一条 lane 的代价是具体的：§2 被手抄时漏了 `min_improve`，那波就跑在 0.02 默认值上，拒掉了两天里最大的一个结果。参数现在住在版本控制里的一个文件：

```
kernel_workflow/lanes/coldstart_newgate_20260819.json
```

发起一波的**唯一**方式：

```bash
python3 kernel_workflow/scripts/lane_args.py --check kernel_workflow/lanes/coldstart_newgate_20260819.json   # 必须 exit 0
python3 kernel_workflow/scripts/lane_args.py --print kernel_workflow/lanes/coldstart_newgate_20260819.json   # 渲染出 Workflow(...)
```

然后**原样**调用 `--print` 出来的那个调用。不要手抄，不要增删任何键。`--check` 非 0 就停下来修文件，不要绕过它。

评分口径：11 例套件 geomean，相对 **direct rocblas_gemm_ex**。

## 3. 不要做的事

- **不要从任何 incumbent 播种。** 具体是：`exp/state_greedy_coldstart_20260817`、`exp/task_bf16_*`、`exp/v98_*`、`exp/qd_*`、`exp/opt_bf16_20260814`、`exp/round1*`、`exp/dense_bf16_gemm_budget30_state`。本 lane 的意义就在于它是冷启动，一旦热启动，与旧 lane 的对比即失效。lane 自己的 `state_dir` 从第一轮起会被 workflow 填满，那是正常接力，不是热启动。
- **不要改 `examples/tasks/dense_bf16_gemm_fused` 本身**（三条 lane 的共同种子，改了就都不可比了）。workflow 会自己复制工作区。
- **不要改 `harness_lib.py` / `scripts/task_runner.py` / oracle**。它们的 SHA256 记在 `PROVENANCE.json`，度量被扰动等于结论作废。
- 候选策略不变：候选实现禁止调用 rocBLAS / hipBLAS / hipBLASLt / Tensile / Composable Kernel / MIOpen。`src/rocblas_baseline.cpp` 只作为不可变 oracle/分母存在，policy scan 用 `--immutable` 豁免它，绝不豁免候选产物。
- 任何"物理上不可能"的加速比先当成度量事故排查，不要写进结论。

## 4. 度量纪律

所有 GPU 命令（compile / correctness / performance / profile）一律经过 gpu_lock：

```bash
GEAK_GPU_ALLOWED=2,3,4,5,6,7 bash kernel_workflow/scripts/gpu_lock.sh \
  2,3,4,5,6,7 python3 scripts/task_runner.py {compile|correctness|performance}
```

没有计时收据或未预热的结果不可发布。

## 5. 换机后的开工检查单（这条 lane 预计会用很多次）

旧 lane 两天半换了 9 次机器，每次都要重新注册纪元、重测底噪、修被打红的测试。`gpu_lock.sh` 对 `check_measurement_frame.py` 的 exit 4 **直接拒绝所有 GPU 命令**，所以这一步绕不过去：

```bash
python3 kernel_workflow/scripts/check_measurement_frame.py
```

- **exit 0** —— 纪元与本机相符且底噪已测，直接开工。
- **exit 3** —— 纪元已注册但是 provisional，整表都是 `DEFAULT_NOISE_FLOOR = 0.072`。可以跑，但任何窄于 7.2% 的胜利都不可采信，所以第一件 GPU 工作应该是测底噪。
- **exit 4** —— 本机没注册纪元，或纪元与 `CURRENT_MACHINE` 不符。**必须**给本机注册一个**新的**纪元字母（绝不复用退休字母——同一台机器的第二个容器要新字母，那是 finding 126），然后：

```bash
python3 kernel_workflow/scripts/measure_noise_floor.py   # 8 次同变体全套预热重复
python3 kernel_workflow/scripts/deprovisionalize_epoch.py --verdict <json> --machine <字母> --apply
```

注册纪元目前需要改 `kernel_workflow/scripts/noise_floor_stats.py` 源码（`MACHINE_HOSTNAME`、`PROVISIONAL_MACHINES`、`CURRENT_MACHINE`）。这是已知的架构债：通用工具里焊死了任务与机器专属知识。改完跑一遍 `python3 -m unittest test_noise_floor_stats test_check_measurement_frame`，别把测试留红。

## 6. 这一波要盯的三件事（相对旧 lane 的新行为）

1. **Setup 阶段的 effective-config 回显。** lane 现在会在第一个 agent 调用之前打印生效配置，不是来自 args 的值标 `(default)`。确认 `min_improve=0.005` 不带 `(default)`。
2. **`Commit gate: PER-ROUTE, N bands (derived from baseline_per_case[].samples_ms)`。** 这行如果没出现，说明 benchmark engineer 没交回 ≥3 次预热重复，gate 退回了 suite 阈值——那本 lane 最主要的变量就没生效，要先修这个再继续。
3. **`isa_evidence` 是否真的产出了 verdict。** 观察模式下父归档现在由 verifier 自己采集并按 hash 采纳。旧 lane 是 11 次干净采集、0 个可读判决；这一波如果还是 0，说明采纳路径没走通。

## 7. 进展日志（每波追加，最新在最上面）

<!-- 还没有。第一波尚未启动。 -->
