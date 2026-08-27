# BF16 GEMM — 冷启动 lane `coldstart_ladder_20260820`

> **这是一份带日期的运行日志，不是操作手册。** 下面的记录里有一整套命令现在跑不了了：
> `check_measurement_frame.py`、`register_epoch.py`、`measure_noise_floor.py`、
> `noise_floor_stats.py`、`route_floors.py`、`deprovisionalize_epoch.py` 随每机纪元子系统
> 一并删除，per-route 底噪表也一起没了。当时这些命令是真的跑过、结果是真的——所以正文**不改**，
> 改了就等于伪造记录。但**不要照着执行**。
>
> 两处具体的失效：纪元字母（N/P/Q/R/T/V）已经无法解析回一个底噪百分比，唯一幸存的读数是
> epoch R 在 tw008 的 7.12%；而 `if (winner && ROUTE_BANDS)` 这个门条件已经不存在了，
> `routeGate` 现在在 `judgeCandidate` 里无条件调用，只是仍然只对通过 CANDIDATE_FLOOR 的候选
> 生效——结论没变，机制变了。
>
> 当前可用的对应物：封闭机制清单用 `closed_mechanisms.py --brief`，
> 天花板/余量用 lane 的 `SOL_STATUS`。


本文件只服务这一条 lane。容器里还有另外三条线的接力物，**不要混写、也不要把它们的成绩当本 lane 的起点**：

| 文件 | 属于哪条线 | 状态 |
|---|---|---|
| `PIPELINE_PROGRESS.md` | QD 线（v98） | 历史，只读 |
| `PIPELINE_PROGRESS_GREEDY.md` | `greedy_coldstart_20260817`，终点 1.4045 | 历史，只读 |
| `PIPELINE_PROGRESS_NEWGATE.md` | `coldstart_newgate_20260819`，wave5 时 1.2707 | 历史，只读；本 lane 的对照组 |
| **`PIPELINE_PROGRESS_LADDER_20260820.md`（本文件）** | `coldstart_ladder_20260820` | **在跑** |

机器每几小时回收一次并从 docker 快照恢复，**本文件是唯一可靠的交接物**。每完成一个阶段就更新它。

---

## 1. 这个实验要回答什么

同一个种子、同一套 harness、同一个不可变 oracle、同样的 budget 12 和六张卡，第三次冷启动。唯一改变的是 workflow 代码里的**证据阶梯**（提交 `686d9521`）：

- **L3 不再读反汇编，改读 lowering 轨迹。** `ir_capture.py` 用 `-mllvm -print-changed=quiet` 重放构建自己的设备编译命令并切分成逐 pass 的 stage，`ir_signals.py` 在其中导航、给相邻 pass 的 delta 排序。旧 L3 的输出是 `vgpr=152, scratch=0`——一堆没有指向任何 pass 的真数字；新 L3 的输出是一个 **pass** 加一个 **structure**，例如「`si-insert-waitcnts` 在 stage 93 加了 14 个同步操作」。
- **L4 在同一轮内进入。** 旧代码只能等下一轮，而停机预算会先结束 lane（`noImprove` 到 2 就退出，`MAX_NO_IMPROVE` 默认 2）。旧代码里唯一活着的到达最深层的路径是 `priorRefuted`，而它**恰好跳过了 L3**——最深的那一级只能由那条什么都不给它的路走到。
- **驳斥现在指向 L3，而不是绕过它。**「我的改动没进二进制」首先是一个「哪个 pass 撤销了它」的问题。
- **L4 先查知识层。** `perf_knowledge/compiler_grounding/`（pass 导航图、问题模板、实测的后端不变量），Tier 1 优化 remark 和 Tier 2 后端源码只在前面没解决时才走。
- **记录带版本号。** `evidence.model = geak.evidence-ladder/v2`，`requested_stage` 与 `reached_stage` 分开记。在这之前 bank 的轮次读作 `legacy_machine_code`，不满足任何一级——resume 的 lane 不许继承没人产出过的证据。

所以本 lane 与 `coldstart_newgate_20260819` 的逐轮曲线之差，可以归因到证据阶梯上：参数文件里每一个旋钮都是那条 lane 的取值，唯一的差别是代码。

对照对象（只读，不要动）：
- `exp/state_coldstart_newgate_20260819` —— 上一条 lane 的累计状态，wave5 时 cumulative 1.2707，31 条 insight、30 条 ledger
- `exp/state_greedy_coldstart_20260817` —— 更早那条，终点 1.4045
- 这两个数字不是要超过的指标，它们是用来比对曲线形状的

## 2. 实验定义（硬约束，任何一波都不许改）

**参数不写在这里。** 代价有记录：某一波的调用被手抄时漏了 `min_improve`，那波就跑在 0.02 默认值上，拒掉了两天里最大的一个结果。参数住在版本控制里的一个文件：

```
kernel_workflow/lanes/coldstart_ladder_20260820.json
```

发起一波的**唯一**方式：

```bash
python3 kernel_workflow/scripts/lane_args.py --check kernel_workflow/lanes/coldstart_ladder_20260820.json   # 必须 exit 0
python3 kernel_workflow/scripts/lane_args.py --print kernel_workflow/lanes/coldstart_ladder_20260820.json   # 渲染出 Workflow(...)
```

然后**原样**调用 `--print` 出来的那个调用。不要手抄，不要增删任何键。`--check` 非 0 就停下来修文件或修纪元，**不要绕过它**。

评分口径：11 例套件 geomean，相对 **direct rocblas_gemm_ex**。

## 3. 不要做的事

- **不要从任何 incumbent 播种。** 具体是：`exp/state_coldstart_newgate_20260819`、`exp/state_greedy_coldstart_20260817`、`exp/task_bf16_*`、`exp/v98_*`、`exp/qd_*`、`exp/opt_bf16_20260814`、`exp/round1*`、`exp/dense_bf16_gemm_budget30_state`。本 lane 的意义就在于它是冷启动。lane 自己的 `state_dir`（`exp/state_coldstart_ladder_20260820`）从第一轮起会被 workflow 填满，那是正常接力，不是热启动。
- **不要把另外三条线的进度文件当成本 lane 的进度。** 可以当历史资料读，但它们的结论不是本 lane 的起点。
- **不要改 `examples/tasks/dense_bf16_gemm_fused` 本身**（四条 lane 的共同种子，改了就都不可比了）。workflow 会自己复制工作区。
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

`gpu_lock.sh` 对 `check_measurement_frame.py` 的 exit 4 **直接拒绝所有 GPU 命令**，所以这一步绕不过去：

```bash
python3 kernel_workflow/scripts/check_measurement_frame.py
```

- **exit 0** —— 纪元与本机相符且底噪已测，直接开工。
- **exit 3** —— 纪元已注册但是 provisional，整表都是 `DEFAULT_NOISE_FLOOR = 0.072`。注意本 lane 的 `route_bands` 是 `@current_epoch`，而 `route_floors.py` **拒绝** provisional 表，所以 exit 3 状态下 `lane_args.py --check` 会拒绝发起——必须先测底噪。
- **exit 4** —— 本机没注册纪元，或纪元与 `CURRENT_MACHINE` 不符。**必须**用脚本注册一个**新的**字母，不要手改 `noise_floor_stats.py`（手改在 §13.18 出过一次静默 no-op 事故，`register_epoch.py` 就是为它写的：它做那四处编辑、逐处断言改到了、改完重新 import 自检，不过就回滚）：

```bash
python3 kernel_workflow/scripts/register_epoch.py --letter <新字母> --host $(hostname) --note '<哪次换机>'
python3 kernel_workflow/scripts/measure_noise_floor.py --task <kernel_path> --repeats 8 --machine <字母> --out <json>
python3 kernel_workflow/scripts/deprovisionalize_epoch.py --verdict <json> --machine <字母> --apply
```

整个 sweep 要包在**一次** `gpu_lock.sh` 单卡调用里，保证 8 次重复落在同一张卡上。

**字母必须是新的，绝不复用退休字母**（finding 126）。截至本 lane 启动，`L M N O P Q R S T U V W X Y Z A B C` 全部已烧掉；**下一个可用字母是 `D`**，之后是 E F G H I J K。

**启动时的机器是 tw045，未注册，所以第一件事就是注册 `D` 并测底噪。**

## 6. 启动时已知的红测试（4 个，已定位，不要让它们吃掉一波）

`python3 -m unittest test_noise_floor_stats test_check_measurement_frame` 在 tw045 上是 `FAILED (failures=3, errors=1)`。四个都已经查清，**都不阻塞本 lane**：

1. `test_the_epoch_letter_matches_the_host` —— 只要本机还没注册纪元就必然失败（`CURRENT_MACHINE` 还是 C→tw042）。**注册 `D` 之后自动消失，不是缺陷。**
2. `test_an_unknown_machine_gets_the_widest_floor_anywhere_not_another_machines`（0.0764 != 0.0088）
3. `test_an_unknown_route_gets_the_widest_floor_not_the_narrowest`（0.0764 vs 0.072）

   这两个是**同一个根因**：纪元 Z 的 `decode_m8_up` 实测到 0.0764，超过了 `DEFAULT_NOISE_FLOOR = 0.072`。这两个测试编码的不变量是「fail-closed 默认值必须不窄于任何地方实测过的最宽底噪」，好让未知路线/未知机器拿到最保守的界。修法是把那个常量抬到 ≥0.0764，方向是**更保守**、安全。但它**与本 lane 无关**：`route_floors.py` 直接拒绝 provisional 表，所以 `DEFAULT_NOISE_FLOOR` 根本到不了本 lane 的 gate。要修就顺手修，别为它开一波。
4. `test_host_override_reproduces_the_historical_failure`（ERROR: `TypeError: expected str, bytes or os.PathLike object, not NoneType`）—— 测试自身的缺陷：它从 `MACHINE_HOSTNAME` 里挑「另一台主机」时没有过滤掉 hostname 为 `None` 的前约定字母 `L`/`M`，于是把 `None` 传给了 `subprocess.run`。修法是挑选时跳过 `None`。

## 7. 这一波要盯的四件事（相对上一条 lane 的新行为）

1. **Setup 阶段的 effective-config 回显。** 确认 `min_improve=0.005`、`candidate_floor=0.29`、`isa_evidence=observe`、`ir_diagnostics=on` 四个都**不带 `(default)` 标记**。带了就说明参数没送进去，本 lane 的变量没生效。
2. **`Commit gate: PER-ROUTE, N bands (derived from baseline_per_case[].samples_ms)`。** 这行没出现说明 benchmark engineer 没交回 ≥3 次预热重复，gate 退回了 suite 阈值。
3. **L3 是否真的返回了 pass 名 + stage 变化。** `admissibleAttribution` 会拒绝没有 `attributed_pass` 和 `stage_transition` 的 `attributed` / `needs_compiler` 返回。上一条 lane 是 11 次干净采集、0 个可读判决；这一波如果 L3 仍然给不出 pass，先查 `ir_capture.py` 的 provenance 校验（轨迹来自重编译，必须逐 kernel 证明与实测对象一致，否则那些 stage 描述的是"某个"程序而不是"那个"程序）。
4. **L4 是否在同一轮内被进入过。** 这是这次改动最主要的可观测后果。如果整波没有一次 `needs_compiler` → L4，记下来是"没触发"还是"触发了但没进"——两者要修的地方完全不同。

## 8. 进展日志（每波追加，最新在最上面）

<!-- 还没有。第一波尚未启动。 -->
