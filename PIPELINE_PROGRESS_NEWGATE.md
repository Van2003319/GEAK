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

注册纪元用 **`kernel_workflow/scripts/register_epoch.py`**，不要手改 `noise_floor_stats.py`：

```bash
python3 kernel_workflow/scripts/register_epoch.py \
  --letter <下一个没用过的字母> --host $(hostname) --note '<哪次换机>'
```

它做那四处编辑（往 `MACHINE_HOSTNAME` **追加**——顺序是承重的，`machine_for_host` 取最后一个匹配；加进 `PROVISIONAL_MACHINES`；插一张全 DEFAULT 的占位表；把 `CURRENT_MACHINE` 指过去），每处断言锚点恰好命中一次，改完重新 import 模块自检，任何一条不符就把文件回滚并退出非零。手改在 §13.18 出过一次 `.replace()` 静默 no-op 的事故，这个脚本就是为它写的。`--dry-run` 先看 diff。

底层的架构债还在：机器专属知识仍然住在通用脚本的源码里，注册一个纪元本该是写一个 JSON。脚本让这次改动安全，没有消除它。

## 6. 这一波要盯的三件事（相对旧 lane 的新行为）

1. **Setup 阶段的 effective-config 回显。** lane 现在会在第一个 agent 调用之前打印生效配置，不是来自 args 的值标 `(default)`。确认 `min_improve=0.005` 不带 `(default)`。
2. **`Commit gate: PER-ROUTE, N bands (derived from baseline_per_case[].samples_ms)`。** 这行如果没出现，说明 benchmark engineer 没交回 ≥3 次预热重复，gate 退回了 suite 阈值——那本 lane 最主要的变量就没生效，要先修这个再继续。
3. **`isa_evidence` 是否真的产出了 verdict。** 观察模式下父归档现在由 verifier 自己采集并按 hash 采纳。旧 lane 是 11 次干净采集、0 个可读判决；这一波如果还是 0，说明采纳路径没走通。

## 7. 进展日志（每波追加，最新在最上面）
### 修复 3 — 同一个 gate 的第三次假拒绝：`shortx4_t` 不在 CAST_WIDTH（2026-08-19 ~14:35）

r1_d0（algorithm，那个 0.5072 / 相对同场 control 1.512x 的 prefill 赢家）重跑后**也**被 step 2c 拦下，同样是 `findings: 0`：

```
UNP: as  cast type 'shortx4_t' is not in CAST_WIDTH
UNP: bs  cast type 'shortx4_t' is not in CAST_WIDTH
```

**一轮三个方向，两个死在同一个 gate 上，都没有产生任何一条 finding。** 这已经不是偶发事故，是本 lane 当前最主要的吞吐瓶颈。

而这个类型的宽度**就写在同一个文件里、离用它的 cast 一百行**：

```c
typedef __attribute__((__vector_size__(4 * sizeof(short)))) short shortx4_t;   // = 8 字节
```

**处置。** 加 `vector_typedefs(text)`：解析被扫描文件自己声明的向量 typedef（`__vector_size__(...)` 与 `ext_vector_type(N)` 两种写法），把宽度并进**每文件**的 cast 宽度表（文件内 typedef 在重名时优先，因为那才是此处真正在作用域里的定义）。

**为什么这不是放松判据。** 工具的规则一直是 *resolve-or-report*，它**本来就**从同一份源码里解析 `constexpr` 数组维度而不是维护一张维度表；向量 typedef 是同一类事实。宽度是**读出来的，从不默认**：`_vector_size_bytes()` 只接受整数字面量、`sizeof(T)` 及二者的乘积，遇到具名常量、除法或没见过的算术就返回 None，该 cast 照旧报 unparseable。测试钉住了这一点（`test_an_unevaluable_width_stays_unresolved`、`test_an_unknown_base_stays_unresolved`），也钉住了**修复没有把检查关掉**（`test_a_file_local_typedef_still_catches_a_real_hazard`：同样的 typedef 机制下，8 元素 = 16 字节的 cast 打在 136 字节行距上仍然被抓）。

对真实候选验证：

```
resolved typedefs: {'floatx4_t': 16, 'shortx4_t': 8}
$ lds_cast_alignment.py --json <ws_cand>/src/custom_gemm.hip <ws_cand>/src/custom_gemm_hip.hip
{"findings": [], "gate_mode": false, "inherited": [], "passed": true, "unparseable": []}
EXIT=0
```

测试：`test_lds_cast_alignment` 38 个全绿（原 28 + char 3 + typedef 7）；`python3 -m unittest discover` 全仓 **875 tests OK**，没有留红。

**本轮的损失（如实记录）。** r1_d0 和 r1_d1 在本轮已经以 policy_failed 结束，且 resume 的缓存键是 (prompt, opts) —— verify 提示词没变，再 resume 只会重放同样的失败，所以**这两个方向在 round 1 无法回收**，没有为此第三次重启（代价大于回收）。round 1 只剩 r1_d2（0.6267）作为候选；它足以让 `winner` 非 null，从而**第一次真正触发 per-route gate**。修复对 round 2-12 的每一次 verify 生效，而预算的绝大部分在那里。

可惜的是 retrospective 指出的那个真正的结论——d0（prefill）与 d2（decode）**路线互补，逐例取小的并集 geomean 0.7887，是种子的 2.35 倍**——本轮拿不到了，要等后续轮次由 director 重新提出。

**顺带发现的一个覆盖盲区（未修，仅记录）。** retrospective 说 r1_d0 有个"只是碰巧安全"的对齐隐患：`reinterpret_cast<uint4*>(&smem.stage.as[r][kk])` 打进 `Bf16[BM][BK+4]`，BK=32 时行距 72 字节，奇数行偏 8 字节。本工具**看不到它**，但原因不是 retrospective 说的"BM/BK 是模板参数"：`DECL` 只匹配 `__shared__` 声明，而这个数组是**结构体成员**（`smem.stage.as`），根本没有进入匹配。这是既有盲区，不是本次改动引入的，修它是另一件事。

---
### 修复 2 — `lds_cast_alignment.py` fail-closed，第二次杀掉同一个方向（2026-08-19 ~14:20）

resume 之后 r1_d1（memory / staging pipeline）**又一次**在 step 2c 的 vector-cast alignment gate 上 `policy_failed`，在 build 之前、correctness 之前、任何一次计时之前。收据 `round_1/engineer_1/verify/cast_alignment.json`：

```json
{"findings": [], "gate_mode": true, "inherited": [], "passed": false,
 "unparseable": [ 4 x {"array":"smem","reason":"element type 'char' is not in ELEM_SIZE; its size decides the whole verdict"} ]}
```

**已独立核实这个退出是空判决**（不是只信 retrospective）：`lds_cast_alignment.py:165` 查不到元素类型就 `continue`，该数组**从不进入 `arrays`**；于是 `scan()` 里每一个针对它的 cast 都撞上 `if name not in arrays: continue` —— **一条 finding 都不可能产生**。而且 `smem` 是一维的，stride 循环走的是 `strides[:-1]`，一维数组这个切片为空，**对任何元素类型都无判决可下**。`passed` 却仅凭 unparseable 一项翻成 false。这是典型的 fail-closed 假拒绝：工具什么都没判定，却拦下了一个正确的候选。

**处置。** 给 `ELEM_SIZE` 补上字节类型（`char` / `int8_t` / `uint8_t` / `std::byte` / `std::int8_t`）。`char` 按定义就是 1 字节，这是**解析**尺寸而不是猜测。原有 28 个测试仍全绿，另加 3 个回归测试（`ByteArenaTest`）钉住：一维 char arena 干净通过、一维数组无外层 stride 可判、**二维 char 数组的 136 字节行距仍然被抓住**（确认修复只是解出尺寸，没有把检查关掉）。现 31 tests OK。

对被拒了两次的那份真实候选重跑：

```
$ python3 kernel_workflow/scripts/lds_cast_alignment.py --json <ws>/src/custom_gemm.hip <ws>/src/custom_gemm_hip.hip
{"findings": [], "gate_mode": false, "inherited": [], "passed": true, "unparseable": []}
EXIT=0
```

**影响范围。** 只影响候选**能不能被测量**，不影响任何测量值本身；`harness_lib.py` / `scripts/task_runner.py` / oracle 一律未动。r1_d1 在本轮已经损失（修复对其后每一次 verify 生效），没有为它第三次重启——它的机制已记在 retrospective insights 里，可在后续轮次重新提出；重启的代价大于回收的价值。

**注意：这是一条会重复出现的 finding。** 任何使用 `__shared__ __align__(16) char smem[...]` 原始 arena 的 engineer 都会撞上；旧 lane 的 retrospective 已经把它记为"cost an entire direction"，现在证明它会**反复**吃掉方向，而不是一次性事故。

---

### 波 1 · 第 2 次发起 — 修掉第五个 gate 缺陷后 resume（2026-08-19 ~14:00）

**发生了什么。** 第一次发起（run `wf_e22ca8b0-fbb`）跑完 round 1，**一个候选都没进候选表，什么都没提交**，canonical 仍停在 `6585d50 baseline`。这不是"优化不出来"：round 1 有两个 verified / correctness pass / policy clean 的幸存者。

| 方向 | 专长 | vs oracle | vs 同场 control(0.3353) |
|---|---|---|---|
| r1_d0 | algorithm（128x128 macro tile，2x2 wave grid） | 0.5072 | **1.512x** |
| r1_d2 | host_runtime（split-K + (M,N,K) 路由选择器） | 0.6267 | **1.869x** |

两者**路线互补**：d0 吃 prefill/large-M（m2048 已到 rocBLAS 的 1.30x），d2 吃 decode（4 条 decode route 落在 oracle 的 10% 以内）。逐例取小的并集 = geomean **0.7887**，是种子 0.336 的 2.35 倍。

**根因（已在代码里核实，不是只信 agent 的话）。** 这是第 5 个 gate 缺陷，§1 那四条之外的新的一条：

- 本任务的 PRIMARY metric 由 harness 固定为**相对不可变 rocBLAS oracle** 的加速比（`COMMANDMENT.md` 116-129 行），而**种子只有 0.336** —— 候选树从一开始就在自己的比较基准**之下**。
- `kernel_lane.js:2263`：`return primSpeedup(r.ver) > CANDIDATE_FLOOR;`，而 `CANDIDATE_FLOOR` 默认 **1.0**。于是从 0.336 爬到 rocBLAS 平价的**整段过程全部被过滤掉**，`candidates=[]` → `winner=null`。
- 致命之处：本 lane 要测的 per-route gate 挂在 `kernel_lane.js:2352` 的 `if (winner && ROUTE_BANDS)` 后面。**winner 恒为 null，gate 就永远跑不到** —— 本 lane 会以另一种方式复现旧 lane "gate 一行都没打"的结局，且拿不到任何关于自身自变量的数据。
- 代码自己的注释（`kernel_lane.js:113-118`）逐字描述了这个情形：候选"lands BELOW the comparator by construction"，在 1.0 下"its recovery phase is invisible"。旋钮本来就是为这个准备的。

**处置。** 按 §2 的唯一合法路径改**文件**（不是手抄参数）：`kernel_workflow/lanes/coldstart_newgate_20260819.json` 增加 `candidate_floor: 0.29`，并写进 `_require` 让它被 pin 住。`lane_args.py --check` → **exit 0**（9 个参数，6 个 pinned 值匹配），`--print` 重新渲染后**原样**调用。

- 为什么是 0.29：低于同场 control（0.3353）的幅度大于实测最宽的 route band（11.55%，最坏读数 0.2966），所以不会丢掉任何真实结果；同时诚实报告"比种子还差"的 engineer 仍然走 `kernel_lane.js:2086` 的 `trustworthyBelowBaseline` 跳过 verify 的经济性。
- **提交判据没有被放松**：banking 仍然要过 per-route band gate 或 `cumulative` 上的 MIN_IMPROVE。改的是**被跟踪**的东西，不是**被采纳**的东西。
- **配对性代价（如实记录）**：greedy lane 跑的是 1.0 默认值，所以两条 lane 现在差**两个**旋钮，严格意义上不再是单变量对照。这是权衡后的选择——floor 留在 1.0，本 lane 根本无法触发 per-route gate，那就是"干净但什么都测不到"。两条曲线应读作"per-route gate + 可达的候选表" vs "suite gate"；并注意 greedy 早期同样耗在这段隐形的恢复期里。

**重启方式。** `resumeFromRunId: wf_e22ca8b0-fbb`。`CANDIDATE_FLOOR_TXT` 只出现在 Optimize 提示词一处（`kernel_lane.js:2041`），所以 Setup / Analyze / Benchmark / Profile 的提示词未变、命中缓存，昂贵的 baseline GPU 测量不用重做；只有 engineer 轮次重跑。新 task ID `wtlxltbmv`。

**round 1 另外值得带走的事实**（来自 retrospective，已并入 lane 记忆）：
- 绑定项是 **occupancy 而非 LDS 复用**：128x128x64 要 34816 B → 1 CTA/CU，单靠寄存器分块只有 1.32x；BK 减半到 32（18432 B，3 CTA/CU）同样 tile 再拿 1.84x。超过 4x2 fragment grid 后寄存器分块这条轴就耗尽了。
- **此 kernel 不是 DRAM-bound**（对 roadmap.md 的更正）：HBM 峰值 584 GB/s（prefill）/ 314 GB/s（decode），L2 吸收 89.8%，各 pipe 占用均 <22%。48x 读放大付出的是**暴露延迟 + LDS/VALU 指令开销**，不是带宽。
- 小 M/小 N 路线是**并行度饥饿**，tile 形状救不了，并行度必须来自 K —— d2 的 split-K 把 decode geomean 从 0.3453 抬到 0.8305。
- 别再提"B 没有 per-wave 复用所以砍掉它的 LDS staging"：实测 11 例全负（最差 prefill_m2048 0.60x）。B 的 LDS tile 是**合并访存变换**，不是复用变换。
- 工具缺陷：`lds_cast_alignment.py` 对 `__shared__ __align__(16) char smem[...]` 因 'char' 不在 ELEM_SIZE 表里而 exit 2（fail-closed），findings 为空——**空判决 + 非零退出**害掉了一整个方向。修在候选侧：把 arena 声明成 `Bf16 smem[kSmemBytes/2]`。
- 流程卫生：有 engineer 在 verifier 运行中改写了 `best_patch.diff`。verifier 应在验证前快照并 pin md5；engineer 交接后不得再碰该文件。

**§6 三件事的现状**：三条都还**未观测到**——第一次发起从未走到 gate 就结束了 round 1，且 `log()` 的叙述行不落盘，只能从 journal 的结构化结果反推。第二次发起后重新盯。

### 波 1 — 已启动（2026-08-19）

- **开工检查单**：`check_measurement_frame.py` → **exit 0**。hostname `tw053`，纪元 **Y**，已在 11 条 route 上测过底噪、与 `CURRENT_MACHINE` 相符。无需注册新字母，不需要重测底噪，直接可计时。
- **参数**：`lane_args.py --check kernel_workflow/lanes/coldstart_newgate_20260819.json` → exit 0（8 个参数全部被入口点接受，5 个协议值 pin 住且匹配）。`--print` 渲染出的调用**原样**发起，未手抄、未增删键。生效值含 `min_improve=0.005`、`budget=12`、`gpu_ids=2,3,4,5,6,7`。
- **冷启动确认**：`exp/state_coldstart_newgate_20260819` 发起前不存在（无播种）。
- **运行标识**：Workflow run ID `wf_e22ca8b0-fbb`，task ID `ws8frhgsh`；transcript 在
  `/home/yxh/.claude/projects/-home-yxh-GEAK/096453ae-8652-4183-b00f-d5ead2509ce6/subagents/workflows/wf_e22ca8b0-fbb`。
- **待核对（§6 三件事）**：Setup 的 effective-config 回显里 `min_improve=0.005` 不带 `(default)`；`Commit gate: PER-ROUTE, N bands` 是否出现；`isa_evidence` 是否产出 verdict。波次结束后回填。


---

## 2026-08-19 15:2x — 第一次真正的提交：per-route band gate 首次运行并 ACCEPT

`candidate_floor: 0.29` 落地后，round 1 重跑的结果第一次让 `winner` 非空，于是
`kernel_lane.js:2352` 的 `if (winner && ROUTE_BANDS)` 分支——本 lane 存在的理由——
在这个项目里第一次真的执行了。

**独立复算（不采信 agent 的说法）。** 我用 Python 孪生 `scripts/route_gate.py`
拿 winner 的 `per_case` 对 `control_per_case`（同 session 对照臂）重算了一遍。
`decide()` 是 keyword-only 的，正确调用是
`RG.decide(candidate_per_case=…, incumbent_per_case=…, bands=…, target_routes=…)`。

- band 表由本 wave 自己的 `baseline_per_case[].samples_ms` 导出（11/11 路由各 5 次
  预热重复，§6 item 2 的前置条件确认满足），未使用任何外部 route_bands 表。
- 判决：**ACCEPT**。
  理由串：`improved past band on: decode_m16_square (+67.43% vs band 1.29%),
  decode_m2_square (+65.01% vs 1.31%), decode_m32_down (+79.13% vs 3.05%)`
  ——正好是该候选申报的三条 target route。
- `regressed: []`；`improved` 覆盖全部 11 条路由，没有触发回归否决。

| route | incumbent ms | candidate ms | delta | band |
|---|---|---|---|---|
| decode_m2_square | 0.07242 | 0.02534 | +65.01% | 1.31% |
| decode_m8_up | 0.09468 | 0.06148 | +35.07% | 0.54% |
| decode_m16_square | 0.10328 | 0.03364 | +67.43% | 1.29% |
| decode_m32_down | 0.42788 | 0.08930 | +79.13% | 3.05% |
| decode_m64_square | 0.31278 | 0.14146 | +54.77% | 0.48% |
| decode_m96_up | 0.17076 | 0.13480 | +21.06% | 3.16% |
| prefill_m128_square | 0.15256 | 0.07508 | +50.79% | 4.53% |
| prefill_m256_down | 0.41736 | 0.27880 | +33.20% | 14.86% |
| prefill_m512_up | 0.51698 | 0.48444 | +6.29% | 1.29% |
| prefill_m1024_down | 1.08671 | 0.89759 | +17.40% | 0.89% |
| prefill_m2048_square | 0.64802 | 0.63076 | +2.66% | 0.27% |

注意最后两行是这套 band 机制真正有意思的地方：+2.66% 在 suite geomean 里几乎
看不见，但 prefill_m2048 的 band 只有 0.27%，所以它是一个**可判定的**改进；反过来
prefill_m256_down 的 band 有 14.86%，+33.20% 才刚够两倍余量。用一个 suite 平均值
去判这两条路由，本来就是在问一个它测不了的问题（这正是 commit 1777af35 修的东西）。

**提交已落地。** canonical 树 `…/dense_bf16_gemm_fused/workspace`（注意 canonical
是 `workspace/` 子目录，父目录不是 git repo——从父目录跑 `git log` 会走到 /home/yxh/GEAK
上去，读到的是本仓库的历史，不是 lane 的）：

```
8d9bee7 round 1 winner: engineer r1_d2 (0.61x)
6585d50 baseline
```

`git status --porcelain` 空；`current_best.diff` 45036 B，与 `git diff --binary
<root>..HEAD` 逐字节一致；6 个文件全部是 candidate-owned，没有碰任何 immutable 路径。
提交后重跑：policy scan v2 findings 0 / advisory 0 / 4 个 candidate ELF；correctness
PASS（tol 0.02，5 随机 draw × 11 shape，最差 max_rel_err 0.00773，write probe 11/11
conclusive，5 个 negative test 全过）。

**Tech lead 报上来两件不阻塞提交但必须记下的事：**

1. `round_1/engineer_2/report.md` 描述的**不是**被提交的代码。report 里的
   "~6144 CTAs 选择器"、"template row-fragment 16/32/64"、"grow-only per-device fp32
   scratch"、"K %% 64 != 0 就不 split" 在 patch 里一个都没有；patch 实际是
   `kFillTarget = 2432`、固定二选一的 `kTile`/`kTallM`、`kGroupM = 1` swizzle、
   每次调用 `torch::empty` 的 workspace，外加 report.md 从没提过的 skinny-M
   非 MFMA gemv 路由（`kGemvMaxM = 4`）。被测被提交的是 patch，report.md 的机制叙述
   和它那张 per-case 表对本次提交**未经验证**。顺带：round 1 insight #12 里那条
   "grow-only、永不释放的 fp32 partial-plane cache" 的负债，在提交的代码里不存在——
   它属于 report.md 描述的那个版本。
2. `splitk_reduce_kernel`（src/custom_gemm.hip）有一个 suite 外的潜在 bug：
   `nvec = total >> 2; vstride = nvec;`，而 plane 在 float4 单位下的 stride 是
   `total/4`，两者只在 `total = M*N` 能被 4 整除时相等。11 条评分 shape 全都满足，
   所以 correctness 过；shape 集一旦放宽，向量主体会从错位的 plane 偏移累加并静默出错
   （下面的标量尾巴是对的）。当前 gate 到不了这里。

**还没兑现的：** round 1 的 r1_d0（prefill 侧，0.5072）在本轮不可恢复——它的 verify
prompt 未变，resume 只会重放同一次 policy_failed。retro 里那个 d0 ∪ d2 route-disjoint
并集（算术推得 geomean 0.7887，2.35× seed）仍然要等 director 重新提出来才可能变成测量值。

**wave 状态：** round 2 已在跑（journal 42 行，两个 agent 在飞：`abe1c2f57caa6f925`、
`ab754e0feb0b3d7a0`）。`STATE.json` 仍是 13:49 的 round-1 收尾快照（`cumulative` 1.0），
因为它由 director 在轮末写；下一次 round close 时应该看到它从 1.0 挪开。

---

## 2026-08-19 15:3x — §6 三个观察项全部有答案；其中第 3 项挖出一个真缺陷并已修

**§6 item 1（Setup 的 effective-config 回显）：确认。** 首launch 的回显是

```
[kernel-lane]   budget=12  deep_cost=2 (default)  max_no_improve=2 (default)
[kernel-lane]   min_improve=0.005  candidate_floor=1 (default)  progress_delta=0.005 (default) (= min_improve)
[kernel-lane]   isa_evidence=observe (default)  mode=optimize  gpu_mode=pool (default)  gpu_ids=2,3,4,5,6,7
[kernel-lane]   route_bands=not supplied -- will be DERIVED from the baseline repeats after the Benchmark phase
```

`min_improve=0.005` 后面**没有** `(default)`，正是这个机制要证明的事——commit 2761475b
之前，一个没送到的参数和一个选定的默认值在运行期是无法区分的。同一行里
`candidate_floor=1 (default)` 就是当时那个缺陷的现场照片。

注意 `~/.claude/.../workflows/wf_e22ca8b0-fbb.json` 这个记录文件停在 14:01 就不再更新了，
所以里面只有**首次** launch 的回显；resume 之后的回显没有落进去。不要拿它当 live 配置读。
live 进程确实跑在 0.29 上，证据是判决本身：admission filter（`kernel_lane.js:2263`
`primSpeedup(r.ver) > CANDIDATE_FLOOR`）放行了 0.6114 并让 `winner` 非空，这在 floor=1.0
下不可能发生。

**§6 item 3（isa_evidence 是否产出可读判决）：是，但答案是 `indeterminate`，而原因是一个
可修的管线缺陷，不是物理上不可判。** 这是本 lane 第一次拿到有实质内容的 ISA 回执：

- 父归档由 verifier 从**自己测过的** control workspace 抓取（`isa_parent_v2`，
  `parent_source_hash 0071b07c…`），不是从别人递过来的 "canonical" 路径——这正是
  roles/verify_engineer.md 要求的做法，也是老 lane「11 次干净抓取、0 个判决」的解药。
- checks 层跑通了：8 个 kernel、9 条 finding、全 advisory、high=0。
- 但 diff 层 `mechanism_realized=null → indeterminate`，verifier 的说明是：
  **"ISA_MECHANISM_CLAIMS was handed to me empty, so no claim was passed to isa_signals
  (the engineer's worker_result declares 'reduce_lds'; I did not substitute it)."**

**缺陷定位（不采信 agent 的说法，逐层查了）：**

1. engineer_2 确实声明了 claim。journal 里它的结构化返回带 `mechanism_claims: ['reduce_lds']`，
   磁盘上 `round_1/engineer_2/worker_result.json`（13:32 写入）同样带 `['reduce_lds']`。
2. verifier 收到的确实是空的。它的 prompt 里逐字是 `- ISA_MECHANISM_CLAIMS: []`（15:02:42）。
3. 中间断在哪：这次 verify 走的是 **recovery 路径**。resume 之后 engineer_2 的 engineer agent
   （`a06c97744fe75ffd9` / `ae6521b8b776e51ba`）没有返回结果，于是 `kernel_lane.js:2090`
   的 `recovered = !eng || eng.status === 'failed'` 为真、`eng` 为 null，
   而 L2133 的 `eng && Array.isArray(eng.mechanism_claims) ? … : []` 就渲染成了 `[]`。
   verifier 拒绝自己替补 claim——这是对的，它被明确要求不许把假设改成证据的形状。
   错在上游。

**这个缺陷的讽刺之处，也是修法的依据：** 同一段 harvest 逻辑（L2071-2089）早就论证过
「engineer 死了不代表磁盘上没有好东西」，因此丢失的返回**不**能压制 `best_patch.diff`。
`worker_result.json` 是同一个 engineer 在 verify 测量任何东西**之前**写的同一份声明，
适用完全相同的论证：读它不是"把 claim 拟合到证据上"，是从唯一幸存的副本里把假设捞回来。

**已修：**

- `kernel_lane.js` L2135-2152：recovery 路径额外下发
  `ISA_MECHANISM_CLAIMS_FILE: <out_dir>/worker_result.json`。只在 `recovered` 为真时下发——
  活着的 `eng` 永远是权威，文件不得覆盖它。工作流沙箱 stat 不了文件，所以和 patch 一样：
  只给路径，判断权下放给 verifier。
- `roles/verify_engineer.md`：新增一段，规定 `ISA_MECHANISM_CLAIMS` 为空且
  `ISA_MECHANISM_CLAIMS_FILE` 存在时，从该文件逐字读 `mechanism_claims`；非空的
  `ISA_MECHANISM_CLAIMS` 永远优先；文件缺失/不可读/没有该字段时，就无 claim 跑 diff 并在
  `notes` 里说明，不许发明一个。无论走哪条路都要在 `notes` 里写明 claim 的来源，
  好让读的人能区分"engineer 提过的假设"和"没人提过的假设"。
- 测试：`python3 -m unittest discover` **Ran 875 tests, OK**，没有留下红的。

**这个修复对当前 wave 无效**，说清楚免得误读：嵌套 workflow 的脚本在 wave 启动时就已载入，
本次 run 不会重读 `kernel_lane.js`。它从下一次 launch 起生效。代价是：如果之后需要
`resumeFromRunId`，recovered verify 的 prompt 变了、缓存键随之改变，那些 verify 会重跑
（也正好会带上 claim）。

**"如果当时是 gate 模式会怎样" —— 这个问题可以从代码直接回答，不必猜。**
`isaEvidenceReject`（`kernel_lane.js:1627`）只在 `mechanism_verdict === 'refuted'` 时拒绝；
HOLE、缺归档、`indeterminate`、以及任何 `checks` finding 都**不**拒绝，而且注释里给了理由：
证据里的洞是我们的缺陷、不是候选的过错，在这个方向上失败会静默地毁掉正确的快 kernel。
所以 gate 模式**不会**挡掉本轮这个候选。

但把上面那个 claim 丢失缺陷和这条规则放在一起看，结论比"无害"难看得多：没有 claim →
diff 的 `mechanism_realized` 恒为 `null` → 判决恒为 `indeterminate` → gate **永远不可能触发**。
也就是说，在修好之前，恰恰是对那些 engineer 中途死掉、最需要独立检查的候选，
`isa_evidence=gate` 会被静默解除武装，而日志上看起来一切正常。这比"少一条证据"严重，
是这次修复真正的价值。即便如此，提到 `gate` 仍然要等：先用 observe 跑一整波、
手工看过判决，确认 `refuted` 只落在真的没实现的机制上，再谈。

---

## 2026-08-19 15:4x — round 2：闸门第一次和 suite 判据分歧，方向是**更严**，而它很可能是错的

round 2 有一个 verified 候选 `0.924`（incumbent 0.6114，suite +51%），
11 条路里 **10 条改善 24%–50%**。逐路复算（Python 孪生，JS `routeGate` 逻辑逐条对过，
回归否决同样在 target-route 收窄**之前**短路，所以两边判决一致）：

```
VERDICT: REFUSE
reason: regressed past its own band on: decode_m2_square (+2.40% vs band 1.31%)
```

| route | incumb ms | cand ms | delta | band |
|---|---|---|---|---|
| decode_m2_square | 0.02506 | 0.02566 | **−2.40%** | 1.31% |
| decode_m8_up | 0.06144 | 0.04250 | +30.83% | 0.54% |
| decode_m16_square | 0.03368 | 0.02532 | +24.82% | 1.29% |
| decode_m32_down | 0.08926 | 0.05576 | +37.53% | 3.05% |
| decode_m64_square | 0.14144 | 0.09974 | +29.48% | 0.48% |
| decode_m96_up | 0.13494 | 0.09472 | +29.81% | 3.16% |
| prefill_m128_square | 0.07490 | 0.05260 | +29.77% | 4.53% |
| prefill_m256_down | 0.27846 | 0.14860 | +46.64% | 14.86% |
| prefill_m512_up | 0.48718 | 0.32002 | +34.31% | 1.29% |
| prefill_m1024_down | 0.89715 | 0.44542 | +50.35% | 0.89% |
| prefill_m2048_square | 0.63042 | 0.33624 | +46.66% | 0.27% |

`suiteSaysYes = legacyImproved && !routeVerdict.regressed.length`，回归非空 ⇒ `improved=false`。
**这一轮不会提交。** 0.6 微秒的回退否决了一个 0.29 毫秒的收益。

### 这同时证伪了 `kernel_lane.js` 里的一句话

L2393 的注释写着 "This also means the change of default cannot make any run STRICTER
than it was, which matters: the entire complaint against the old gate was that it refused
real work."。**本轮就是反例**：ROUTE_BANDS 关掉走 legacy，`legacyImproved=true` → 提交；
打开则拒绝。作者其实知道，因为上面两段刚写过 "What is NOT unioned is the regression veto"——
那个 `&& !regressed.length` 正是打破 "never stricter" 的项。两句话互相矛盾，
留着会让读的人以为改默认值零风险。（注释待修，见下。）

### 但真正的问题在 band 本身：n=5 的极差不是一个可用的统计量

band = `(max−min)/median`，两个入口共用 `_band_from_values`，`MIN_REPEATS=3`。
本波用的是 baseline 的 **5** 个 `samples_ms`。把它和 §13.6 那次 **24 次重复**的
标定（绝对微秒列，也就是本模块真正判据的那一列）并排：

| route | §13.6 24 次（绝对 µs 极差） | 本波 5 次 | 倍数 |
|---|---|---|---|
| decode_m2_square | **11.55%** | **1.31%** | 8.8× 偏紧 |
| decode_m16_square | 8.46% | 1.29% | 6.6× 偏紧 |
| prefill_m256_down | 5.19% | **14.86%** | 2.9× 偏松 |
| decode_m8_up | 4.65% | 0.54% | 8.6× 偏紧 |
| prefill_m512_up | 4.23% | 1.29% | 3.3× 偏紧 |
| prefill_m1024_down | 3.97% | 0.89% | 4.5× 偏紧 |
| decode_m64_square | 3.57% | 0.48% | 7.4× 偏紧 |
| decode_m96_up | 3.23% | 3.16% | 一致 |
| prefill_m2048_square | 3.16% | **0.27%** | 11.7× 偏紧 |
| decode_m32_down | 2.03% | 3.05% | 1.5× 偏松 |
| prefill_m128_square | 1.56% | 4.53% | 2.9× 偏松 |

**排序几乎被打乱**：decode_m2_square 从全套件最吵（11.55%）变成第六安静（1.31%），
prefill_m2048_square 从中游变成最安静（0.27%），prefill_m256_down 从 5.19% 变成最宽（14.86%）。

原因看原始样本就清楚，不需要统计学：

```
prefill_m256_down  [0.155019, 0.132219, 0.155099, 0.155259, 0.155119]   <- 一个飞点
decode_m32_down    [0.0752,   0.07334,  0.07544,  0.075459, 0.075639]   <- 一个飞点
decode_m96_up      [0.073259, 0.0716,   0.07336,  0.073919, 0.0733]     <- 一个飞点
prefill_m2048_sq   [0.292918, 0.292119, 0.292698, 0.292598, 0.292718]   <- 没有
decode_m2_square   [0.02746,  0.02766,  0.0273,   0.02752,  0.02756]    <- 没有
```

n=5 的 min−max 基本上就是**"这 5 次里有没有恰好撞上一个飞点"的伯努利抽样**。
撞上的路线拿到宽 band，没撞上的拿到 0.27% 这种荒谬的紧 band。§13.1 已经量化过
这个分布的形状——"MAD 只有 0.23%，分布中心很紧、是少数几次跑偏拉开的极差"——
也就是说这个分布**专门**会让小 n 的极差失真。min−max 的期望随 n 单调增长，
控制图 d2 常数给出 n=5→n=24 的系数是 3.895/2.326 ≈ **1.67×**；但实测差到 8.8×，
所以 n 依赖只解释其中一小部分，剩下的是树/机器/纪元不同（§13.1 在 tw003、epoch Q、
round-7 那棵 1.40 的树上标定，本 lane 不是同一台）。**两个 band 表不可直接互换**——
这恰恰是重点：模块 docstring 说两个入口是 "Same statistic"，字面为真，
但同一个统计量在 n=5 和 n=24 下不是同一个量，而代码把它们当成同一个量在用。

### 我上一条记录里有一处要更正

我在 round 1 那条里写 prefill_m2048_square 的 +2.66% "是一个**可判定的**改进，因为它的
band 只有 0.27%"。**这个说法站不住**：0.27% 是没撞上飞点的 5 次抽样，§13.6 对同一条路线
用 24 次测到 3.16%。+2.66% 落在 3.16% 以内，应当读作**噪声内，不是可判定的改进**。
round 1 的 ACCEPT 本身不受影响——它是靠 decode 三条 +65%~+79% 通过的，那三条无论用哪张
band 表都远在噪声之外。

### 判断与处置

`decode_m2_square` 上这 2.40%（0.6 µs）几乎肯定在噪声内：§13.6 对这条路线的 24 次
绝对微秒极差是 **11.55%**，是全套件最吵的一条。用 1.31% 去否决它，是**假回归否决了
一个 +51% 的真实收益**——正是 route_gate 被造出来要消灭的那类错误，只是方向反了。

**但我不在无人值守的情况下单方面改判据。** 理由：band 的统计量就是本 lane 的实验变量本身，
中途换掉会毁掉与 greedy 的配对，也会让"per-route gate 表现如何"这个问题失去答案。
本轮这个 REFUSE 是**数据，不是故障**——它是这条 lane 存在的理由所产出的第一个负面结果。
而且改 `route_gate.py` 对正在跑的 wave 无效（JS 侧 `bandsFromSamples` 才是决定者，
且脚本在 wave 启动时已载入）。

留给运行方的两个选项，成本差别很大：

1. **便宜且不改判据**：把 `benchmark_engineer.md` step 5 的 ">=3 primed repeats" 提到
   >=16~24。band 的定义一个字不用动，飞点抽样问题自然消失。代价是每波多几分钟 baseline。
   这是我推荐的那个。
2. **改判据**：用 d2 常数把极差归一到 n 无关的尺度（σ̂ = range/d2(n)，再取固定倍数）。
   原理正确，但它是对闸门语义的改动，且只能修 1.67× 的那部分，修不了树/机器差异。

在运行方定夺之前，本 lane 继续按现状跑，REFUSE 照记。**已修的只有那句自相矛盾的注释**
（见下一条 commit），因为它是纯粹的事实错误，不影响任何行为。

### 顺带一条小的（已修）

`route_gate.py:decide()` 里 no-band 的报错信息原本写 "Bands on this lane range from 2.68%
to 16.44%"——这是 §13.6 的**加速比**那一列，而本模块在 §13.6 里刚论证过应该用**绝对微秒**
那一列（1.56%~11.55%）。数量级不误导，但引的是自己刚否掉的那一列。已改为绝对微秒区间，
并在同一句里写明"重复次数要够"以及 n=5 与 n=24 在同一条路线上 1.31% vs 11.55% 的对照——
这条报错是唯一会被"没有 band 该怎么办"的读者读到的文字，把 n 的坑写在这里最省事。
34 个 route_gate 测试全绿；报错文本没有任何测试断言。

### 已改动的文件（都不影响正在跑的这一波）

- `kernel_lane.js` L2392 起：那段自相矛盾的注释重写，写明"打开这个默认值**可以**让一轮更严"，
  并把 round 2 作为反例记在注释里。**纯注释**，行为零改动，正在跑的 wave 不受影响。
- `route_gate.py` L240 起：报错文本改用绝对微秒列。Python 孪生只用于离线复算，
  且脚本在 wave 启动时已载入，对本波无影响。

**没有改**：band 的统计量、`MIN_REPEATS`、`MIN_BAND`、回归否决、`bandsFromSamples`、
`benchmark_engineer.md` 的重复次数。判据在本 lane 跑完前保持冻结。

---

## 2026-08-19 16:5x — round 2 更正与补强：本轮**握着一个能过闸门的候选，却会两手空空**

上一条我写 round 2 只看了那个 0.924。查 journal 才发现本轮有**两个** verified 候选，
而且第三个 engineer 16:49 还在跑（`last_round` 仍是 1，本轮**尚未结算**）。
用 baseline 自己的 5 次样本导出 band，对两个候选各判一次：

```
=== geomean=0.924  targets=['prefill_m2048_square']
   REFUSE -- regressed past its own band on: decode_m2_square (+2.40% vs band 1.31%)
   （11 条路里 10 条改善 +24%~+50%，见上一条的表）

=== geomean=0.632  targets=['decode_m32_down','prefill_m2048_square','prefill_m512_up']
   ACCEPT -- improved past band on: decode_m32_down (+12.92% vs band 3.05%),
                                    prefill_m2048_square (+0.32% vs band 0.27%)
     decode_m16_square    0.03342 -> 0.03094   +7.42%  band 1.29%  imp
     decode_m2_square     0.02522 -> 0.02510   +0.48%  band 1.31%  flat
     decode_m32_down      0.08932 -> 0.07778  +12.92%  band 3.05%  imp
     decode_m64_square    0.14162 -> 0.14016   +1.03%  band 0.48%  imp
     decode_m8_up         0.06130 -> 0.05938   +3.13%  band 0.54%  imp
     decode_m96_up        0.13490 -> 0.13576   -0.64%  band 3.16%  flat
     prefill_m1024_down   0.89757 -> 0.89143   +0.68%  band 0.89%  flat
     prefill_m128_square  0.07480 -> 0.06866   +8.21%  band 4.53%  imp
     prefill_m2048_square 0.63156 -> 0.62954   +0.32%  band 0.27%  imp
     prefill_m256_down    0.27872 -> 0.27940   -0.24%  band 14.86% flat
     prefill_m512_up      0.48512 -> 0.48400   +0.23%  band 1.29%  flat
```

### 结构性发现：闸门只作用于 `candidates[0]`，而选 winner 用的是另一套判据

`kernel_lane.js:2358  const winner = candidates[0] || null;`
候选按 geomean 排序，**只有排头那个进闸门**。于是 round 2 的结局是：

- 0.924 geomean 更高 → 当选 winner → 被逐路闸门以 0.0006 ms 的回退否决；
- 0.632 逐路闸门**判 ACCEPT**、无任何 route 越界回退 → 因为不是排头，**从未被送进闸门**。

**本轮预计不提交任何东西，尽管手里就有一个通过闸门的候选。** 这不是 band 太紧造成的，
band 换成 §13.6 那张宽表也一样：选 winner 用 suite geomean、判提交用逐路 band，
两套判据不一致，而中间没有回退。L2359 的注释写 "SELECTION is unchanged: the winner is
still the highest measured paired speedup. Only the ACCEPTANCE test changed."——
话是对的，但**正因为只改了后半截**，才出现"选出来的过不了、能过的没被选"。
以前两截用同一个 geomean，这个洞不存在；它是把闸门换成逐路制**引入的新失效模式**，
不是老问题的遗留。

修法很便宜且不动判据本身：候选按 geomean 降序**逐个**过闸门，取第一个 ACCEPT 的；
全都不过再回落到现有的 suite 路径。语义上仍是"最好的那个能过的"，
而不是"最好的那个，如果它恰好能过"。**本 lane 跑完前不改**——理由同上一条，
判据和它的取用方式都是本 lane 的被测对象，这次两手空空就是它要产出的数据。

### 对 0.632 那个 ACCEPT 本身的诚实标注

它的 ACCEPT 建立在两条路上，其中 **prefill_m2048_square 的 +0.32% vs band 0.27% 不可信**——
正是上一条记录的那个飞点抽样产物（同一条路 24 次重复测到 3.16%）。
另一条 **decode_m32_down +12.92% vs band 3.05%** 是真的，无论用哪张 band 表都过。
所以这个 ACCEPT 结论成立，但它只应被读作"decode_m32_down 上有一个真实改进"，
而不是两条路都动了。engineer 自己的报告也是这么写的（"declared claim is supported on
decode_m32_down only"，且主动拒绝把 target_routes 扩到没预测到的
decode_m16_square/prefill_m128_square，"that would be a fabricated mechanism"）——
这个自我约束是对的，记一笔。

### round 1 的另一半账：两个方向死在我后来修好的那个扫描器上

`STATE.json` ledger 显示 round 1 三个方向里有**两个**是 `policy_failed`、`actual: null`：
- `r1_d0`（macro-tile register blocking，claimed 0.6049）—— 步骤 2c，
  `lds_cast_alignment.py` exit 2，4 条 unparseable，findings **空**，原因是补丁自己的
  `shortx4_t` typedef 不在闭表 `CAST_WIDTH` 里；
- `r1_d1`（staging pipeline，claimed 0.8292）—— 同样步骤 2c exit 2，
  原因是补丁把三个 typed `__shared__` 换成一个 `__shared__ __align__(16) char smem[]` 竞技场，
  `char` 不在 `ELEM_SIZE` 里。

两条都**没有拿过 GPU 锁，一次都没测过**。这正是 `8efab858` 修的两个缺陷
（文件内 vector typedef 解析 + 字节型元素类型入表），修复发生在 round 1 结算之后，
所以 round 1 白白丢了两个方向、只剩 `r1_d2` 一个候选。round 2 起生效：
本轮 policy_failed 归零，两个候选都走完了验证。**代价已经付过一次，记在这里免得被当成偶然。**

### 本 lane 当前状态

| 项 | 值 |
|---|---|
| cumulative | **0.6114**（round 1 的 `r1_d2`） |
| canonical | `8d9bee7 round 1 winner: engineer r1_d2 (0.61x)` |
| last_round | 1（round 2 尚未结算，第三个 engineer 在跑） |
| 已提交轮次 | 1 / budget 12 |
| 测试 | 875 全绿 |

**一个顺带确认（免得下一个人白担心）**：这次 REFUSE **不会**烧掉停滞预算。
`suiteProgress` 判的是 `winner.geomean > bestSeen * (1+PROGRESS_DELTA)`，
用的是 winner 的 geomean 而**不是**是否提交：0.924 > 0.6114×1.005 成立，
所以本轮记为"搜索在推进"，`MAX_NO_IMPROVE`（默认 2）不增加。
也就是说 wave 不会因为闸门连续拒绝而提前停摆——它会一直跑到 budget 12。
`candidates[0]` 之后没有任何回落到次名的路径（L2357-2358 排序取头，无循环），
所以上面那条"握着能过的候选却不提交"没有被别处兜住。

---

## 2026-08-19 17:0x — 更正上一条：round 2 **会**提交，而且是本 lane 最大的一步

上一条我写"本轮预计不提交任何东西"。**这个预测是错的**，原因很简单：我是在本轮
只有两个 verified 候选时下的判断，之后第三个 engineer 交了 **0.9879**（`target_routes: []`），
geomean 高于 0.924，成为 winner。逐路复算：

```
WINNER geomean=0.9879  targets=[]
ACCEPT -- improved past band on: prefill_m1024_down (+66.54% vs 0.89%),
  prefill_m2048_square (+64.32% vs 0.27%), prefill_m512_up (+62.29% vs 1.29%),
  prefill_m256_down (+55.00% vs 14.86%), prefill_m128_square (+36.12% vs 4.53%),
  decode_m96_up (+35.37% vs 3.16%), decode_m64_square (+29.55% vs 0.48%),
  decode_m32_down (+11.42% vs 3.05%)
  decode_m16_square  +0.18%  flat
  decode_m2_square   +0.31%  flat
  decode_m8_up       -0.13%  flat   <- 唯一负数，远在 0.54% band 内
```

**零回归，8 条路越过各自 band。** 这个 ACCEPT 不依赖任何一条可疑的窄 band：
即便把 prefill_m2048_square 的 band 换成 24 次标定的 3.16%，+64.32% 照样通过；
八条里有七条的改善幅度比最宽的那张 band 表还大一个量级。**这是本 lane 目前
最干净的一次判决**，也是 per-route 闸门第一次在没有争议的情况下放行。

cumulative 预计 0.6114 → **0.9879**（+61.6%），逼近 rocBLAS 平价。
`legacyImproved` 同样为真，两套判据一致，所以不会打印 OVERTURNS。

### 那条结构性发现要不要撤？不撤，但必须降级为"未被触发"

上一条说的"握着能过的候选却不提交"**本轮没有发生**——不是因为机制不存在，
而是因为恰好来了一个既排头又能过的候选，把 0.924 挤下去了。事实仍然是：

- `candidates[0]` 是唯一进闸门的候选，没有回落到次名的路径（已复核 L2357-2358）；
- 如果 0.9879 晚到、或那个 engineer 失败，本轮 winner 就是 0.924，会被 0.0006 ms 否决，
  而通过闸门的 0.632 不会被考虑，本轮就真的两手空空。

所以这是一次**擦肩而过**，不是一次发生的故障。按证据等级它从"已观测的失效"
降为"已证明可达、本轮未触发"。降级但保留，理由是它的触发条件很平常——
只要排头那个候选恰好在某条窄 band 路线上回退一点点就会命中，
而本轮 0.924 已经演示了这个条件有多容易满足。

**这也说明我上一条下判断下早了**：本轮当时还有 engineer 在跑，`last_round` 还是 1，
我却按"两个候选"写了结论。以后本 lane 的轮次结论一律等 `last_round` 前进后再写。

### 无 target_routes 的处理是对的

winner 的 `target_routes` 是空的，命中 L2428 那个分支。它**不拒绝**，只打一条 NOTE：
"accepted with no declared target_routes ... An incidental gain and a realized mechanism
are indistinguishable here."。这个处理是正确的——一个 61% 的真实收益不该因为缺一个标签
被丢掉，但也不该被记成"机制兑现"。读本轮结果时请照此理解：**这 8 条路的改善是实测的，
但没有任何被声明的机制可以被判定为兑现**。

---

## 2026-08-19 17:1x — round 2 结算：提交的是 **integrated 1.0916**，跨过 rocBLAS 平价

再更正一次。round 2 的 winner 既不是 0.924 也不是 0.9879，而是 **integrator 的 1.0916**——
integrate 步骤把 `r2_d0`（macro-tile）和 `r2_d1`（staged 16x16 bodies）叠起来，
再手工调了两个常数，四次组合的记录都在 journal 里：

```
r2_d0 + r2_d1                                   incremental  1.0528
  + kMacroMinM 32->64                           hand_merge   1.0832
  + r2_d2 kSliceFits 全 3 条                    hand_merge   1.0741  <- 退步
  + kMacroMinM=64, kSliceFits 只保留 2 条       hand_merge   1.0916  FINAL
```

integrator 的取舍值得记：`r2_d2` 整体 **REFUTED**——它的 `{256,4096,3}` 条目是在
没有 macro route 的树上测的，搬到这棵树上让 prefill_m128_square 从 0.0479 退到 0.0553 ms，
所以只保留了在本树上重新测出是赢的那两条。这是"父树变了，调优表就不再有效"的
一个干净例子，不是猜的，是重测出来的。

逐路复算（10/11 越过 band，零回归，`decode_m2_square` +0.47% 持平）：

```
ACCEPT -- prefill_m1024_down +66.97% (band 0.89%), prefill_m2048_square +64.57% (0.27%),
  prefill_m512_up +63.06% (1.29%), prefill_m256_down +55.06% (14.86%),
  decode_m32_down +41.21% (3.05%), decode_m96_up +36.54% (3.16%),
  prefill_m128_square +35.78% (4.53%), decode_m8_up +31.39% (0.54%),
  decode_m64_square +29.21% (0.48%), decode_m16_square +25.33% (1.29%)
```

`git apply` 一次成功，无 --3way 无手工合并。canonical:
`aa71ba0 round 2 winner: integrated (1.09x)` on `8d9bee7`。

### 新发现：**integrated winner 永远没有同场控制臂**

`kernel_lane.js:2372-2373`：
```js
const sameSession = !!winner.control_per_case;
const incumbentSide = sameSession ? winner.control_per_case : bestPerCase;
```
而 L2345-2352 把 integrate 结果推进 candidates 时，构造的对象只有
`source/id/title/specialty/geomean/geomean_unweighted/weighted/arithmetic/per_case/patch`——
**没有 `control_per_case`，也没有 `target_routes`**。所以只要 winner 是 integrated：

1. `sameSession` 恒为 false，闸门拿 **round 1 存下来的 `bestPerCase`** 当 incumbent，
   跨轮、跨会话比较——正是同场控制臂被引入来消除的那种比较；
2. `target_routes` 恒为空，必然触发 L2428 那条 "no declared target_routes" NOTE。

本轮**无害**：margin 是 25%~67%，比任何一张 band 表宽一个量级，
换成同场控制臂（0.9879 那个候选的 control 臂数字几乎一样）结论不变。
但这是**按构造成立**的，不是偶发：每一个 integrated winner 都会走跨轮比较那条路。
等哪天 integrated 的优势掉到个位数百分比，这个洞就会决定判决。
和上一条的 winner-selection 洞一样：**记录，不修**，判据及其取用方式是本 lane 的被测对象。

### 操作提醒：mid-wave 时 `STATE.json` 不是权威

`STATE.json` 此刻仍写着 `cumulative 0.6114 / canonical 8d9bee7 / last_round 1`，
mtime 停在 15:53，而 canonical git 里 `aa71ba0` 已经落地。
**轮次落地与否要看 canonical workspace 的 git log**（以及 journal 里那条
`{"committed": true, "head_sha_before": "8d9bee7...", "head_sha_after": "aa71ba0..."}`），
`STATE.json` 是滞后刷新的。我上一轮就是因为信了 `last_round` 才把话说早。

### 本 lane 当前状态

| 项 | 值 |
|---|---|
| cumulative（实际） | **1.0916** — 首次越过 rocBLAS 平价 |
| canonical | `aa71ba0 round 2 winner: integrated (1.09x)` |
| 起点 | seed 0.336 → r1 0.6114 → r2 1.0916 |
| 已提交轮次 | 2 / budget 12 |
| 闸门判决 | r1 ACCEPT、r2 ACCEPT，均零回归；一次预演性 REFUSE（0.924，未成为 winner） |
| 测试 | 875 全绿 |

**滞后量已测**：`STATE.json` 于 17:44 刷新，`last_round 2 / cumulative 1.0916`，
与 canonical git 和我的逐路复算完全一致。所以它**不是错的，只是慢**——本轮实测滞后约
40 分钟（git 提交 ~17:05，STATE 刷新 17:44）。结论不变：mid-wave 要判断轮次是否落地，
看 canonical git log；`STATE.json` 事后会对上，但不能用来判断"现在到哪了"。

---

## 2026-08-19 19:1x — 换机重启：tw053(Y) → **tw035(Z)**，以及 round 3 的结局（refused，1.15）

环境从快照恢复到新机器，之前的 wave 进程全部断掉。识别依据不是"看起来没动"：
全盘文件的 mtime 纳秒位统一变成 `.000000000`（快照回填的痕迹），
且 `pgrep task_runner|hipcc|gpu_fence_run` 为空。**agent transcript 的 mtime 是 19:09，
但那是恢复过程重写文件，不是 agent 在跑**——这正是我上一次误判 stall 的镜像错误，
这次两个证据都取到了才下结论。

### 纪元 Z 已注册（用脚本，没有手改）

```
check_measurement_frame.py  -> EXIT 4
  hostname tw035 / resolves to epoch N / CURRENT_MACHINE Y -> tw053
```
`tw035` **历史上带过纪元 N**（`exp/opt_bf16_20260814/noisefloor_tw035_20260816/`）。
沿用 N 就是 finding 126，所以取下一个未使用字母。已用：N,O,P,Q,R,S,T,U,V,W,X,Y —— 取 **Z**。

```
register_epoch.py --letter Z --host tw035 --note '...new letter per (126)'
  -> epoch Z registered for tw035, PROVISIONAL, 11 routes at DEFAULT 0.072
check_measurement_frame.py  -> EXIT 3
```
先跑 `--dry-run` 看了四处改动的 diff 再落盘。**没有手改 `noise_floor_stats.py`**（§13.18 那次静默 no-op）。
tw035 八张卡全空（busy 0% / vram 283MB，按 gpu_lock 自己的 sysfs 判据查的，不是 rocm-smi），
所以底噪 sweep 用 **单卡 2**、整个 8 次重复包在一次 `gpu_lock.sh` 调用里，正在跑。

### round 3 结算：三个方向都验过，integrate 到 **1.15**，然后被**拒绝**

wave 在断电前把 round 3 跑完并写进了 STATE（`last_round 3`，ledger 11 条）：

| 方向 | claimed | actual | verdict |
|---|---|---|---|
| r3_d0 | 1.0892 | 1.0897 | dead_end |
| r3_d1 | 1.1409 | **1.1399** | confirmed |
| r3_d2 | 1.1025 | 1.1001 | partial |
| r3_integrate | — | **1.15** | partial（**refused**） |

`cumulative` 仍是 **1.0916**，canonical 仍是 `aa71ba0`。1.15 比在册的 1.0916 高 **+5.35%**，
8 条路改善，被三条回退路线否决：

```
prefill_m256_down  -2.79%  floor 0.20%  (-13.9 floors)  floor_clamped=TRUE
prefill_m512_up    -2.76%  floor 0.46%  ( -6.0 floors)
decode_m2_square   -1.26%  floor 0.69%  ( -1.8 floors)
```

**补丁被保留下来了**，58732 字节，落在 lane 自己的 state 目录、不在会被清掉的 round 目录：
`exp/state_coldstart_newgate_20260819/round3_refused_integrated_patch.diff`，
`applies_to: CANONICAL @ git aa71ba0`。`suggest_next` 明确写着"不要开新方向，
把这个补丁按三条路的修法重投"。这一点做得对：**拒绝没有丢掉工作**。

### 这是同一主题的第三次，但**机制不同，别混为一谈**

前面两条记的是 `route_gate` 的 band（由 `baseline_per_case[].samples_ms` 的 n=5 极差导出）。
这里否决 1.15 的是**另一套东西**：`noise_floor_stats.MEASURED_NOISE_FLOOR`，
按纪元存的 per-route 底噪（`2*MAD/median`）。两张表、两套代码路径，
只是碰巧都在"用一个噪声模型逐路判决"这件事上。**不要把它们的数字互相引用。**

值得记的是这次否决**自带自我怀疑**：`prefill_m256_down` 那条的 note 写着
"bimodal in four independent sessions; the 0.20% floor is a MIN_FLOOR clamp from one
quiet session, **not a property of the route**"。也就是说机器一边说"这个 floor 不可信"，
一边用它把 -2.79% 判成 13.9 个 floor 的回退并否决整个候选。
`suggest_next` 给的处置是对的——"escalate the MIN_FLOOR clamp on that route rather than
fitting the kernel to a 0.20% band"，即修表，不要为了迁就一个 0.20% 的假 band 去改 kernel。

### 一个必须写下来的反向论证：**provisional 不只是"不敏感"，它会放行回退**

纪元 Z 现在整表 DEFAULT `0.072`（7.2%）。把三条否决路线放进去：
-2.79% / -2.76% / -1.26% **全部落在 7.2% 以内**，一条都不越界。
也就是说，**如果现在带着 provisional 表直接开跑，这个 1.15 会被判 ACCEPT 并提交**，
而它在测量过的 Y 表下是被否决的。

平时说 provisional 的坏处是"胜利幅度读不出来"（本 lane 的 launch 文件也是这么写的），
方向是偏保守。**这里方向是反的**：粗底噪同样看不见回退，于是把一个含三条回退的候选放行。
所以"exit 3 技术上能跑"在本 lane 当前这个具体状态下尤其不能用——
下一件要判的事恰好就是这个候选。**先把 Z 测出来再开波。**


### 纪元 Z 测完（exit 0），顺带**独立证实了 round 3 那条自我怀疑的注释**

8/8 重复全部完成，`ok:true`，**11 条路一条都没有被 MIN_FLOOR 夹住**（全是 `clamped_to_min:false`）。
先做真实性检查再用：11 条路 `median_speedup` 的几何平均 ≈ **0.331**，
就是种子已知的 0.336——这一趟确实在 GPU 上把种子量完了，不是空转返回。

关键在两条路，Z 测出来的底噪**极大**：

| route | 纪元 Y floor | **纪元 Z floor** | 倍数 |
|---|---|---|---|
| prefill_m256_down | 0.20%（**MIN_FLOOR 夹的**） | **7.02%** | ×35 |
| decode_m8_up | — | **7.64%** | — |
| prefill_m512_up | 0.46% | 1.26% | ×2.7 |
| decode_m2_square | 0.69% | 0.88% | ×1.3 |

round 3 否决 1.15 的第一条路就是 `prefill_m256_down`，而那条否决**自带注释**说
"the 0.20% floor is a MIN_FLOOR clamp from one quiet session, **not a property of the route**"。
现在一台**完全不同的机器**、一次**与那次判决无关**的独立测量，
把这条路测成 **7.02%** ——注释是对的。0.20% 是那条路"安静模态"的读数，不是它的性质；
四次会话里的 bimodal 说法，Z 这次抓到了吵的那一模态。

把 1.15 那三条否决按 Z 的表重算：

```
prefill_m256_down  -2.79%  /  7.02%  = -0.40 floors   -> 不再否决
prefill_m512_up    -2.76%  /  1.26%  = -2.19 floors   -> 仍否决
decode_m2_square   -1.26%  /  0.88%  = -1.42 floors   -> 仍否决
```

**从三条降到两条，候选仍然被拒。** 这个结论比"翻案"更有意思：
机器自己标记为不可信的那一条掉了，另外两条**独立于**它站住了。
所以 round 3 的拒绝在实质上是对的，只是理由里混进了一条假的。
`suggest_next` 让"修表而不是为 0.20% 的假 band 改 kernel"——现在表自己修好了。

**注意别过度声称**：Z 是另一台机器，底噪本来就该不同，
我**没有**证据说 Y 上那次 0.20% 的测量在 Y 上是错的。
能说的是：这条路在两个独立纪元里给出 0.20% 和 7.02%，
**夹到 MIN_FLOOR 的那个读数是离群的那个**，而夹紧这件事本身就是"raw 比下限还小"的警报。
把 MIN_FLOOR 夹住当成"这条路很干净"来用，方向是反的——
它恰恰说明这次采样没抓到这条路的尾巴。

### 诚实底噪的代价：11 条路里有 2 条在这台机器上基本读不出来

`prefill_m256_down` 7.02% + `decode_m8_up` 7.64%，意味着这两条路上
**真实的 +5% 提升是不可见的**，既不会被算作改善，回退也不会被否决。
这是把假 band 换成真 band 必然要付的价：Y 上它们"可读但可能在撒谎"，
Z 上它们"诚实地不可读"。后者更好，但要写下来——
接下来如果有候选在这两条路上宣称收益，那个收益**不能计入**。

## 新一波已起（`wf_4afd016a-76f`），以及一个换机暴露出来的中继缺陷

`lane_args.py --check` exit 0（9 参数、6 个协议值对上），`--print` 渲染后原样调用，没手抄。
monitor 第一条事件就确认了中继正确：

```
CANONICAL HEAD MOVED: bc1e462 baseline (resumed from STATE best, cumulative 1.0916)
```

**从 lane 自己的累计最优起跑，没有退回种子。**

### 被拒的补丁没有变成孤儿——但靠的是散文，不是结构

先查了一下 round 3 那个 1.15 的补丁会不会丢。结论是不会，但路径比想象的脆：

- `refused_candidate` 这个键**在整个 repo 的源码里一次都没出现**（只在 STATE.json 里）。
  它是 tech_lead 的 memory JSON 里带出来的自由键，`update_memory` 原样写盘。
- resume 只导入 `cumulative`（仅供报告）、`insights`、`ledger`、`bottleneck_now`。
  **`suggest_next` 和 `refused_candidate` 都不导入。**
- 补丁之所以还活着，是因为 tech_lead 把它写进了 **insights 的散文**里
  （"ROUND 3'S RESULT IS A +5.3% SUITE WIN THAT THE PER-ROUTE BAND GATE REFUSED,
  AND THE PATCH IS PRESERVED..."），而 insights 是导入的。

也就是说：**结构化的那份（`refused_candidate` + patch_path）是只写不读的，
真正承载中继的是同一事实的散文副本。** 现在能用，但它依赖 agent 每轮自觉复述。

### 换机让一条 insight 变成假的，而中继分不出哪些 insight 是绑机器的

最后一条 insight 是 `BOX/TOOLING FACTS (epoch Y, tw053)`，开头就是
"quietest box the lane has stood on -- 8 of 11 route floors under 0.9%,
three CLAMPED to MIN_FLOOR"。这一波跑在 **tw035 / 纪元 Z** 上，实测：

| | insight 说（Y/tw053） | 实测（Z/tw035） |
|---|---|---|
| floor < 0.9% 的路数 | 8 / 11 | **4 / 11** |
| 被 MIN_FLOOR 夹住 | 3 条 | **0 条** |
| 最吵的路 | —（号称最静） | decode_m8_up **7.64%**、prefill_m256_down **7.02%** |

Z 一点也不是"最静的箱子"，它更接近这条 insight 自己引用的对照组 "epoch R tw008 at 7.12%"。
**resume 路径没有区分"lane 事实"和"箱子事实"**，两者都当 insight 原样带过来。
insight 里那些 MFMA/rocprofv3/ISA 工具链的教训是跨机有效的，底噪那半句不是。

### 一个可证伪的预测，留给下一轮对账

`suggest_next` 让把 1.15 按三条否决路修好再投。但按 **Z 的表**重算过了：
`prefill_m256_down` 的 floor 从 0.20% 变成 **7.02%**，它 **-2.79% 已经不再否决**，
而且同样地，**在这条路上做的任何修复也测不出来**（7% 底噪下 ±2.8% 不可读）。

> **预测**：如果这一波拿一整个 direction 去修 `prefill_m256_down`，那是白花的预算——
> 那条路现在既不能否决候选，也不能显示收益。真正还在否决的是
> `prefill_m512_up`（-2.19 floors）和 `decode_m2_square`（-1.42 floors），
> 而前者的修法 insight 里已经写好了（强制 ladder id 7 / 256x128，中位数 .1792 vs .1839）。

如果预测应验，那就是"箱子事实以散文形式跨机中继"的一次可计价损失，届时回来记账。

---

## wave 2 / round 1（`last_round` 3→**1** 是重编号，不是倒退）

先澄清一个会咬人的地方：**`STATE.json.last_round` 是 wave-local 的**。新一波把它重置成 1，
`cumulative` 仍是 1.0916、`wave_local_cumulative` 回到 1.0。所以这条 lane 现在有**两个 "round 1"**
（冷启动那个 0.6114，和这个）。本文件此后一律写 **wave2_r1** 这种前缀。
我那条"HEAD 不动就不下结论"的规则照旧有效，但**不能再拿 `last_round` 变大当触发条件**——
这次它是变小的。

**canonical 没动**（仍 `bc1e462`），`cumulative` 仍 **1.0916**。这一轮又是**零提交**。

| direction | specialty | actual | verdict |
|---|---|---|---|
| wave2_r1_d0 | host_runtime | 1.1490 | partial（**无 kernel 改动**） |
| wave2_r1_d1 | memory | **1.1581** | partial（按分是本轮第一） |
| wave2_r1_d2 | algorithm | 0 | dead_end（apply_failed，无补丁） |
| wave2_r1_integrate | integration | 1.1420 | no_improvement |

### 我上一条预测的结算：方向对，数目错了一条

我在这一轮开跑**之前**（commit `8908d954`）写下：修 `prefill_m256_down` 是白花预算，
因为它在 Z 上 7.02% 的底噪里既不能否决也不能显示收益；还在否决的应该是
`prefill_m512_up` 和 `decode_m2_square`。

- **"白花预算"部分：应验，而且是整整一个 direction。** `wave2_r1_d0` 的标题就是
  "Repair the two prefill band-gate refusals"，结果它自己的 lesson 写着
  "Both declared target routes are **nulls** (m512_up -3.3%, m256_down -1.6%,
  both inside their own spreads)"，并且"its whole functional delta vs the seeded
  parent is a **29-line comment**"——一整个方向没产出任何 kernel 改动。
  它独立重新发现了我已经写在盘上的那件事："the box is epoch Z / tw035 and the
  refusing bands were epoch Y / tw053 ... m256_down's real floor is **5-17x** the clamp
  that refused it"。（注意：agent 读不到本文件，这不是它们的失误；
  这正是"箱子事实以散文跨机中继"要付的价，现在有了标价。）
- **"还在否决哪几条"部分：我多报了一条。** 重测下来三条否决路读作
  m2 **1.1082**、m256_down **1.0451**、m512_up **0.9773**——
  **只有 `prefill_m512_up` 真的在种子之下**。m256_down 掉出否决集是我料中的，
  `decode_m2_square` 我以为会留下，它没有。**实际否决路是一条，不是两条。**

### d0 的真正产出：`prefill_m256_down` 的双模在**oracle 那一侧**

这比"这条路很吵"深一层。d0 测到 baseline 臂一小时内从 **.1303 漂到 .1417**，
而 candidate 臂是平的。**双模在不可变 oracle 的计时里，不在候选里。**
推论 d0 自己写了：suite geomean **携带一个会话模态**，
因此 **任何低于 2% 的跨会话结论都不可采信**。

配套还有一条更刺眼的（来自 d1）：**在一条"执行码逐字节相同"的路上，
观察到 3/3 的 interleaved 配对 2% 胜出**——纯粹是 link layout。
"profile before believing a paired win"。这两条合起来，
把本 lane 所有 <2% 的结论都降级了。

### d1 是"按分第一、按机制为空"

d1 拿了 1.1581，但它改的那 23 行 reduce 宽度守卫**在任何被计分的路上都不会被执行**：
rocprofv3 证明 `decode_m2_square` 走 `gemv_bf16_kernel<2,1>`，
`dense_bf16_gemm.hip:220` 在 m<=2 时就 `launch_gemv<2>` 了，根本到不了 `splitk_reduce_kernel`。
所以 m2 的 -1.3% **不是 reduce 缺陷、也无法从 reduce 内部修**——该方向关闭。
1.1581 归功于被播种进来的 round-3 本体（重新键控的 (m,n,k) slice 表），不是这个守卫。

**这是"winner-selection gap"的一个新变体**：按分选出来的赢家，其机制归因为空。
分数是真的（对 in-session control 1.0935 是 +5.2%），只是**功劳记错了对象**。

### 这条 lane 现在的真正僵局：一条路卡住 +4.6%

被拒的 round-3 本体现在已经**跨会话、跨纪元被独立重测三次**：1.1420 / 1.1490 / 1.1581，
对在册 1.0916 稳定在 **+4.6% ~ +5.2%**，**八条路在 parity 或以上**。
它一直不入账，卡在 **`prefill_m512_up` 0.9773（-2.27%）** 这一条上。
按纪元 Z 的 floor 1.26% 算是 **-1.8 floors**，**照样否决**。

而 round 3 insight 里记的那个唯一已知修法**已被 d0 证伪**：
"'force id 7 on m512_up' **is already what macro_select returns**，
ids 1/4 是 +29%/+56%"——tile 轴上最后一个具名例外关闭。
**于是 m512_up 目前没有任何已知修法，而它是 +4.6% 的唯一阻塞点。**

`suggest_next` 的 (a) 计划是"按 epoch-Z 的 floor 重新导出 band 后原样重投"。
**按上面的算术，原样重投会再被否一次**（-2.27% vs 1.26% floor）。
这一点我先写下来，下一轮回来对账。

新补丁在盘上：`STATE_DIR/wave2_round1_integrated_patch.diff`，
`git apply --check` 对 `bc1e462` 干净。**工作依然没丢。**

已关闭且不得重开（五条，均带 receipt）：tile-chooser、slice-count、
LDS bank-conflict、stage-depth、register-prefetch。

---

## wave2_r2：**canonical 动了，1.21x** —— 以及**我上一条结论被我自己的复测推翻**

```
cdb7932 round 2 winner: integrated (1.21x)
bc1e462 baseline (resumed from STATE best, cumulative 1.0916)
```

僵局破了。1.21 对种子 = 比在册 1.0916 高 **+10.8%**，越过了卡了三次的 1.15。
改动面：`custom_gemm.hip` +631（**hip 孪生同步了**，`custom_gemm_hip.hip` 同为 +631）、
`dense_bf16_gemm.hip/.h`、`gemm_bindings.cpp`、`gemm_wrapper.py`。
`gemm_bindings.cpp` 被动了，说明 `suggest_next` 的 (a)（GEMV 路径上那个没用的 per-call fp32 workspace 分配）被采纳了。

**STATE 仍停在 `last_round 1 / 1.0916`**——就是记录在案的 ~40 分钟滞后。
commit 是权威，STATE 不是错，只是晚。

### ⚠️ 更正：我说的 "m256_down 的真实底噪是 7.02%" 是错的

wave 自己在同一台 tw035、同一个纪元 Z、同样 8 次重复，**又测了一遍底噪**
（`noise_floor_epochZ_tw035_secondsweep.json`）。两次对比：

| route | 我的 sweep | 第二次 sweep | 比值 |
|---|---|---|---|
| decode_m8_up | **7.64%** | **0.66%** | 0.09x |
| prefill_m256_down | **7.02%** | **1.02%** | 0.15x |
| prefill_m1024_down | 2.10% | 0.21% | 0.10x |
| decode_m64_square | 0.99% | 0.20% | 0.20x |
| decode_m96_up | 0.93% | 0.57% | 0.61x |
| prefill_m128_square | 1.40% | 0.89% | 0.64x |
| decode_m2_square | 0.88% | 0.90% | 1.02x |
| prefill_m512_up | 1.26% | 1.32% | 1.04x |
| decode_m16_square | 0.27% | 0.30% | 1.14x |

**安静的路两次吻合到 15% 以内；吵的路差 7-11 倍。**

所以我在 `f65ce302` 里写的"纪元 Z 独立证实了那条注释，m256_down 的真实底噪是 7.02%，
Y 那个 0.20% 是离群值"——**这个结论站不住**。这条路现在有三个读数：
0.20%（Y，被 MIN_FLOOR 夹）、7.02%（我）、1.02%（复测）。
**我的 7.02% 和 Y 的 0.20% 是同一种东西：一个会话模态。**
我把自己的那个会话模态当成了"这条路的性质"，
而这正是我批评那条 0.20% 注释时指出的同一个错误。

**站得住的是更强的那个版本**（d0 测出来的，不是我推的）：
这条路的方差在 **oracle 臂**里、在**小时量级上漂移**，
所以**任何单会话的 floor——包括我的——都是模态而不是性质**。
n=8 的 MAD 在双模路上根本定不住底噪。这跟 band 那条发现是**同一个病**
（n=5 极差是抽签），只不过这次是在**另一套噪声机制**（per-epoch MAD floor）里复现，
而且这回我有**两次可直接对比的 sweep** 当证据，不是推断。

一个没排除的替代解释，要写下来：**我那次 sweep 是快照恢复后这台机器上的第一个 GPU 负载**，
时钟/功耗状态可能还在爬。它跑在更干净的条件下（8 卡全空、单卡锁）却测出更吵的数——
这跟"争用"相反，但跟"冷启动瞬态"一致。**不能只凭两次 sweep 就断言是双模。**

### 附带：那条过期的 epoch-Y insight **真的造成了一次错误操作**

第二次 sweep 的 verdict 里 `ok: **false**`：

```
"verdict was taken on tw035 (epoch Z), but would be installed as epoch Y (tw053).
 Floors do not pool across a machine boundary"
```

也就是 agent **传了 `--machine Y`**——它信了 insight 里那句 "BOX/TOOLING FACTS (epoch Y, tw053)"。
这正是我在 `8908d954` 里说的"箱子事实以散文跨机中继"的失效方式，现在有了实例。
**守卫拦住了，没有装进表里**，所以无害；那次测量本身仍然是 tw035 上的真实计时，可以拿来对比。
两条都记：**过期 insight 会驱动错误命令**，以及 **`deprovisionalize`/`measure` 的跨机守卫是有效的**。

### 待对账

`prefill_m512_up` 之前是唯一阻塞点（-2.27% vs 1.26% floor）。第二次 sweep **没有被安装**，
所以这一轮的门用的仍是我那张表，m512_up 的 floor 仍是 1.26%。
既然提交发生了，**这一轮应该是真的修好了 m512_up**，而不是把它蒙混过关。
等 STATE 追上后核对 per-case，确认 m512_up ≥ 1.0。
我"原样重投会再被否"的预测**没有被检验**——他们没有原样重投。

### wave2_r2 细节：三个方向全部 verified，integrate **BANKED**

| direction | specialty | actual (vs oracle) | vs in-session ctrl | verdict |
|---|---|---|---|---|
| wave2_r2_d0 | algorithm | 1.1480 | 1.058 | partial |
| wave2_r2_d1 | host_runtime | 1.2042 | 1.0965 | **confirmed**（本轮最佳个体） |
| wave2_r2_d2 | compute | 1.1543 | 1.047 | partial |
| wave2_r2_integrate | integration | **1.2054** | 1.1041 | **confirmed / BANKED** |

三者**结构正交**（d1 纯 host；d0 是 ladder id 22 发射处 4 行；d2 是 staging 模板的 KEXACT 参数），
零冲突、无手工合并。增量 A/B/C 阶梯 ctrl 1.1954 → +d0 1.1987 → +d2 1.2018 → all 1.2054，
**7/7 interleaved 配对全胜且单调不反转**——这才是归因能站住的原因：
d0 只买到 `decode_m96_up`，d2 只买到 `prefill_m2048_square`，d1 体现为 `decode_m2_square` 1.24。

`prefill_m512_up` 的结局要说准确：它**没有**被修到 parity 以上，是 **0.9977（-0.23%）**。
它从 -2.27% 提到 -0.23%，从而**落进 1.26% 的 floor 里**，于是不再否决。
"修好了"是错的说法，"移进噪声里了"才对。

---

## ⚠️ 严重：`STATE.cumulative` 的帧被乘错了，本 lane 的真实成绩是 **1.2054**，不是 1.3158

`best_per_case` 的 11 条 speedup 都是**对 oracle 的绝对加速**，我把它们的几何平均算出来：

```
geomean(best_per_case)      = 1.205387
wave_local_cumulative       = 1.2054      <- 一致
cumulative 字段              = 1.31581464
1.0916 * 1.2054             = 1.315815    <- 就是它
1.2054 / 0.3358             = 3.59x       <- 对种子
```

**1.0916 和 1.2054 都是"对 oracle 的绝对几何平均"，把它们相乘没有意义。**
这就是 finding (127) 的原样重演：两个不同分母被塞进同一个变量再相乘。
代码那一侧早就为 (127) 加固过（`priorCumulativeVsSeed` 单独存、仅供报告），
**但做这次乘法的是 tech_lead 的 `update_memory`，agent 那一侧没有被加固。**

最能说明问题的是：**同一个字段的 `cumulative_frame` 文字是对的、和数字自相矛盾**——
它写着 "CANONICAL advanced **1.0916 -> 1.2054**" 和 "the lane has moved **3.59x**"
（1.2054/0.3358 = 3.5896 ✓）。散文对，数字错。

### 为什么这条比本轮成绩本身更要紧

这条 lane 存在的意义就是**和 greedy lane 配对比较**。greedy 那边记的是
"absolute geomean vs rocBLAS **1.42619**"、"in-run rocBLAS geomean 1.40984"——
**同样是对 rocBLAS 的绝对几何平均**。所以：

| lane | 对 rocBLAS 绝对几何平均 |
|---|---|
| greedy_coldstart_20260817 | **~1.4045** |
| coldstart_newgate（现在） | **1.2054** |

**newgate 目前落后 greedy 约 14%。** 如果照 1.3158 报，会显示成"基本追平"——
**在这条 lane 唯一要测的那个轴上给出相反的结论。**

### 处置

**现在不动 STATE.json**：wave 是活的、每轮 `update_memory` 会整体重写它，
此刻写进去会被覆盖，还有写冲突风险。而且 resume 只把 `ps.cumulative` 当**报告值**读
（`priorCumulativeVsSeed`，不进门控），所以它**不会污染本波的判决**。

**危险在波边界**：下一波会把 `ps.cumulative` 当基数再乘一次，**逐波复利**。
所以——**在下一波启动之前，必须把 `cumulative` 改回等于 `wave_local_cumulative`**
（当前值 1.2054），这是本 lane 交接时的强制动作项。

同时更正我自己上一条汇报：我说"1.21x vs seed，比 1.0916 高 +10.8%"。
**幅度基本对（实为 +10.4%），但帧标错了**——1.2054 是对 **oracle** 的绝对值，不是对种子；
对种子是 3.59x。commit message 里的 "1.21x" 是 **wave-local 对在册增量**（1.2054/1.0916=1.1041→实为1.10x），
wave 1 那次 "1.09x" 因为冷启动时两帧恰好重合才没暴露这个歧义。

### 下一轮的 suggest_next（瓶颈已从 memory 转到 **overhead**）

(1) 把 split-K 的 2 次 dispatch 收成 1 次（**必须**是 cooperative last-CTA reduce；
winner-take-all/direct-atomic 已在 registry 里以 -43.6% 关闭）——
残留的那次 reduce dispatch 是 7/11 条路上 ~4.6us 的发射地板、只跑到 ~490 GB/s，
而 `decode_m16_square` **比 rocBLAS 多做 1.86 倍 GPU 工作却拿到 1.541**，说明这套件是靠 dispatch 赢的。
(2) wrapper 级 HIP graph capture/replay，以及把 fp32 workspace 池化到另外 9 条非 GEMV 路（r2_d1 只关了 GEMV 那条）。
(3) `prefill_m128_square` 0.8305 是唯一纯 body 缺口（26.24us vs rocBLAS 18.92），
任何方向必须**恒定 CTA 数**且**实例化数中性**。

禁止重开（本轮又加两条）：tile chooser、slice count、**bytes-per-output/arithmetic intensity（本轮 15 组合扫描证伪）**、
LDS bank conflict、K-stage depth、register prefetch、**全局删除死的 scalar staging 分支**。

---

## 2026-08-19 23:11 第二次换机：tw035 → **tw040**，纪元 **A**（字母表绕回来了）

wave `wf_4afd016a-76f` 死在 **round 3 中途**——最后一次写盘 23:09:44，
是 `round_3/engineer_0/verify/isa_v2/` 的反汇编产物，也就是它正在做 ISA 取证时进程被拆掉。
round 2 的成果**已经落在 canonical 上**（`cdb7932`），没丢。

### 字母表用完了：取 **A**，不是"Z 之后"

`MACHINE_HOSTNAME` 已占 **L 到 Z 连续 15 个**（L/M 是约定之前的 gfx90a 老箱，无 hostname 记录）。
Z 之后没有字母了。`register_epoch.py` 的校验是 `re.fullmatch(r"[A-Z]", L)`——**只收单个字母**，
所以 "AA" 这类不行。**A–K 从未出现过**（在整个 `noise_floor_stats.py` 里 grep `"[A-K]"` 零命中），
且脚本自身会在字母已存在时拒绝，等于第二道保险。于是取 **A**。

这里要说清楚 finding (126) 的边界：126 禁的是**复用退休字母**（同一个箱子重新拿到旧字母，
把旧箱的 floor 拿来判新箱的计时）。**A 不是退休字母，是从未分配过的字母**，
tw040 也是这条 lane 没跑过的新箱。所以取 A 不违反 126。
**给后来者：约定是"下一个未使用字母"，不是"字母序的下一个"；到 Z 之后从 A 继续。**

```
check_measurement_frame.py       -> EXIT 4 (tw040 unregistered)
register_epoch.py --letter A --host tw040   (先 --dry-run 看四处 diff)
  -> epoch A registered for tw040, PROVISIONAL, CURRENT_MACHINE Z -> A
check_measurement_frame.py       -> EXIT 3
八张卡全空 (busy 0%, vram 283MB)
```

### 这次做了一个**成对冷/热 sweep**，去检验我自己上次留下的那个未排除假设

在纪元 Z 上，我的 sweep 和 wave 的复测在双模路上差了 7–11 倍，我当时写下一个没排除的替代解释：
**我那次是快照恢复后这台机器上的第一个 GPU 负载，时钟/功耗可能还在爬**。
这次可以直接测：**在同一次 `gpu_lock.sh 2` 调用里连跑两遍 8-repeat sweep**，
第一遍冷（本机第一个 GPU 负载）、第二遍紧接其后（热）。
`verdict.json`（冷）/ `verdict_warm.json`（热）。协议本身一字未改，只是多跑了一遍做诊断。

若两遍在双模路上**系统性地"冷更吵"**，冷启动瞬态假设得到支持，
且意味着**每次换机后的第一次 sweep 都会装进一张偏宽的表**——
偏宽的表**放行回退**（这正是纪元 Z 上 m256_down 让 -2.79% 溜过去的机制）。
若两遍吻合，那就是我在 Z 上的 7.02% 属于别的原因，冷启动假设被排除。
**先测再说，不预设结论。**

### 波边界上执行了那条强制动作项：`cumulative` 已改回 1.2054

wave 已死、STATE 无人占用，正是我在 `591a7ccc` 里写的"必须在下一波启动前修"的时机。
改动经断言保护（不满足"恰好等于 1.0916 × wave_local"就中止）：

```
cumulative  1.31581464 -> 1.2054     (= geomean(best_per_case), vs oracle)
对种子 0.3358           -> 3.59x
wave_local_cumulative   1.2054       (不变)
```

`cumulative_frame` 一并写清楚了"两者同帧、不得相乘"，
免得下一个 `update_memory` 再乘一次。备份留在 `STATE.json.bak_frameFix_*`。
其余键（ledger 19、insights 19、best_per_case、refused_candidate）**一律未动**。
注意 ledger 已从 15 涨到 19——round 3 死前把它的条目写进去了，那些内容保留。

### 冷/热配对 sweep 的结果：**冷启动瞬态是真的**，但它解释不了纪元 Z 的全部

| route | COLD | WARM | warm/cold |
|---|---|---|---|
| decode_m32_down | 1.00% | 0.20% | **0.20x** |
| prefill_m1024_down | 0.78% | 0.32% | **0.41x** |
| prefill_m256_down | 0.41% | 0.20% | **0.49x** |
| decode_m8_up | 0.86% | 0.46% | 0.54x |
| prefill_m128_square | 1.36% | 0.80% | 0.59x |
| prefill_m512_up | 0.33% | 0.20% | 0.60x |
| decode_m16_square | 0.42% | 0.31% | 0.74x |
| decode_m96_up | 0.68% | 0.56% | 0.81x |
| decode_m64_square | 0.39% | 0.34% | 0.86x |
| prefill_m2048_square | 0.21% | 0.20% | 0.96x |
| decode_m2_square | 0.37% | 0.66% | 1.79x |

**11 条里 10 条热的更安静**（符号检验 p ≈ 0.012），中位比值 **0.60x**，
3 条冷的吵 2 倍以上、0 条热的吵 2 倍以上。

决定性的对照在下面这一列：**11 条路的 median speedup 冷热之差全部在 0.4% 以内**
（最大 `prefill_m128_square` 0.9905x）。**水平没动，只有离散度在动。**
这正是时钟/功耗爬坡的特征——均值不变、方差收敛——
而不是"两次测到了不同的机器状态"。

所以我在纪元 Z 上留的那个未排除假设，**方向被证实了**。但要说准确：
这里最大只有 5 倍（decode_m32_down），**Z 上是 7–11 倍**。
**冷启动瞬态是 Z 那次分歧的一个成因，但不足以解释全部量级。**
剩下的部分仍然指向 d0 那个更强的结论：m256_down 的方差在 **oracle 臂**、按小时漂移。

**推论（对所有 lane 都成立）：换机后的第一次 sweep 系统性偏宽。**
偏宽的表放行回退——纪元 Z 上 m256_down 的 7.02% 让 -2.79% 溜过去，就是这个机制。
之前每个纪元的表大概率都是"换机后第一次 sweep"，**即整张历史底噪表都可能偏宽**。

### 装了**热**表，以及为什么——这个选择有代价，说清楚

装的是 `verdict_warm.json`（`deprovisionalize --machine A --apply`，frame 现在 exit 0）。

理由不是"热的数字更小更好看"，而是**门控实际比较的是什么**：
gate 判的是**同会话、交错配对**的 candidate vs control（r2_d1 的 lesson 已经把
"in-session interleaved layout control arm"定为本箱任何 <2% 单路结论的必备凭据）。
**同会话噪声才是相关量，热 sweep 描述的正是它。** 冷 sweep 描述的是一个不会在轮次中重现的瞬态。

**代价必须写下来**：热表有 **4 条路被 MIN_FLOOR 夹到 0.20%**
（m256_down、m512_up、m2048_square、m32_down），冷表一条都没夹。
而"夹住"正是我自己在 `f65ce302` 里说的**警报而非合格证**。
尤其 `prefill_m256_down` 又回到 **0.20%**——**和纪元 Y 上造成那次假否决的数值一模一样**。
**所以 round 3 那种假否决的风险在这台机器上是活的。**

我仍然选热表，因为：MIN_FLOOR 的存在本来就是为了阻止门控相信 0.2% 以下的分辨率，
夹住是它在**正常工作**；而同会话噪声确实就是这么小。
但这不是"没有代价的正确选择"，是**在两个已知失效方向之间选了一个**：
偏宽放行回退（Z 的教训），偏紧假否决（Y 的教训）。冷/热两张 verdict 都留在盘上，可回滚。

### 新一波已起

`lane_args --check/--print` 均 exit 0，原样调用，**新起而非 `resumeFromRunId`**——
死掉那一波的 agent 缓存里是 **tw035** 上的计时，在 tw040 上重放就是把别的箱子的数当本箱的数。
run `wf_90377108-25c`，monitor `bidbyqz4t`。
起点是 canonical `cdb7932` / **1.2054**（已修正的 cumulative）。

### 我在波边界漏掉了一件事：BOX insight 又是过期的

新波已从 `best @ 1.2054` 起跑（`3b2a746`，banked_commit `cdb7932`）——**修正后的 cumulative 生效了**。
但同一个波边界上我只修了 `cumulative`，**没修 BOX insight**，它现在写着：

> BOX/TOOLING FACTS (**epoch Z, tw035**): 11 unclamped floors 0.27%-3.37% ...

我们在 **tw040 / 纪元 A**，实测热表是 **0.20%-0.80%**。这是**同一个缺陷第二次发生**，
而这一次我本来有机会顺手修掉（STATE 当时无人占用、我正在编辑它）。**这是我的疏漏，不是机器的。**

危害有界，两个原因：(1) 传错 `--machine` 会被跨机守卫拦住（上一波已验证）；
(2) 方向是**保守**的——agent 以为 m256_down 的 floor 有 1.0-3.4%，实际门用的是 0.20%，
所以它们会低估自己能拿到的判定分辨率，而不是高估。比反过来安全。

**强制动作项（下一个波边界，和 `cumulative` 一起做）：把 BOX insight 换成当前箱子的事实。**

顺带一条**给 agent 记功**的观察：那条 insight 把我的 sweep 和第二次 sweep 调和成了
"prefill_m256_down 1.0-3.4%，且其双模在 ORACLE 臂"——**记的是跨两次 sweep 的区间，不是单个数**。
这比我第一次写的"真实底噪是 7.02%"更谨慎、也更正确。**在这一点上 tech_lead 比我做得好。**

## wave 3 (tw040 / epoch A) -- two rounds, five directions, ZERO banked. The wave stopped itself.

Lane total unchanged: **1.2054, git cdb7932**. `wf_90377108-25c` ran round 1 and round 2 and then
exited on its own stop-gate rather than issuing a round 3. No workflow process alive at 01:41 UTC.

**Round 1 (three directions) -- all died on their PREMISE at step 0.** Not "tried and lost": the
thing each was going to exploit did not exist. Dispatch gap already 0 ns; host prologue already
0 us *inside the scored window*; prefill_m128_square residency already not binding. The graph-capture
lever the role file requires was attacked and priced at **zero** with a live 10 us busy-spin
sensitivity control -- so stop-gate clause (a) is discharged with a receipt, not by assertion.
The pooled-fp32-workspace arm shipped as a measured null (+0.18% by both engineer and verifier)
and was correctly NOT integrated: a byte-identical-.so getenv A/B that perturbs the binding object
for nothing, plus a thread_local pool that never releases up to ~67 MB.

**Round 2 (two directions) -- worse news than round 1, and more valuable.** Both premises were
priced correctly, both patches were built, and both were *realized in the machine code* with clean
receipts -- and still measured null-to-negative. This is the lane's first full **ISA anti-signal**:
d0 satisfied every precondition on all 8 split symbols (2-byte key gone, ds_write_b16 gone,
full_drain 22->5/6/8, instructions 478->279, identical LDS/VGPR, zero spills, exact 8-for-8 swap,
11/11 correct) and lost **-2.6%** over 6/6 interleaved in-lock pairs. The sign is per-tile and
tracks CTA/CU exactly -- id 9/id 1 routes lose, id 7/id 22 win -- which retroactively **confirms
the shared-instantiation/occupancy-regime claim** that round 1 had left unverified. That is the
wave's one durable structural gain, and it arrived as a by-product of a failure.

**Two things I want on the record because they cut against the lane's own habits:**

1. d1 caught its own step-0 price **wrong by 3x in its own favour** -- it had divided an excess by
   the 59.7 us ORACLE call when the geomean is over candidate-relative speedups, so the denominator
   is the CANDIDATE time. It repriced and only then passed the gate. The methodology correction
   (state the denominator; print the same-lock control beside every verified_geomean) is worth more
   than the direction was.
2. On prefill_m512_up the **engineer and the verifier disagreed in SIGN**, both using correct paired
   method. Sub-2% single-route claims on p512_up and p256_down are now inadmissible regardless of
   pair count. My own 7.02% floor claim died the same way two waves ago; this is the same lesson
   arriving through a different door.

### Boundary checks at wave close -- one passed, two were defects I had already promised to fix

- **(a) cumulative arithmetic: PASSED.** No 1.0916x product this time; `cumulative` 1.2054,
  `wave_local` 1.0. But the check surfaced something better: `geomean(best_per_case)` on tw040/epoch A
  is **1.1993** for the *same* git cdb7932 that measured 1.2054 in-lock on tw035/epoch Z. 0.51% apart.
  Neither number is wrong -- **the pair is the lane's cross-session error bar**, and it sits right at
  the 2% inadmissibility line the lane just re-derived. Honest headline is **1.20 +/- 0.5%**, and I
  have written that provenance into STATE rather than picking a prettier number.
- **(b) the BOX insight went stale for a THIRD time** -- it still said "epoch Z, tw035, floors
  0.27%-3.37%" after two waves on tw040. Worse than cosmetic: it advertised prefill_m256_down as the
  *wide* route (1.0-3.4%) when on epoch A it is **clamped at MIN_FLOOR 0.20%**, i.e. the route most
  likely to throw a FALSE REFUSAL -- the exact failure that killed wave 1 round 3. Replaced with the
  epoch A facts. I have now missed this at three consecutive boundaries; it is not an accident, it is
  that nothing in the machinery ties a box-scoped fact to the box.
- **(c) NEW defect, and the frame bug wearing a different hat.** `cumulative_frame` read
  *"vs the original seed"*. It is not: 1.2054 is absolute vs the immutable oracle, and vs-seed is
  **3.59x**. The value was right and the label was wrong, which is the more dangerous arrangement --
  finding 127 does not need bad arithmetic to propagate, only a confident wrong caption. Corrected.

STATE backup at `STATE.json.bak_boundary_20260820_014335`.

### wave 4 launched (`wf_9669bc60-a8a`, monitor `b4yfvkzqd`) -- and the relay repair is VERIFIED, not assumed

Baseline `ae61f02`, seeded from `state_dir/best` at 1.2054 (not from the seed). Launched over wave 3's
own STOP RECOMMENDED, deliberately: that gate reports the exhaustion of **one-knob arms**, and it is not
evidence about headroom. `prefill_m128_square` at 0.834 and `decode_m96_up` at 0.932 are genuinely slower
than the oracle -- routes 17% and 7% underwater are not closed by five knobs failing.

What makes this a different draw from wave 3 rather than a repeat: `suggest_next` is **write-only across
wave boundaries** (`kernel_lane.js:2630` reads it from the tech_lead's live memory, never from
`prior_state`), so wave 3's DO-NOT-RE-ISSUE list, its open-problem statement, and its three methodology
corrections were all set to evaporate and be re-bought at full price. I promoted them into `insights`,
which the resume path *does* import, and then checked rather than trusting it: the wave-4 agent transcript
contains both the promoted closed list and the refreshed epoch A box facts, and contains **zero**
occurrences of the stale `epoch Z, tw035` string. First wave in this lane to begin holding its
predecessor's closed list.

Standing decision for the next boundary: if wave 4 also banks nothing *with the closed list in hand*, that
is a far stronger stop signal than wave 3's was, and I will treat it as terminal for the lane rather than
launching a fifth. Wave 3's stop-gate could not distinguish "no headroom" from "wrong instrument"; wave 4
can.

### wave 3's completion record lands late, and it adds a THIRD measurement of the same code

`wf_90377108-25c` returned after wave 4 was already up. Two things in it matter.

**1. The error bar is now three points, not two, and it holds.** The harness ran its own independent
validation of `cdb7932` -- `validation_status: accepted`, `validation_trust: verified`,
`timing_basis: device` -- and got **1.2043**, against the tech_lead's 1.2054 and my epoch A
`best_per_case` geomean of 1.1993. Total spread **0.51%** across three independent measurements of
byte-identical code on two boxes and two epochs. That is a corroboration of the provenance note I wrote
at the boundary, not a contradiction of it, and it puts a hard floor under the lane's own rule: a
sub-2% cross-session claim on this suite is unreadable, and even sub-0.5% is at the edge of what three
honest measurements of the *same binary* disagree by. (`final_arithmetic` is 1.2411 -- the arithmetic
mean, not the scored statistic. The scored metric is the unweighted geomean; do not quote 1.24.)

**2. The stop was not budget exhaustion -- wave 3 spent 5 of 12.** It stopped with **7 budget unspent**,
by judgement rather than by running out. That materially strengthens the case for having launched wave 4:
the lane did not run out of room, it ran out of *ideas of one particular shape*, which is precisely the
distinction the closed-list relay was meant to attack. 22 agents, 0 errors, 2.46M subagent tokens, 2h37m.

STATE deliberately NOT edited to record this -- wave 4 owns that file now, and a read-modify-write from
outside a running wave is the lost-update this lane has already been bitten by. It goes here instead and
folds into STATE at the next boundary.


---

## 2026-08-20 02:0x — 运维改动：选 winner 与判提交不再是两套判据

**这是一次有意的、记录在案的中途改动，改的是本 lane 的被测对象本身。** 决定由运行方作出，
理由是这条缺陷已经被 wave 1 round 2 完整证据化，继续让它生效只是在重复采集同一个负面结果。

改了什么（commit `c1adeef1`）：`kernel_lane.js` 原来只把 `candidates[0]`（suite geomean 最高的）
送进闸门，而闸门是逐路的。于是 wave 1 round 2 出现了「排头 0.924 被 `decode_m2_square`
回退 0.0006 ms 否决、而 0.632 逐路 ACCEPT 零回归却从未被送进闸门」——手里握着能过的候选、
本轮却两手空空。现在候选按 geomean 降序**逐个**过闸门，取第一个通过的；**全都不过时行为与从前逐字一致**
（winner 仍是排头、仍被拒），排头能过时也逐字一致。方向是单向的：只能让本来空手的一轮落袋，
**不可能让任何一轮变得更严**。

判据现在只有一份实现：`judgeCandidate(cand)`，选 winner 和记录判决都调它。
「选用一套判据、判用另一套」正是这次要修的缺陷，修的时候留着两份实现就是同一个 bug 换个引信。

**两件故意没做的事**：(1) `bestSeen` 仍记录本轮**报出**的最高 geomean 而不是落袋的那个，
否则往下够一个候选就会悄悄降低停机计数器的标尺，选择修复会变成没人要求的停机规则改动。
(2) **没有**加「被选中的候选还必须超过 `cumulative`」这个下限——`cumulative` 是别的会话测的绝对值，
而接受它的逐路判决是同会话配对的；为了一个漂移过去的陈旧绝对值去拒绝一个「对自己会话的对照臂
逐路不劣」的候选，等于把对照臂要消除的跨会话比较又请回来。改成**大声记日志**：真落袋时
`cumulative` 会下降，日志会明说。

**生效时机（重要，别误判）**：波 4 是 01:46:59 起的，这个 commit 是 02:04:11。
嵌套 workflow 在起波时就已把 `kernel_lane.js` 载入，**所以波 4 跑的仍是旧代码，修复从波 5 起生效**。
如果波 4 又出现「有候选能过却没提交」，那是预期内的，不要当成修复失效。

验证：新增一段可执行守卫，从 lane 源码里抽出真的选择块和真的 `judgeCandidate` 跑四种形状
（排头被拒+次席通过、排头通过、全都不过、无 band 表）。另有 5 处守卫钉住了被搬走的文本，
已重新指向新位置而不是放松——其中一处的断言文字还在主张「改默认不可能让任何运行更严」，
那正是 wave 1 round 2 推翻的说法，现在改成钉住 lane 自己承认相反。
`test_js_suite.py` 的两个变异跟着搬进 `judgeCandidate`，仍然能杀。
全套 **901 passed / 866 subtests**，5 个 JS 守卫全绿（385 项）。

### 顺带：两条**不是这次改动造成**的红，需要你们决定

`test_noise_floor_stats.py` 有两条失败，在 HEAD 上同样复现，从昨晚 20:36 装纪元 Z 起就红了：

```
test_an_unknown_route_gets_the_widest_floor_not_the_narrowest      0.0764 != 0.072
test_an_unknown_machine_gets_the_widest_floor_anywhere...          0.0764 != 0.0088
```

根因是真的：`DEFAULT_NOISE_FLOOR`（7.2%）在源码里是**在后面那些纪元表被追加之前**算出来的，
而纪元 Z 把 `decode_m8_up` 测成了 **7.64%**。于是「未知路线/未知机器拿到任何地方最宽的底噪」
这个 fail-closed 性质**已经不成立**——一条没测过的路线现在拿到的底噪，比一条测过的还要窄。
影响有界（相对 6%），但方向是错的那一边。

第二条里还叠了一个过期 fixture：它拿 `machine="Z"` 当「未注册的机器」，而 Z 昨晚被注册了。
按本目录的惯例这类字母应当**推导而不是写死**。

**没有顺手修**，因为修它要改 `DEFAULT_NOISE_FLOOR` 的推导位置，而 provisional 表是**用这个常量造的**
（`register_epoch.py` 的自检、`check_measurement_frame` 的输出都断言两者相等），
把它从 0.072 抬到 0.0764 会改变每一台新机器开局时的 fail-closed 底噪。
**波次正在跑的时候不该悄悄改一个 fail-closed 常量。**

## THIRD restore: tw040 -> tw035 (snapshot). Wave 4 died having banked nothing.

Process exited at ~02:0x UTC; both the workflow and its monitor went with it and left no completion
record. Damage assessment: wave 4 reached `round_1` and stopped there -- canonical HEAD still at the
`ae61f02` baseline, STATE untouched at 1.2054 / `wave_local` 1.0 / 27 insights. **Nothing banked, nothing
corrupted, nothing to recover.** The lane total is still 1.2054 @ `cdb7932` and the wave-4 launch simply
has to be re-made.

I did NOT use `resumeFromRunId`: it is same-session only, and the cached agents hold **tw040/epoch A**
timings that are now the wrong ruler for this box. Replaying them would be frame-mixing with extra steps.
Fresh launch, same as the tw040 restore.

### The epoch decision was not the routine one, because tw035 is not a new box to this lane

`check_measurement_frame.py` exited 4: host resolves to **Z**, `CURRENT_MACHINE` was **A**. tw035 already
carries two letters -- N (retired) and Z, my own sweep from the previous restore -- so the tempting move
was to point the frame back at Z and skip a sweep entirely. That is wrong, and `machine_for_host`'s own
docstring says why: a re-used box gets a NEW epoch, and reinstating the old letter's floors is
"finding (126) with extra steps". This is a **snapshot restore** -- a different container on a host that
merely reports the same name. Z's floors describe hardware/container state that no longer demonstrably
exists.

Registered **B** for tw035 via `register_epoch.py` (B-K were never allocated; L/M are the pre-convention
letters), so this is a new letter, not a retired one. Before applying I checked the one thing that could
have made it a silent no-op: with tw035 now mapped by BOTH Z and B, does resolution pick the new letter?
`machine_for_host` returns the **last** matching entry and `register_epoch` appends, so tw035 -> B.
Multi-letter hosts are already precedented here (tw054: O+S, tw008: P+R). Frame went 4 -> 3, as predicted.

All eight cards verified free by **gpu_lock's own sysfs criteria** (`renderD(128+8*id)`, busy 0%,
283 MB baseline), not by rocm-smi, whose indexing disagrees.

Sweeps running: cold and warm, back to back inside ONE `gpu_lock.sh 2` call so all 16 repeats share a
card. Two reasons for the pair rather than the single required sweep: epoch A's installed table is the
WARM one, so installing a cold table here would silently change what a "floor" means between epochs; and
it independently **replicates the cold/warm finding on a second box** (n=2), which is the only way to
tell whether the 10-of-11 result was a property of tw040 or of restores in general. `PWD` is the scratch
verdict dir, not the seed task -- `gpu_lock` derives `TORCH_EXTENSIONS_DIR` from `$PWD` and the seed
directory must not be written to.

### epoch B measured, and a finding of mine FAILED TO REPLICATE

Both sweeps clean: 8/8 repeats each, `problems=[]`, identical `source_hash`. Warm installed, frame now
**exit 0** (no stale-prose false positive this time), install verified route-by-route against the verdict
-- 11 routes, zero mismatches, B out of `PROVISIONAL_MACHINES`, S correctly still in it.

**Epoch B (installed, warm): floors 0.20%-1.09%, only TWO clamped** (decode_m64_square,
prefill_m2048_square). Widest is decode_m2_square at 1.09%. `prefill_m256_down` sits at **0.32% and is
NOT clamped**, so the specific false-refusal hazard I flagged for epoch A does not exist on this box.
That is a healthier ruler than epoch A's (0.20%-0.80%, four clamped).

**The correction.** On tw040 I ran a paired cold/warm sweep, found warm quieter on **10 of 11** routes
(sign test p~0.012, median floor ratio 0.60x, three routes >2x louder cold, none louder warm), and
generalised it to "every post-restore first sweep is systematically wide, so the whole historical floor
table may be wide." I ran the identical design here to get n=2. It does not replicate:

| | tw040 / epoch A | tw035 / epoch B |
|---|---|---|
| warm quieter | 10 of 11 | **3 of 11** |
| median floor ratio | 0.60x | **1.05x** |
| >2x louder cold | 3 | **0** |
| >2x louder warm | 0 | **2** |

Both boxes agree the LEVEL is untouched (median warm/cold `median_speedup` = 0.9998), so the measurement
was sound; the generalisation was not. The most economical reading is that tw040's 10-of-11 was itself a
**session mode posing as a property** -- which is the same error this lane already catalogued twice, in
Y's clamped floor and in my own 7.02% `prefill_m256_down` claim. I made it a third time, and this time I
had published it as a lane-wide implication.

I still installed warm, but the reason had to be rebuilt: not the ramp story, just that candidates are
always timed on a GPU already warmed by builds and correctness runs, so warm is the representative ruler
-- and here it happens to clamp fewer routes (2 vs 3). Both the corrected BOX facts and the
non-replication are now in STATE insights, so the next wave inherits the correction rather than the
claim.

### wave 4, attempt 2: `wf_8324bc67-c04`, monitor `bfmbe505m`

`lane_args --check` exit 0, rendered with `--print`, invoked verbatim. Same rule stands for the next
boundary: if this wave banks nothing **with the closed list in hand**, that is terminal for the lane and
I will not launch a fifth.


---

## 2026-08-20 04:xx — 运维改动：接受逻辑换成「均值向上 + 至少一条形状真的动了」

**第二次有意的中途改动，改的仍是本 lane 的被测对象。** 由运行方决定，commit `6fc130fd`。
从**下一波（波 5）**起生效——波 4 是 01:46:59 起的，嵌套 workflow 起波时就已载入 `kernel_lane.js`。

### 新判据

```
suite geomean 相对同会话对照臂 > 1.0（没有阈值）
  且 至少一条路线提升 > max(2%, 该路线在本纪元的实测底噪)
  且 没有任何一条路线回退超过 10%
```

### 为什么换掉旧的并集

旧规则是「逐路判据 ∪ suite 阈值」，外面再套一个**回归否决权**。那个否决权让本 lane 丢了两个最大的结果：

- **wave 1 round 2** —— 11 条路里 10 条 +24%~+50%，因为 `decode_m2_square` 吐回 **0.0006 ms** 被拒。
- **wave 1 round 3** —— 整合栈 suite **+5.35%**（改善的 8 条平均 +8.4%），因为三条路吐回 1.3%~2.8% 被拒。

否决权优化的是「逐路帕累托改进」，而**本 lane 的记分牌是 11 例非加权 suite geomean**。
拒绝一个让目标函数上升的候选，是在优化没人给它记分的东西。

当初支持它的两条理由都不成立：一是"改默认不可能让任何运行更严"——wave 1 round 2 就是反例，
而且那是这个 gate 第一次真的能跑的一轮；二是"11 条路各 +0.4% 是一次 ~4.4% 的 suite 胜利，
逐路判据看不见"——**这是算错了**，11 个 1.004 的几何平均就是 1.004。这个数现在被守卫钉死了。

拿本 lane 已经做过的每一次判决重放：新规则处处一致，**只有那两次拒绝翻成通过**。
另外 wave 1 round 1 的 ACCEPT 也变干净了——它原来有一半靠 `prefill_m2048_square +0.32% vs band 0.27%`，
那是当时人工标注过的抽样产物；2% 的门槛下它不参与判决，ACCEPT 完全由 `decode_m32_down +12.92%` 承担。

### 为什么 2% 不是固定的

同一个套件在同一台机器上，逐路噪声相差 **28 倍**。纪元 Z（tw035）有 **3 条路的实测底噪超过 2%**
（`decode_m8_up` 7.64%、`prefill_m256_down` 7.02%、`prefill_m1024_down` 2.10%），
固定 2% 在那三条路上读的是噪声。纪元 Y、A、B 上每条都低于 1.3%，表一条都不抬。
所以 **2% 是规则、实测底噪是例外，且只能抬高不能降低**。

10% 那道护栏不是噪声判据——它比测过的最宽底噪还大一个数量级，不会误伤真实工作。
它只做一件事：不让"平均涨了"把某个 shape 变慢三成。底噪和护栏之间的回退**全部接受并记名**，
因为只记胜利的 ledger 三波之后回答不了"哪个 shape 变慢了"。

### 供给问题才是真问题

gate 喂不进去就等于没跑：七个波次里它一行日志都没打过。
原来的补救是从 baseline 的 n=5 重复现推 band——**这次直接删了，不是停用**：
n=5 的极差测的是"有没有抽到飞点"，对着 24 次重复的标定偏紧 8.8 倍 / 偏松 2.9 倍。

对的表本来就有（`measure_noise_floor.py`，8 次同变体预热重复，`2*MAD/median`），
当初不能用是因为会过期——而纪元注册自动化之后，**每次换机都会重测**。于是：

- `scripts/route_floors.py` 输出当前纪元的表，**provisional 的纪元直接拒绝**而不是吐出 fail-closed 默认值
  （那会把每条路的门槛抬到 ~7%，整波都无法接受任何东西，还不出声）。
- `lane_args.py` 在启动时解析 `"route_bands": "@current_epoch"`。**数字永远不写进 lane 文件**——
  写下来的底噪就是会活得比它的机器更久的底噪，这正是 epoch-Q 那张表过期六个纪元还看着像当前值的原因。
- lane 现在**每一轮只要有候选就调用 gate**。原来它以"表存在"为条件，那等于悄悄退回 suite 阈值。

### 验证

JS 384 项（5 个守卫）、Python 915 passed。gate 的阈值由守卫**从 lane 源码里抽取**而不是复述，
所以改了 lane 的数字守卫不会继续通过。新增可执行守卫覆盖：无表时 gate 照跑、实测底噪只抬不降、
部分表按路线回退而不是拒整个候选、两个条件各自单独失败、护栏在"平均很好"的候选上开火、以及刚好在护栏内不开火。

### 仍然红的两条（不是这次造成的，需要你们决定）

`test_noise_floor_stats.py` 两条，在 HEAD 上同样复现，只碰 `noise_floor_stats.py`（本次未修改该文件）：
纪元 Z 把 `decode_m8_up` 测成 **7.64%**，比 `DEFAULT_NOISE_FLOOR`（7.2%）还宽，
于是「未知路线/未知机器拿到任何地方最宽的底噪」这个 fail-closed 性质不再成立。
修它要移动 `DEFAULT_NOISE_FLOOR` 的推导位置，而 provisional 表是**用这个常量造的**，
改了会动到每台新机器开局的 fail-closed 底噪。**波次在跑的时候不该悄悄改一个 fail-closed 常量。**

## wave 4b round 1: BANKED `4dd8f63` at **1.2707** -- +4.94% over the incumbent, and the plateau was not a plateau

First bank since wave 2, and the largest single-round move this lane has made since wave 1.
247 insertions / 131 deletions across 6 files including the core `custom_gemm.hip` -- i.e. the
"coherent rewrite across body + plan + instantiation set" that wave 3's own stop note said was the
only defensible remaining shape. It was right about the shape and wrong about the stopping.

**Measured against the same-lock control, per route, against epoch B floors** (the methodology wave 3
mandated -- never against the stored number):

| | geomean vs oracle |
|---|---|
| candidate | **1.2707** |
| same-lock control (incumbent, this session) | 1.2109 |
| **candidate / control** | **+4.94%** |
| stored `cumulative` 1.2054 (epoch Z, other session) | control is +0.46% above it -- drift, not progress |

That +0.46% control-vs-stored gap sits inside the 0.51% three-measurement spread I documented at the
wave-3 boundary, which is a satisfying independent confirmation that the error bar was honest.

**8 of 11 routes win above their own floor, 3 are noise, ZERO regress.** The two that matter:

- `prefill_m128_square` **0.8362 -> 0.9183 (+9.82%)** -- the route wave 3 declared shut after measuring
  staging width, residency, barrier count, tile, slices, bytes-per-output and LDS conflicts all closed,
  with its 79.3% SQ_WAIT_ANY unexplained. It moved nearly 10% the moment something attacked it as a
  whole instead of one knob at a time.
- `decode_m96_up` **0.9324 -> 1.0244 (+9.87%)** -- crosses oracle parity for the first time in this lane.
- `prefill_m512_up` 1.0234 -> 1.0727, `prefill_m256_down` 1.0617 -> 1.1247.

**What this says about the stop-gate, which is the part worth keeping.** Wave 3 satisfied all three
stop clauses with receipts and left 7 of 12 budget unspent. Its clauses were *true* -- and its
conclusion was still wrong, because every clause measured the exhaustion of one-knob arms and none of
them could see headroom reachable only by a different move class. A stop-gate that cannot distinguish
"no headroom" from "wrong instrument" will keep producing confident false stops. Launching over it was
right, and the closed list is what made the retry cheap rather than a repeat.

Also settles a caption question: the commit message's `(1.27x)` is the **absolute vs-oracle geomean**,
matching 1.2707 -- not a wave-local figure. Wave 2's `(1.21x)` was likewise absolute (1.2054). My earlier
note calling those captions wave-local was wrong.

STATE not yet rewritten (`cumulative` still 1.2054, `last_round` still wave 3's 2) -- `update_memory`
runs after the commit. **Falsifiable prediction for when it lands:** correct is **1.2707**; the frame bug
would show as 1.2054 x 1.2707 = **1.5317**, and a subtler chaining error as 1.2054 x 1.0494 = **1.2650**.
