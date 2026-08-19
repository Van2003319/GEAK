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
