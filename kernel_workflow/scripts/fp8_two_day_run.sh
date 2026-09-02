#!/usr/bin/env bash
# Run a lane for a fixed WALL-CLOCK budget, in waves, unattended.
#
# Why this exists
# ---------------
# kernel_lane.js used to take `deadline_epoch` / `no_stop_s` / `max_forced_replans`. That
# subsystem was removed when the launcher was decoupled, and those keys are now HARD ERRORS --
# they are not in KNOWN_ARGS. The lane's only stop conditions left are `dispatched >= budget` and
# `deep_explore_exhausted`, neither of which is a clock. So "run this for two days" has to live
# outside the lane, and this is where it lives.
#
# What it does, in order:
#   1. Refuses to start unless `lane_args.py --check` passes. A lane file that would be rejected
#      after two hours of setup is rejected now instead.
#   2. Refuses to start unless the agent runtime can actually reach its API. This is not
#      defensive padding: the bf16 lane spent its last nine hours retrying
#      `401 {"errorCode":"Primus.00009","errorMessage":"Unavailable"}` against the AMD Primus
#      gateway, holding six GPU locks and producing nothing, because a dead credential looks
#      exactly like a slow agent from the outside. Fail loudly in the first minute instead.
#   3. Records the deadline ONCE, in the state dir. A restart resumes the SAME two days rather
#      than starting a fresh two -- otherwise "two days" silently becomes "two days per crash".
#   4. Runs waves until the deadline. Each wave after the first is a CONTINUATION: the lane finds
#      <state_dir>/best on disk and resumes instead of reseeding from the original kernel.
#   5. Stops at the deadline. A wave still running is terminated; committed rounds are safe in
#      git and in STATE.json, so at most the in-flight round is lost.
#
# Usage:
#   fp8_two_day_run.sh <lane.json> [hours]        # hours defaults to 48
#   DRY_RUN=1 fp8_two_day_run.sh <lane.json>      # run every precondition, launch nothing

set -euo pipefail

LANE_FILE="${1:?usage: fp8_two_day_run.sh <lane.json> [hours]}"
HOURS="${2:-48}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANE_ARGS="${SCRIPT_DIR}/lane_args.py"

command -v python3 >/dev/null || { echo "FATAL: python3 not on PATH"; exit 1; }

# Find claude ourselves rather than trusting the caller's PATH. It lives in ~/.local/bin, which
# only a LOGIN shell adds, so this script ran fine from `bash -lc` and died instantly from tmux's
# default command, a piped `bash -s`, or a systemd-style launch -- all of which are how an
# unattended two-day daemon actually gets started. Measured 2026-08-27: three launch attempts
# exited on "claude CLI not on PATH" while the identical command worked in the foreground. A
# behaviour that depends on who invoked it is the same fault as a build that depends on who
# exported PYTORCH_ROCM_ARCH, and it gets the same fix.
if ! command -v claude >/dev/null; then
    for _cand in "$HOME/.local/bin" /usr/local/bin /opt/claude/bin; do
        if [ -x "$_cand/claude" ]; then
            PATH="$_cand:$PATH"
            export PATH
            break
        fi
    done
fi
command -v claude >/dev/null || {
    echo "FATAL: claude CLI not found on PATH nor in \$HOME/.local/bin, /usr/local/bin, /opt/claude/bin"
    exit 1
}
[ -f "$LANE_FILE" ] || { echo "FATAL: no such lane file: $LANE_FILE"; exit 1; }

STATE_DIR="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("state_dir",""))' "$LANE_FILE")"
[ -n "$STATE_DIR" ] || { echo "FATAL: $LANE_FILE has no state_dir; a wave could not be a continuation"; exit 1; }
REPO_ROOT="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["workflow_dir"].rsplit("/",1)[0])' "$LANE_FILE")"
LANE_NAME="$(basename "$LANE_FILE" .json)"
PROGRESS="${REPO_ROOT}/PIPELINE_PROGRESS_$(echo "$LANE_NAME" | tr '[:lower:]' '[:upper:]').md"

mkdir -p "$STATE_DIR"
LOG="${STATE_DIR}/supervisor.log"
DEADLINE_FILE="${STATE_DIR}/.wall_clock_deadline"

say() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }

# ---- 1. the lane file must be launchable -------------------------------------------------------
say "checking lane file: $LANE_FILE"
if ! python3 "$LANE_ARGS" --check "$LANE_FILE" >>"$LOG" 2>&1; then
    say "REFUSED: lane_args.py --check failed. Fix the file; do not bypass it."
    python3 "$LANE_ARGS" --check "$LANE_FILE" || true
    exit 3
fi
say "lane file OK"

# ---- 2. the agent runtime must be able to reach its API ----------------------------------------
# One real round trip through the exact path a wave uses. Cheaper than any alternative and the
# only check that catches an expired gateway credential.
say "checking agent API reachability (one tiny call)"
API_OUT="$(timeout 120 claude --dangerously-skip-permissions -p 'Reply with exactly: PONG' 2>&1 || true)"
if ! printf '%s' "$API_OUT" | grep -q 'PONG'; then
    say "REFUSED: the agent runtime cannot reach its API, so a two-day run would spend two days retrying."
    say "  response: $(printf '%s' "$API_OUT" | tr '\n' ' ' | cut -c1-300)"
    say "  This is an operator fix, not a pipeline fix: renew the credential in ~/.claude/settings.json"
    say "  (ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL) or restore access to the gateway, then re-run."
    exit 4
fi
say "agent API OK"

# ---- 3. the deadline is recorded once ----------------------------------------------------------
NOW="$(date -u +%s)"
if [ -f "$DEADLINE_FILE" ]; then
    DEADLINE="$(cat "$DEADLINE_FILE")"
    say "resuming an EXISTING budget: deadline $(date -u -d "@$DEADLINE" +%Y-%m-%dT%H:%M:%SZ) \
($(( (DEADLINE - NOW) / 3600 ))h left of the original ${HOURS}h)"
    if [ "$NOW" -ge "$DEADLINE" ]; then
        say "the recorded budget is already spent. Delete $DEADLINE_FILE to start a new one deliberately."
        exit 0
    fi
else
    DEADLINE=$(( NOW + HOURS * 3600 ))
    # A dry run must not start the clock. Writing the file here would mean a rehearsal silently
    # spent part of the budget it was rehearsing for.
    if [ "${DRY_RUN:-0}" = "1" ]; then
        say "would record a NEW ${HOURS}h budget, deadline \
$(date -u -d "@$DEADLINE" +%Y-%m-%dT%H:%M:%SZ) (not written: DRY_RUN)"
    else
        echo "$DEADLINE" > "$DEADLINE_FILE"
        say "recorded a NEW ${HOURS}h budget, deadline $(date -u -d "@$DEADLINE" +%Y-%m-%dT%H:%M:%SZ)"
    fi
fi

WAVE_PROMPT="$(cat <<PROMPT
发起一波 ${LANE_NAME}。

1. 先跑 python3 ${LANE_ARGS} --check ${LANE_FILE} —— 必须 exit 0。不是 0 就停下来修文件，不要绕过。
2. 再用 python3 ${LANE_ARGS} --print ${LANE_FILE} 渲染出 Workflow(...) 调用。
3. 原样调用它。不要手抄参数，不要增删任何键。
4. **等它跑完再结束你这一轮。** 不要把 Workflow 丢到后台然后回报「已启动」就收工 ——
   一波就是一次完整的 lane 运行，几小时量级。它没结束你就没结束。

这个 task 的两条规矩，违反了这一波就作废：
- examples/tasks/dense_fp8_gemm_rowwise/oracle_wrapper.py 是不可变 oracle，是所有 speedup 的分母。
  永远不要改它。每次 candidate_policy_scan.py 都要用 --immutable 把它豁免掉。
- 候选实现里禁止 rocBLAS / hipBLAS / hipBLASLt / Tensile / Composable Kernel / MIOpen / aiter。
  rocWMMA 是 header-only，允许。

没人值守。别停下来等我确认；有障碍就自己解决或者换条路走。这一波的结果追加到 ${PROGRESS}。
PROMPT
)"

# ---- 4. waves until the deadline ---------------------------------------------------------------
if [ "${DRY_RUN:-0}" = "1" ]; then
    say "DRY_RUN=1: all preconditions passed, launching nothing."
    say "  progress file would be: $PROGRESS"
    say "  wave prompt:"; printf '%s\n' "$WAVE_PROMPT" | sed 's/^/    /' | tee -a "$LOG"
    exit 0
fi

# `claude -p` treats a Workflow as a BACKGROUND task and kills it at a 600s ceiling, then exits
# rc=0 -- "finished cleanly". Measured 2026-08-27: wave 1 ran 11 minutes, was cut off in the
# lane's Setup phase ("Background tasks still running after 600s; terminating"), reported success,
# and the loop started wave 2 with a SECOND Workflow against the same state_dir. Left alone that
# is a cold restart every 11 minutes for two days, never reaching the optimize loop, with
# concurrent writers to one state dir -- the exact "start over" failure the protocol forbids,
# dressed up as a clean run. 0 means wait for the workflow instead of the ceiling.
export CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0

# Defence in depth for the same class of bug. A real wave runs for hours: it builds a COMMANDMENT,
# profiles, dispatches engineers, verifies. An rc=0 wave that lasted minutes did NOT do that, and
# whatever ended it early will end the next one early too, so looping is how a broken run burns two
# days looking busy. Treat it as a failure so the three-strike stop catches it.
MIN_WAVE_S="${MIN_WAVE_S:-1800}"

wave=0
while :; do
    NOW="$(date -u +%s)"
    LEFT=$(( DEADLINE - NOW ))
    if [ "$LEFT" -le 0 ]; then
        say "DEADLINE REACHED after ${wave} wave(s). Stopping."
        break
    fi
    # Under ~20 minutes there is not enough room for a wave to reach its first commit, and a wave
    # killed before that banks nothing while still paying full setup cost.
    if [ "$LEFT" -lt 1200 ]; then
        say "only $(( LEFT / 60 ))min left -- too short for a wave to bank a round. Stopping."
        break
    fi
    wave=$(( wave + 1 ))
    say "wave ${wave} starting ($(( LEFT / 3600 ))h $(( (LEFT % 3600) / 60 ))min left)"
    # `timeout` is the hard stop: a wave may not outlive the budget.
    wave_t0="$(date -u +%s)"
    set +e
    timeout --signal=TERM --kill-after=120 "$LEFT" \
        claude --dangerously-skip-permissions --effort max -p "$WAVE_PROMPT" \
        >>"${STATE_DIR}/wave_${wave}.log" 2>&1
    rc=$?
    set -e
    wave_s=$(( $(date -u +%s) - wave_t0 ))
    case "$rc" in
        0)   say "wave ${wave} finished cleanly after $(( wave_s / 60 ))min" ;;
        124) say "wave ${wave} was cut off by the deadline after $(( wave_s / 60 ))min" ;;
        *)   say "wave ${wave} exited rc=${rc} after $(( wave_s / 60 ))min -- see ${STATE_DIR}/wave_${wave}.log" ;;
    esac
    if [ "$rc" -eq 0 ] && [ "$wave_s" -lt "$MIN_WAVE_S" ]; then
        say "  but $(( wave_s / 60 ))min is too short to be a real wave (< $(( MIN_WAVE_S / 60 ))min)."
        say "  Something ended it early while reporting success. Check the tail of"
        say "  ${STATE_DIR}/wave_${wave}.log for a background-task ceiling, a refused lane file,"
        say "  or an agent that launched the workflow and returned instead of waiting."
        rc=90   # counts as a failure below, so three of these stop the run
    fi
    # A wave that dies in seconds is failing, not working. Three of those in a row and the loop is
    # burning the budget on a broken precondition rather than on search.
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ]; then
        fails=$(( ${fails:-0} + 1 ))
        if [ "$fails" -ge 3 ]; then
            say "three consecutive failed waves. Stopping rather than spending the rest of the budget."
            break
        fi
        sleep 60
    else
        fails=0
    fi
done

say "run over after ${wave} wave(s). State: ${STATE_DIR}. Progress: ${PROGRESS}"
