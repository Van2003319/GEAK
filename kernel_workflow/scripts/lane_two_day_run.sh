#!/usr/bin/env bash
# Run ANY lane for a fixed WALL-CLOCK budget, in waves, unattended.
#
# Relationship to fp8_two_day_run.sh
# ----------------------------------
# This is that script with the two task-specific rules lifted out of the prompt. Those rules --
# which file is the immutable oracle, which libraries candidates may not link -- are properties of
# the TASK, not of the launcher, so they belong in the lane file. A launcher that names one task's
# oracle cannot start another task without being edited, and a launcher that gets edited per task
# is how the fp8 lane and the bf16 lane end up running under two slightly different protocols
# that nobody diffed. Read them from `_wave_rules` in the lane file instead.
#
# fp8_two_day_run.sh is left in place and unchanged, because it is the launcher that the fp8 lane
# has actually been started with. Once a wave of this one has run for real, that one is redundant.
#
# What it does, in order:
#   1. Refuses to start unless `lane_args.py --check` passes. A lane file that would be rejected
#      after two hours of setup is rejected now instead.
#   2. Refuses to start unless the agent runtime can actually reach its API. This is not
#      defensive padding: the bf16 lane spent nine hours retrying
#      `401 {"errorCode":"Primus.00009","errorMessage":"Unavailable"}` against the AMD Primus
#      gateway, holding six GPU locks and producing nothing, because a dead credential looks
#      exactly like a slow agent from the outside. Fail loudly in the first minute instead.
#   3. Records the deadline ONCE, in the state dir. A restart resumes the SAME budget rather
#      than starting a fresh one -- otherwise "two days" silently becomes "two days per crash".
#   4. Runs waves until the EARLIER of that deadline and the machine's own expiry.
#   5. Stops there. A wave still running is terminated; committed rounds are safe in git and in
#      STATE.json, so at most the in-flight round is lost.
#
# Usage:
#   lane_two_day_run.sh <lane.json> [hours]                 # hours defaults to 48
#   DRY_RUN=1 lane_two_day_run.sh <lane.json>               # run every precondition, launch nothing
#   MACHINE_DEADLINE_EPOCH=<epoch> lane_two_day_run.sh ...  # see below; strongly recommended

set -euo pipefail

LANE_FILE="${1:?usage: lane_two_day_run.sh <lane.json> [hours]}"
HOURS="${2:-48}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANE_ARGS="${SCRIPT_DIR}/lane_args.py"

command -v python3 >/dev/null || { echo "FATAL: python3 not on PATH"; exit 1; }

# Find claude ourselves rather than trusting the caller's PATH. It lives in ~/.local/bin, which
# only a LOGIN shell adds, so this script ran fine from `bash -lc` and died instantly from tmux's
# default command, a piped `bash -s`, or a systemd-style launch -- all of which are how an
# unattended multi-day daemon actually gets started. Measured 2026-08-27: three launch attempts
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

# The task's own rules, from the lane file. Absent is allowed but says so out loud: a lane with no
# rules block is a lane where nothing stops a candidate from editing the oracle it is measured
# against, and that failure inflates every speedup afterwards without leaving a mark.
WAVE_RULES="$(python3 - "$LANE_FILE" <<"PY"
import json, sys
rules = json.load(open(sys.argv[1])).get("_wave_rules") or []
if isinstance(rules, str):
    rules = [rules]
for r in rules:
    print("- " + str(r).replace("\n", "\n  "))
PY
)"

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

if [ -z "$WAVE_RULES" ]; then
    say "NOTE: $LANE_FILE has no _wave_rules. The wave prompt will carry no task rules of its own,"
    say "      so the lane's task string is all the agent will be told about the oracle and the"
    say "      library ban. That is allowed, but it is a weaker guard than the fp8 lane runs with."
fi

# ---- 2. the agent runtime must be able to reach its API ----------------------------------------
# One real round trip through the exact path a wave uses. Cheaper than any alternative and the
# only check that catches an expired gateway credential.
api_ok() {
    local out
    out="$(timeout 120 claude --dangerously-skip-permissions -p 'Reply with exactly: PONG' 2>&1 || true)"
    API_OUT="$out"
    printf '%s' "$out" | grep -q 'PONG'
}

say "checking agent API reachability (one tiny call)"
if ! api_ok; then
    say "REFUSED: the agent runtime cannot reach its API, so the run would spend its budget retrying."
    say "  response: $(printf '%s' "$API_OUT" | tr '\n' ' ' | cut -c1-300)"
    say "  This is an operator fix, not a pipeline fix: renew the credential in ~/.claude/settings.json"
    say "  (ANTHROPIC_AUTH_TOKEN / ANTHROPIC_BASE_URL) or restore access to the gateway, then re-run."
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

# ---- 3b. the machine's own expiry, which is NOT the same clock ---------------------------------
# The recorded budget says how long the SEARCH may run. The SLURM allocation says how long the
# MACHINE exists. fp8_two_day_run.sh only ever knew the first one, and the gap is not theoretical:
# on 2026-08-27 the recorded budget ran to 08-29 07:16Z while job 1033 on tw008 expired at
# 08-27 10:39Z. Launching against the budget alone would have opened a multi-hour wave with under
# an hour of machine left -- full setup cost, six GPU locks, and nothing banked, reported as a
# deadline cut rather than as a wave that should never have started.
#
# The container has no SLURM client, so the epoch has to be passed in. Whoever launches this knows
# it: `scontrol show job <id>` gives EndTime. Absent, we say so rather than pretend.
MACHINE_DEADLINE="${MACHINE_DEADLINE_EPOCH:-}"
if [ -n "$MACHINE_DEADLINE" ]; then
    say "machine expires $(date -u -d "@$MACHINE_DEADLINE" +%Y-%m-%dT%H:%M:%SZ) \
($(( (MACHINE_DEADLINE - NOW) / 60 ))min from now)"
    if [ "$MACHINE_DEADLINE" -lt "$DEADLINE" ]; then
        say "  that is EARLIER than the recorded budget, so it is the binding stop for this run."
        EFFECTIVE_DEADLINE="$MACHINE_DEADLINE"
    else
        EFFECTIVE_DEADLINE="$DEADLINE"
    fi
else
    say "MACHINE_DEADLINE_EPOCH not set -- running against the recorded budget alone. On a SLURM box"
    say "  that is wrong whenever the allocation ends first; get the epoch on the submitting host"
    say "  with:  scontrol show job <id>  then  date -d <EndTime> +%s"
    EFFECTIVE_DEADLINE="$DEADLINE"
fi

WAVE_PROMPT="$(cat <<PROMPT
发起一波 ${LANE_NAME}。

1. 先跑 python3 ${LANE_ARGS} --check ${LANE_FILE} —— 必须 exit 0。不是 0 就停下来修文件，不要绕过。
2. 再用 python3 ${LANE_ARGS} --print ${LANE_FILE} 渲染出 Workflow(...) 调用。
3. 原样调用它。不要手抄参数，不要增删任何键。
4. **等它跑完再结束你这一轮。** 不要把 Workflow 丢到后台然后回报「已启动」就收工 ——
   一波就是一次完整的 lane 运行，几小时量级。它没结束你就没结束。

这个 task 的规矩，违反了这一波就作废：
${WAVE_RULES}

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
# is a cold restart every 11 minutes, never reaching the optimize loop, with concurrent writers to
# one state dir -- the exact "start over" failure the protocol forbids, dressed up as a clean run.
# 0 means wait for the workflow instead of the ceiling.
export CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0

# Defence in depth for the same class of bug. A real wave runs for hours: it builds a COMMANDMENT,
# profiles, dispatches engineers, verifies. An rc=0 wave that lasted minutes did NOT do that, and
# whatever ended it early will end the next one early too, so looping is how a broken run burns a
# budget looking busy. Treat it as a failure so the three-strike stop catches it.
MIN_WAVE_S="${MIN_WAVE_S:-1800}"

# How long to wait before re-probing the API after a wave died on a transport error, and how long
# to keep doing that before giving up. See the API_DOWN handling in the loop for why this is a
# separate policy from the three-strike counter.
API_WAIT_S="${API_WAIT_S:-600}"

# A wave that ended because the LINK died, rather than because the search failed, must not count
# toward the three-strike stop. Deciding which happened means reading the tail of the wave log:
# the launcher only sees an exit code, and `claude` reports "the gateway went away mid-request" and
# "your lane file is malformed" with the same rc=1.
#
# 2026-08-29 is why this exists, and it cost 1h31m of a 51h reservation. The tunnel to the AMD
# gateway dropped at ~05:20. Both lanes then did exactly this:
#     wave 1 exited rc=1 after 43min   <- ECONNRESET mid-wave
#     wave 2 exited rc=1 after 3min    <- ECONNRESET immediately
#     wave 3 exited rc=1 after 3min    <- ECONNRESET immediately
#     three consecutive failed waves. Stopping rather than spending the rest of the budget.
# Each wave_N.log contained one line: `API Error: Connection dropped (ECONNRESET)`. The lanes then
# sat in `sleep infinity` with 48h of machine left, and nothing would have restarted them: the
# controller's cmd_lanes_up keys on the tmux session, which run_lane.sh deliberately keeps alive.
#
# The asymmetry this fixes was already half-recognised in run_lane.sh, which retries rc=4 with the
# note that it is "the only exit code whose cause is external and self-healing". That was right
# about the cause and wrong about the uniqueness: rc=4 is what a dead link looks like BEFORE a wave
# opens, and rc=1-with-ECONNRESET is what the SAME dead link looks like AFTER. One got retried
# forever, the other stopped the run after three tries.
api_transport_failure() {
    local log="$1"
    [ -f "$log" ] || return 1
    # Only the tail: an ECONNRESET recorded hours ago in a wave that then recovered and ran for 40
    # minutes says nothing about why THIS wave ended.
    tail -5 "$log" 2>/dev/null | grep -qiE \
        'ECONNRESET|Connection dropped|Connection error|socket hang up|ETIMEDOUT|EAI_AGAIN|502 Bad Gateway|503 Service|Unable to connect to API|fetch failed'
}

wave=0
while :; do
    NOW="$(date -u +%s)"
    LEFT=$(( EFFECTIVE_DEADLINE - NOW ))
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
    # Probe the link before opening a wave, not just once at startup.
    #
    # The original check sat outside this loop, so it only ever proved the API was reachable at
    # LAUNCH. Every wave after that opened blind. With a dead tunnel that means paying the full
    # setup cost of a wave to discover, three minutes in, what a two-second probe would have said.
    # Worse, those blind waves are what fed the three-strike counter on 2026-08-29 and stopped two
    # lanes with 48h of machine left.
    if ! api_ok; then
        say "API unreachable before wave $(( wave + 1 )); NOT opening a wave. Retrying in $(( API_WAIT_S / 60 ))min."
        say "  probe said: $(printf '%s' "$API_OUT" | tr '\n' ' ' | cut -c1-200)"
        say "  This is the link, not the search. Nothing here is lost -- the budget clock is"
        say "  wall-clock, so a long outage costs time but never corrupts state. Check the reverse"
        say "  tunnel on the operator's machine (127.0.0.1:${TUNNEL_PROBE_PORT:-18443} on the jumpbox)."
        sleep "$API_WAIT_S"
        continue
    fi

    wave=$(( wave + 1 ))
    say "wave ${wave} starting ($(( LEFT / 3600 ))h $(( (LEFT % 3600) / 60 ))min left)"
    # `timeout` is the hard stop: a wave may not outlive the budget or the machine.
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
    # A dead LINK is not a failed SEARCH. Check this before the strike counter, because the whole
    # point is to keep a transport outage out of it. See api_transport_failure() above.
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] \
       && api_transport_failure "${STATE_DIR}/wave_${wave}.log"; then
        say "  wave ${wave} died on a TRANSPORT error, not a search failure:"
        say "    $(tail -2 "${STATE_DIR}/wave_${wave}.log" | tr '\n' ' ' | cut -c1-200)"
        say "  NOT counting this toward the three-strike stop. Waiting $(( API_WAIT_S / 60 ))min for the link."
        # The wave number is deliberately NOT rolled back: this wave really was dispatched and its
        # log is evidence. The next attempt gets a fresh wave_N.log rather than appending to this
        # one, which keeps "why did wave 7 fail" answerable.
        sleep "$API_WAIT_S"
        continue
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
