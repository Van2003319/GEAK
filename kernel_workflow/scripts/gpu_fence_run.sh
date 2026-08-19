#!/usr/bin/env bash
# Run a command so that it cannot outlive its own GPU lock.
#
# Finding (113). `gpu_lock.sh` holds a flock for as long as the locking process
# lives and releases it when that process exits. A child the command spawned can
# outlive it: when the b4b workflow died, the flock vanished and two
# `task_runner.py profile` processes went on using GPUs 2 and 4 for another
# 29 and 96 minutes with NO holder on any lock file. The next measurement to
# land on those dies was competing with work that, as far as every lock in the
# system was concerned, did not exist. A lock released on parent death does not
# fence work that survives parent death.
#
# `gpu_lock.sh` must not be edited, and it should not be: the hole is not in the
# locking, it is in the assumption that the command tree ends when the command
# does. So the fix goes in the command slot instead. Wrap the payload:
#
#   gpu_lock.sh 2,3 bash kernel_workflow/scripts/gpu_fence_run.sh <cmd...>
#
# and the flock is still held while this script makes sure nothing the payload
# started is still running.
#
# Mechanism
# ---------
# The payload runs via `setsid` in a NEW process group, so "everything the
# payload started" is a group ID we own and did not inherit. After the direct
# child exits we drain that group: TERM, grace, KILL. Nothing outside the group
# is ever signalled -- the group is created here, so it cannot contain the
# caller, the workflow, or a sibling lane.
#
# Zombies
# -------
# PID 1 in this container is `sleep infinity`, which never reaps, so orphaned
# grandchildren become permanent zombies. A zombie satisfies `kill -0` and shows
# up in `pgrep`, so the obvious "wait until the group is empty" loop would spin
# forever against processes that have already exited and cannot exit again. The
# drain therefore reads `ps -o stat=` and counts only processes NOT in state Z.
#
# Exit status is the payload's, unless the drain had to kill survivors, which is
# reported on stderr and does not change the status: the payload's own result is
# still the result. Survivors are logged loudly because a payload that leaves GPU
# work behind is a defect in the payload, and silently cleaning up after it would
# hide the thing worth fixing.
set -uo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: gpu_fence_run.sh <command> [args...]" >&2
  exit 2
fi

GRACE_SECONDS="${GEAK_FENCE_GRACE_SECONDS:-10}"

# Processes in $1's group that are alive in the scheduler's sense. A zombie has
# been reaped by nothing but has run its last instruction; counting it would make
# this loop non-terminating in exactly the environment it has to work in.
live_in_group() {
  local pgid="$1" pid stat n=0
  while read -r pid stat; do
    [ -n "$pid" ] || continue
    [ "$pid" = "$$" ] && continue
    case "$stat" in Z*) continue ;; esac
    n=$((n + 1))
  done < <(ps -e -o pgid=,pid=,stat= 2>/dev/null |
           awk -v g="$pgid" '$1 == g { print $2, $3 }')
  printf '%s\n' "$n"
}

# Job control (`set -m`) is what puts the payload in a new process group: with
# it, bash makes each background job a group leader, so the group ID IS the
# child's pid, by construction and without a lookup.
#
# The first version used `setsid` and read the group back with `ps`. That is the
# race this script exists to close, in miniature: `setsid` forks and exits
# immediately, so `$!` is a process that is often already gone by the time `ps`
# runs, the lookup returns empty, the drain refuses -- and the orphan survives
# while the log says the fence ran. Its own test caught it, which is the only
# reason it is not still in here.
set -m
"$@" &
child=$!
set +m
pgid="$child"

wait "$child"
status=$?

own_pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"
if [ -z "$pgid" ] || [ "$pgid" = "0" ] || [ "$pgid" = "1" ] || [ "$pgid" = "$own_pgid" ]; then
  # Refuse to drain anything we did not demonstrably create. This is the
  # fail-closed direction for a signal: leaving an orphan is bad, signalling the
  # caller's own tree is worse, and the user's standing rule is that the bench
  # teardown must never reach the caller's process tree.
  echo "gpu_fence_run: payload group not identifiable (pgid='${pgid}', own='${own_pgid}'); " \
       "NOT draining. Orphans, if any, are still running." >&2
  exit "$status"
fi

remaining="$(live_in_group "$pgid")"
if [ "$remaining" -gt 0 ]; then
  echo "gpu_fence_run: WARNING -- ${remaining} process(es) outlived the payload in group ${pgid}." \
       "This is finding (113): they would hold GPU memory with no lock holder. Terminating." >&2
  ps -e -o pgid=,pid=,stat=,etime=,args= 2>/dev/null | awk -v g="$pgid" '$1 == g' >&2
  kill -TERM -- "-${pgid}" 2>/dev/null
  waited=0
  while [ "$waited" -lt "$GRACE_SECONDS" ]; do
    [ "$(live_in_group "$pgid")" -eq 0 ] && break
    sleep 1
    waited=$((waited + 1))
  done
  if [ "$(live_in_group "$pgid")" -gt 0 ]; then
    echo "gpu_fence_run: group ${pgid} ignored SIGTERM for ${GRACE_SECONDS}s; sending SIGKILL." >&2
    kill -KILL -- "-${pgid}" 2>/dev/null
    sleep 1
  fi
  still="$(live_in_group "$pgid")"
  if [ "$still" -gt 0 ]; then
    echo "gpu_fence_run: ${still} process(es) survived SIGKILL in group ${pgid} -- almost certainly" \
         "stuck in an uninterruptible GPU wait. The lock is about to be released with work still on" \
         "the device; DO NOT trust the next measurement on this GPU." >&2
    exit 70
  fi
  echo "gpu_fence_run: group ${pgid} drained." >&2
fi

exit "$status"
