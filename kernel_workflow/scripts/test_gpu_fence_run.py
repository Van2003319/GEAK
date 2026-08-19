#!/usr/bin/env python3
"""Behavioural tests for `gpu_fence_run.sh` (finding 113).

The claim under test is narrow and load-bearing: when the payload exits, nothing
it started is still running, and nothing OUTSIDE what it started is ever
signalled. Both halves need proving. A reaper that misses orphans leaves GPU work
behind a released lock, which is the defect; a reaper that over-reaches kills the
caller's own process tree, which is worse and is explicitly forbidden here.

Nothing in this file touches a GPU. The payloads are `sleep`, because what is
being tested is process-tree containment, not anything about HIP.
"""
from __future__ import annotations

import os
import subprocess
import time
import unittest
from pathlib import Path

FENCE = Path(__file__).resolve().parent / "gpu_fence_run.sh"


def run_fence(script: str, timeout: int = 60, env: dict | None = None
              ) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e.setdefault("GEAK_FENCE_GRACE_SECONDS", "3")
    if env:
        e.update(env)
    return subprocess.run(["bash", str(FENCE), "bash", "-c", script],
                          capture_output=True, text=True, timeout=timeout, env=e)


def pids_matching(marker: str) -> list[int]:
    out = subprocess.run(["ps", "-e", "-o", "pid=,stat=,args="],
                         capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or marker not in parts[2]:
            continue
        if parts[1].startswith("Z"):   # a zombie is not running work
            continue
        if "ps -e -o" in parts[2]:
            continue
        pids.append(int(parts[0]))
    return pids


class FenceTest(unittest.TestCase):
    def test_the_script_is_syntactically_valid(self):
        proc = subprocess.run(["bash", "-n", str(FENCE)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_no_arguments_is_an_error_not_a_silent_success(self):
        proc = subprocess.run(["bash", str(FENCE)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("usage", proc.stderr)

    def test_the_payloads_exit_status_is_preserved(self):
        self.assertEqual(run_fence("exit 0").returncode, 0)
        self.assertEqual(run_fence("exit 17").returncode, 17)

    def test_a_clean_payload_produces_no_warning(self):
        proc = run_fence("echo hello")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("hello", proc.stdout)
        self.assertNotIn("WARNING", proc.stderr)

    def test_an_orphan_that_outlives_the_payload_is_reaped(self):
        """The defect, reproduced: a payload that backgrounds GPU-shaped work and
        returns immediately. Before this script that process kept running with no
        holder on any lock file."""
        marker = f"geak_fence_orphan_{os.getpid()}"
        proc = run_fence(f"sleep 300 & exec -a {marker} sleep 300 & echo started")
        self.assertIn("started", proc.stdout)
        self.assertIn("WARNING", proc.stderr)
        self.assertIn("113", proc.stderr)
        # The reap must be complete by the time the script returns -- that is the
        # whole point, since the flock is released the moment it does.
        leftover = pids_matching(marker)
        self.addCleanup(lambda: [os.kill(p, 9) for p in pids_matching(marker)])
        self.assertEqual(leftover, [],
                         f"orphan {marker} survived the fence: {leftover}")

    def test_a_grandchild_is_reaped_too_not_just_a_direct_child(self):
        """Orphans in this tree are grandchildren: task_runner spawns a build or a
        profile, which spawns the process that actually holds the device."""
        marker = f"geak_fence_grandchild_{os.getpid()}"
        proc = run_fence(
            f"bash -c '(exec -a {marker} sleep 300 &) ; sleep 0.2' ; echo done")
        self.assertIn("done", proc.stdout)
        leftover = pids_matching(marker)
        self.addCleanup(lambda: [os.kill(p, 9) for p in pids_matching(marker)])
        self.assertEqual(leftover, [],
                         f"grandchild {marker} survived the fence: {leftover}")

    def test_a_payload_ignoring_sigterm_is_killed_within_the_grace_period(self):
        marker = f"geak_fence_stubborn_{os.getpid()}"
        started = time.time()
        proc = run_fence(
            "trap '' TERM; "
            f"(trap '' TERM; exec -a {marker} sleep 300) & echo armed; sleep 0.3",
            timeout=90, env={"GEAK_FENCE_GRACE_SECONDS": "2"})
        elapsed = time.time() - started
        self.assertIn("armed", proc.stdout)
        leftover = pids_matching(marker)
        self.addCleanup(lambda: [os.kill(p, 9) for p in pids_matching(marker)])
        self.assertEqual(leftover, [], f"stubborn {marker} survived: {leftover}")
        self.assertLess(elapsed, 30,
                        "the grace period is not bounded; a hung payload would "
                        "hold the lock indefinitely, which is the failure mode "
                        "this script exists to avoid, relocated")

    def test_it_never_signals_a_process_outside_the_payloads_group(self):
        """The over-reach half. A sibling started by THIS test -- i.e. in the
        caller's tree, exactly where the standing rule says teardown must never
        reach -- must be untouched after the fence drains an orphan of its own."""
        marker = f"geak_fence_bystander_{os.getpid()}"
        bystander = subprocess.Popen(["bash", "-c", f"exec -a {marker} sleep 30"])
        self.addCleanup(lambda: (bystander.kill(), bystander.wait()))
        try:
            time.sleep(0.3)
            self.assertTrue(pids_matching(marker), "bystander failed to start")
            run_fence("sleep 300 & echo started")
            self.assertIsNone(bystander.poll(),
                              "the fence killed a process in the CALLER's tree -- "
                              "it drained a group it did not create")
            self.assertTrue(pids_matching(marker),
                            "the bystander was signalled by the drain")
        finally:
            bystander.kill()
            bystander.wait()

    def test_a_zombie_in_the_group_does_not_make_the_drain_hang(self):
        """PID 1 here is `sleep infinity`, which never reaps, so orphaned
        grandchildren become permanent zombies. `kill -0` and a bare `pgrep` both
        report a zombie as present, so the naive drain loop would never terminate
        against a process that has already exited and cannot exit again. This is
        the same trap as polling a pid to decide whether a job finished."""
        started = time.time()
        proc = run_fence(
            # A child that exits while its parent never waits: the classic zombie
            # producer. The inner bash sleeps past its child's death.
            "bash -c 'sleep 0.1 & sleep 5' & sleep 0.3; echo armed",
            timeout=60)
        elapsed = time.time() - started
        self.assertIn("armed", proc.stdout)
        self.assertLess(elapsed, 40,
                        "the drain did not terminate promptly; a zombie is being "
                        "counted as a live process")


class RolePromptWiringTest(unittest.TestCase):
    """A tested reaper that nothing invokes is a comment. Finding (113) is only
    closed once every documented GPU invocation actually routes through it."""

    ROLES = FENCE.parent.parent / "roles"

    def role_lines_invoking_the_lock(self):
        for path in sorted(self.ROLES.glob("*.md")):
            for i, line in enumerate(path.read_text().splitlines(), 1):
                if "gpu_lock.sh $GPU_ID" in line:
                    yield path.name, i, line

    def test_the_fence_script_exists_and_is_executable_by_bash(self):
        self.assertTrue(FENCE.is_file(), f"{FENCE} is missing")

    def test_every_documented_gpu_lock_invocation_goes_through_the_fence(self):
        found = 0
        for name, lineno, line in self.role_lines_invoking_the_lock():
            found += 1
            with self.subTest(role=name, line=lineno):
                after = line.split("gpu_lock.sh $GPU_ID", 1)[1].lstrip()
                self.assertTrue(
                    after.startswith("bash $SKILL_DIR/scripts/gpu_fence_run.sh"),
                    f"{name}:{lineno} runs a payload under the lock without the fence, so a "
                    f"child it spawns can outlive the flock:\n  {line.strip()}")
        self.assertGreater(found, 0, "no role documents a gpu_lock.sh invocation any more")

    def test_the_commandment_template_tells_the_author_to_use_it(self):
        bench = (self.ROLES / "benchmark_engineer.md").read_text()
        self.assertIn("gpu_fence_run.sh", bench)
        self.assertIn("113", bench,
                      "the reason is not recorded next to the instruction, so the next "
                      "person to simplify the command will simplify it back out")


if __name__ == "__main__":
    unittest.main()
