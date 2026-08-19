#!/usr/bin/env python3
"""Behavioural tests for `gpu_lock.sh`, the single chokepoint for GPU work.

Every compile / correctness / benchmark / profile command in the kernel lane is
supposed to go through this wrapper, and until now nothing executed it. The one
test that mentioned it (`test_gpu_fence_run.py`) checks that role prompts *spell
the invocation correctly*, which is a different claim entirely -- a perfectly
worded call into a broken fence still measures on a contaminated card.

`gpu_lock.sh` is frozen: it is not to be edited. That makes a test more valuable
rather than less, because the only way a regression can enter is from
underneath it -- and it also fixes the shape of this file. Everything here
either observes the wrapper from outside (exit status, message, environment
handed to the payload, lock contention) or pins a default at the source level.

The wrapper is exercised through a copy whose `LOCK_DIR` literal is repointed at
a temp directory -- one substitution, asserted to be exactly one, with the
original literal pinned separately below. Running against the live
`/tmp/team_gpu_locks` was the first version and it was wrong twice over: the
tests would take real lane locks out from under a running measurement, and on
this host the directory is `root:root drwxrwxr-x`, so an unprivileged run cannot
create a lock file at all. (That last one is not a test artifact -- see
`LockDirectoryIsUsableTest`.)

Two properties are deliberately left unpinned, and saying so here is the point
of writing it down:

  * The DRM-node mapping `renderD$((128 + 8 * id))`. On this host the render
    nodes are `renderD128..renderD135`, i.e. consecutive, so that arithmetic
    addresses the wrong node for every id but 0. A test asserting the current
    formula would cement the bug as the contract.
  * `_gpu_is_idle` fails OPEN: an unreadable or absent counter returns "idle".
    That is why the ids used below are 90+, which no card claims -- the probe
    reports them idle without any GPU being touched. It is a real exposure, not
    a property, and it is recorded rather than asserted.

No test here touches a GPU. Payloads are `echo` and `sleep`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = HERE / "gpu_lock.sh"
REAL_LOCK_DIR = Path("/tmp/team_gpu_locks")
LOCK_DIR_LITERAL = 'LOCK_DIR="/tmp/team_gpu_locks"'

# Well outside any real device id, so even a mis-stubbed run cannot address a
# card, and the idleness probe finds no sysfs node.
FREE_A, FREE_B, FREE_C = "94", "95", "96"


class GpuLockHarness(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gpulock_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.lock_dir = self.tmp / "locks"
        src = LOCK.read_text(encoding="utf-8")
        self.assertEqual(1, src.count(LOCK_DIR_LITERAL),
                         "the lock directory is no longer a single literal; this "
                         "harness would be staging a script that still writes to the "
                         "live one")
        self.script = self.tmp / "gpu_lock.sh"
        self.script.write_text(
            src.replace(LOCK_DIR_LITERAL, f'LOCK_DIR="{self.lock_dir}"'),
            encoding="utf-8")
        self.log = self.tmp / "use.jsonl"

    def run_lock(self, spec: str, *payload: str, env: dict[str, str] | None = None,
                 cwd: Path | None = None, timeout: int = 60):
        environ = dict(os.environ)
        for var in ("GEAK_GPU_ALLOWED", "GEAK_GPU_USE_LOG", "PYTORCH_ROCM_ARCH",
                    "TORCH_EXTENSIONS_DIR"):
            environ.pop(var, None)
        environ["KERNEL_ENV_SKIP_ENUM_REAP"] = "1"
        environ.update(env or {})
        return subprocess.run(["bash", str(self.script), spec, *payload],
                              capture_output=True, text=True, env=environ,
                              cwd=str(cwd) if cwd else None, timeout=timeout)

    def hold(self, gpu: str, seconds: float) -> subprocess.Popen:
        """Hold a lane's flock from outside, the way a foreign engineer would."""
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        path = self.lock_dir / f"gpu_{gpu}.lock"
        path.touch()
        proc = subprocess.Popen(["flock", "-x", str(path), "sleep", str(seconds)])
        self.addCleanup(proc.wait)
        self.addCleanup(proc.terminate)
        time.sleep(0.4)  # let flock actually take it before the test races it
        return proc

    def entries(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(l) for l in self.log.read_text().splitlines() if l.strip()]


class AllocationFenceTest(GpuLockHarness):
    """`GEAK_GPU_ALLOWED` answers "is this GPU MINE?", which idleness cannot."""

    def test_a_gpu_outside_the_allocation_is_refused(self):
        proc = self.run_lock(FREE_A, "echo", "ran", env={"GEAK_GPU_ALLOWED": "0,1"})
        self.assertEqual(1, proc.returncode)
        self.assertNotIn("ran", proc.stdout, "the payload executed anyway")
        self.assertIn(FREE_A, proc.stderr)
        self.assertIn("0,1", proc.stderr,
                      "the error must name the allocated set; an agent that cannot see "
                      "what it does own will guess again")

    def test_an_allocated_gpu_passes_the_fence(self):
        proc = self.run_lock(FREE_A, "echo", "ran",
                             env={"GEAK_GPU_ALLOWED": f"0,{FREE_A}"})
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("ran", proc.stdout)

    def test_one_unowned_member_refuses_the_whole_pool(self):
        """Silently intersecting would let a run labelled "2 GPUs" become a
        1-GPU run with the label intact."""
        proc = self.run_lock(f"{FREE_A},{FREE_B}", "echo", "ran",
                             env={"GEAK_GPU_ALLOWED": FREE_A})
        self.assertEqual(1, proc.returncode)
        self.assertIn(FREE_B, proc.stderr)
        self.assertNotIn("ran", proc.stdout)

    def test_an_unset_allocation_imposes_no_fence(self):
        proc = self.run_lock(FREE_A, "echo", "ran")
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("ran", proc.stdout)

    def test_the_membership_test_is_not_a_substring_test(self):
        """The comma padding in `case ",$ALLOWED," in *",$r,"*` is the whole
        defence: without it an allocation of GPU 1 admits GPU 91, and the fence
        reads as working right up until a two-digit id appears."""
        for allowed, asked in (("1", "91"), ("9", "94"), ("4", "94"), ("0,1", "10")):
            with self.subTest(allowed=allowed, asked=asked):
                proc = self.run_lock(asked, "echo", "ran",
                                     env={"GEAK_GPU_ALLOWED": allowed})
                self.assertEqual(1, proc.returncode,
                                 f"allocation [{allowed}] admitted GPU {asked}")
                self.assertNotIn("ran", proc.stdout)

    def test_a_pool_wholly_inside_the_allocation_passes(self):
        proc = self.run_lock(f"{FREE_A},{FREE_B}", "echo", "ran",
                             env={"GEAK_GPU_ALLOWED": f"{FREE_A},{FREE_B},7"})
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("ran", proc.stdout)


class EnvironmentHandoffTest(GpuLockHarness):
    def payload_env(self, spec: str, var: str, **kw) -> str:
        proc = self.run_lock(spec, "bash", "-c", f'printf "%s" "${{{var}:-UNSET}}"', **kw)
        self.assertEqual(0, proc.returncode, proc.stderr)
        return proc.stdout.strip()

    def test_the_payload_sees_the_chosen_gpu_in_hip_visible_devices(self):
        self.assertEqual(FREE_A, self.payload_env(FREE_A, "HIP_VISIBLE_DEVICES"))

    def test_pool_mode_reports_the_lane_it_actually_won(self):
        self.hold(FREE_A, 4)
        got = self.payload_env(f"{FREE_A},{FREE_B}", "HIP_VISIBLE_DEVICES")
        self.assertEqual(FREE_B, got,
                         "pool mode handed the payload a lane it does not hold")

    def test_the_build_cache_is_isolated_per_workspace(self):
        """Deriving it from `$PWD` is what stops one engineer benchmarking
        another's `.so` out of the single global torch extension cache."""
        workspace = self.tmp / "workspace"
        workspace.mkdir()
        got = self.payload_env(FREE_A, "TORCH_EXTENSIONS_DIR", cwd=workspace)
        self.assertEqual(str(workspace / ".torch_ext"), got)

    def test_a_caller_set_build_cache_is_honoured(self):
        mine = self.tmp / "caller_choice"
        got = self.payload_env(FREE_A, "TORCH_EXTENSIONS_DIR",
                               env={"TORCH_EXTENSIONS_DIR": str(mine)})
        self.assertEqual(str(mine), got)

    def test_keep_arch_leaves_the_callers_arch_list_alone(self):
        got = self.payload_env(FREE_A, "PYTORCH_ROCM_ARCH",
                               env={"KERNEL_ENV_KEEP_ARCH": "1",
                                    "PYTORCH_ROCM_ARCH": "gfx90a;gfx942"})
        self.assertEqual("gfx90a;gfx942", got)

    def test_the_payloads_exit_status_is_the_wrappers(self):
        """A wrapper that swallows a nonzero exit turns a failed correctness run
        into a passing one."""
        for code in (0, 1, 7):
            with self.subTest(code=code):
                proc = self.run_lock(FREE_A, "bash", "-c", f"exit {code}")
                self.assertEqual(code, proc.returncode, proc.stderr)


class MutualExclusionTest(GpuLockHarness):
    def test_a_held_lane_blocks_a_single_gpu_run(self):
        self.hold(FREE_C, 2.0)
        t0 = time.monotonic()
        proc = self.run_lock(FREE_C, "echo", "ran", timeout=30)
        elapsed = time.monotonic() - t0
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertGreater(elapsed, 1.0,
                           "the run started while another process held the lane")

    def test_pool_mode_steps_over_a_held_lane_instead_of_blocking(self):
        self.hold(FREE_A, 3.0)
        t0 = time.monotonic()
        proc = self.run_lock(f"{FREE_A},{FREE_B}", "echo", "ran", timeout=30)
        elapsed = time.monotonic() - t0
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertLess(elapsed, 2.0, "pool mode waited instead of stepping")

    def test_a_fully_held_pool_fails_loudly_at_the_deadline(self):
        self.hold(FREE_A, 5.0)
        self.hold(FREE_B, 5.0)
        proc = self.run_lock(f"{FREE_A},{FREE_B}", "echo", "ran", timeout=30,
                             env={"GEAK_GPU_POOL_WAIT": "1"})
        self.assertEqual(1, proc.returncode)
        self.assertIn("no free+idle GPU", proc.stderr)
        self.assertNotIn("ran", proc.stdout)


class UseLogTest(GpuLockHarness):
    """The append-only record of which lane was actually taken, and how long the
    acquisition cost -- the scheduler's price, which used to leave no trace."""

    def test_nothing_is_written_unless_a_log_is_named(self):
        self.run_lock(FREE_A, "echo", "ran")
        self.assertEqual([], self.entries())

    def test_a_pinned_acquisition_records_its_mode_gpu_and_wait(self):
        proc = self.run_lock(FREE_A, "echo", "ran",
                             env={"GEAK_GPU_USE_LOG": str(self.log)})
        self.assertEqual(0, proc.returncode, proc.stderr)
        rows = self.entries()
        self.assertEqual(1, len(rows), rows)
        self.assertEqual("pin", rows[0]["mode"])
        self.assertEqual(int(FREE_A), rows[0]["gpu"])
        self.assertIn("wait_s", rows[0])

    def test_a_pool_acquisition_records_the_lane_not_the_pool(self):
        self.hold(FREE_A, 4)
        spec = f"{FREE_A},{FREE_B}"
        proc = self.run_lock(spec, "echo", "ran",
                             env={"GEAK_GPU_USE_LOG": str(self.log)})
        self.assertEqual(0, proc.returncode, proc.stderr)
        rows = self.entries()
        self.assertEqual(1, len(rows), rows)
        self.assertEqual("pool", rows[0]["mode"])
        self.assertEqual(int(FREE_B), rows[0]["gpu"],
                         "the log records the requested pool rather than the lane won, "
                         "which is exactly the ambiguity it exists to remove")
        self.assertEqual(spec, rows[0]["pool"])

    def test_a_blocked_acquisition_records_the_time_it_waited(self):
        self.hold(FREE_A, 1.6)
        proc = self.run_lock(FREE_A, "echo", "ran", timeout=30,
                             env={"GEAK_GPU_USE_LOG": str(self.log)})
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertGreaterEqual(self.entries()[0]["wait_s"], 1,
                                "a wait that is always logged as 0 is worse than no "
                                "field: it makes pinning look free")


class FrozenDefaultsTest(unittest.TestCase):
    """Source-level pins on the decisions that make a measurement trustworthy.
    Behavioural coverage would need a fake sysfs root, which the file has no hook
    for and which is not to be added."""

    SRC = LOCK.read_text(encoding="utf-8")

    def test_idleness_is_required_by_default_on_both_paths(self):
        occurrences = self.SRC.count('"${GEAK_GPU_REQUIRE_IDLE:-1}" = "1"')
        self.assertEqual(2, occurrences,
                         "pool mode and pinned mode must both default to requiring an "
                         "idle card; the pinned path defaulted to 0 once, and two "
                         "measurement cells were timed on a card another tenant held")
        self.assertNotIn("GEAK_GPU_REQUIRE_IDLE:-0", self.SRC)

    def test_the_usage_line_names_no_concrete_gpu_ids(self):
        """It read "e.g. 0,1,2,3" once; agents copied that string verbatim into
        15 invocations from runs allocated neither GPU 2 nor 3. An example in a
        usage line is read as a default."""
        line = self.SRC.split("GPU_SPEC=", 1)[1].split("\n", 1)[0]
        message = line.split(":?", 1)[1]  # drop `"${1` -- that 1 is the argv slot
        for digit in "0123456789":
            with self.subTest(digit=digit):
                self.assertNotIn(digit, message, message)

    def test_the_allocation_fence_is_still_a_hard_error(self):
        fence = self.SRC.split("GEAK_GPU_ALLOWED:-", 1)[1] \
                        .split("# ---- GPU selection", 1)[0]
        self.assertIn("exit 1", fence,
                      "a fence that warns and continues still produces a number, and "
                      "the number is the thing that misleads")

    def test_the_lock_directory_is_shared_and_absolute(self):
        """A per-user or relative lock directory would give every engineer their
        own private mutex, which is indistinguishable from no mutex."""
        self.assertIn(LOCK_DIR_LITERAL, self.SRC)


# Lanes this host will not let the pipeline user lock, with the reason. An entry
# is a claim about the machine, not a silencer: the test below fails if a listed
# lane becomes usable, so the list cannot outlive the defect it describes.
#
# `/tmp/team_gpu_locks` is `root:root drwxrwxr-x` here, which has two
# consequences: no NEW lane file can be created at all, and the two lane files
# root happened to create first are not group-writable. `sudo` needs a password,
# so this is not fixable from inside the pipeline -- it is routed around, by
# keeping GEAK_GPU_ALLOWED inside the usable set.
HOST_UNUSABLE_LANES = {
    0: "gpu_0.lock is root:root mode 664 and this user is not in group root; it "
       "was created before the container snapshot by a privileged run",
    1: "gpu_1.lock is root:root mode 644, same origin, not writable by anyone else",
}


def kfd_gpu_count() -> int:
    """GPUs as the driver counts them.

    Not `/sys/class/drm/renderD*`: this container exposes 64 render nodes for 8
    physical cards, and counting those turned a host defect on two lanes into a
    report of 58.
    """
    total = 0
    for props in sorted(Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties")):
        try:
            text = props.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            if line.startswith("gfx_target_version ") and line.split()[1] != "0":
                total += 1
    return total


class LockDirectoryIsUsableTest(unittest.TestCase):
    """Not a property of the script -- a property of the host it is frozen onto.

    The failure mode is not subtle (the wrapper dies on the flock redirect) but
    it is invisible until the first run allocated one of those cards, which on
    an unattended pipeline is the middle of the night. This makes it visible at
    test time instead, and makes the routed-around lanes an explicit list rather
    than folklore.
    """

    def setUp(self):
        if not REAL_LOCK_DIR.exists():
            self.skipTest("no shared lock directory on this host yet")
        self.gpus = kfd_gpu_count()
        if not self.gpus:
            self.skipTest("no GPUs in the KFD topology; not a GPU host")

    @staticmethod
    def lockable(gpu: int) -> bool:
        try:
            with open(REAL_LOCK_DIR / f"gpu_{gpu}.lock", "a"):
                return True
        except OSError:
            return False

    def test_every_lane_is_lockable_or_listed_as_a_known_host_defect(self):
        for gpu in range(self.gpus):
            with self.subTest(gpu=gpu):
                if gpu in HOST_UNUSABLE_LANES:
                    continue
                self.assertTrue(
                    self.lockable(gpu),
                    f"gpu_lock.sh cannot open gpu_{gpu}.lock for writing, so any run "
                    f"allocated GPU {gpu} dies on the flock redirect before the payload "
                    "starts. Fix the host (ownership/mode) or add it to "
                    "HOST_UNUSABLE_LANES with the reason and keep it out of "
                    "GEAK_GPU_ALLOWED.")

    def test_every_listed_defect_is_still_real(self):
        for gpu, reason in sorted(HOST_UNUSABLE_LANES.items()):
            with self.subTest(gpu=gpu):
                self.assertLess(gpu, self.gpus, "this lane is not a GPU on this host")
                self.assertFalse(self.lockable(gpu),
                                 f"gpu_{gpu}.lock is writable now; delete the entry so "
                                 "the lane goes back into the allocatable set")
                self.assertGreater(len(reason), 40,
                                   "an exemption without a stated reason is a silencer")

    def test_enough_lanes_survive_to_run_on(self):
        """The list above routes around a defect; it must not quietly become the
        reason the pipeline has nowhere left to measure."""
        usable = [g for g in range(self.gpus) if g not in HOST_UNUSABLE_LANES]
        self.assertGreaterEqual(len(usable), 4, usable)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class MeasurementFrameFenceTest(GpuLockHarness):
    """Finding (49): a snapshot restore landed mid-round with the process tree still
    running, and three engineers measured straight through the machine change against
    the previous box's floors. The frame preflight is an ENTRY gate and the lane had
    already passed it, so nothing noticed. The fence lives in this wrapper because the
    wrapper is re-entered for every single GPU command, not once per process.

    The checker is located relative to the script, so the staged copy finds whatever
    this harness plants next to it -- which is how these tests choose an exit code
    without needing a second box.

    These cases address FREE_A, not a real device id. They used to say "0", which
    worked only for as long as the restores happened to land on boxes whose GPU 0
    was idle: tw042's is not, so every case that expects the payload to RUN failed
    on the idleness gate before the fence branch was ever reached -- the fence
    itself was never what got measured. A synthetic id has no sysfs node, so the
    idleness probe finds nothing and the fence is the only gate left in play,
    which is the whole point of the FREE_A convention the rest of this file uses.
    """

    def plant_checker(self, exit_code: int, message: str = "planted checker") -> None:
        (self.tmp / "check_measurement_frame.py").write_text(
            "import sys\n"
            f"print({message!r})\n"
            f"sys.exit({exit_code})\n",
            encoding="utf-8")

    def test_a_host_mismatch_refuses_the_command(self):
        self.plant_checker(4)
        r = self.run_lock(FREE_A, "echo", "PAYLOAD-RAN")
        self.assertNotEqual(0, r.returncode)
        self.assertNotIn("PAYLOAD-RAN", r.stdout)
        self.assertIn("does not describe this box", r.stderr)

    def test_mirror_drift_refuses_the_command(self):
        # Exit 5 is "the Python constants and the JS mirror disagree", so the epoch
        # verified is not the epoch that will be applied. Same consequence as a wrong
        # host: the floors judging this command are not this box's floors.
        self.plant_checker(5)
        r = self.run_lock(FREE_A, "echo", "PAYLOAD-RAN")
        self.assertNotEqual(0, r.returncode)
        self.assertNotIn("PAYLOAD-RAN", r.stdout)

    def test_a_provisional_epoch_still_runs(self):
        # The one that must NOT refuse. Exit 3 is a registered-but-unmeasured epoch --
        # the state every new box starts in -- and measuring its floors is itself GPU
        # work routed through this wrapper. Refusing 3 would deadlock the only
        # procedure that clears 3. This case is why the fence branches on the code
        # instead of on "did the checker succeed".
        self.plant_checker(3)
        r = self.run_lock(FREE_A, "echo", "PAYLOAD-RAN")
        self.assertEqual(0, r.returncode, r.stderr)
        self.assertIn("PAYLOAD-RAN", r.stdout)

    def test_a_clean_frame_runs(self):
        self.plant_checker(0)
        r = self.run_lock(FREE_A, "echo", "PAYLOAD-RAN")
        self.assertEqual(0, r.returncode, r.stderr)
        self.assertIn("PAYLOAD-RAN", r.stdout)

    def test_a_missing_checker_warns_but_does_not_block_gpu_work(self):
        # Second line of defence: an infrastructure fault must not become a total GPU
        # outage. It must still be audible, though -- a fence that fails open silently
        # is indistinguishable from one that is working.
        r = self.run_lock(FREE_A, "echo", "PAYLOAD-RAN")
        self.assertEqual(0, r.returncode, r.stderr)
        self.assertIn("PAYLOAD-RAN", r.stdout)
        self.assertIn("WITHOUT the", r.stderr)

    def test_the_fence_runs_before_any_lock_is_taken(self):
        # Refusing after acquiring a lane would leave the lock held across the error
        # path on a box whose frame is already known bad.
        self.plant_checker(4)
        self.run_lock(FREE_A, "echo", "PAYLOAD-RAN")
        self.assertFalse(
            (self.lock_dir / f"gpu_{FREE_A}.lock").exists() and self.entries(),
            "the fence must refuse before the wrapper starts taking lanes")

    def test_the_escape_hatch_is_explicit_and_off_by_default(self):
        self.plant_checker(4)
        r = self.run_lock(FREE_A, "echo", "PAYLOAD-RAN",
                          env={"GEAK_SKIP_FRAME_CHECK": "1"})
        self.assertEqual(0, r.returncode, r.stderr)
        self.assertIn("PAYLOAD-RAN", r.stdout)
