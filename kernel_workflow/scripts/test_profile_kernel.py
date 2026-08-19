#!/usr/bin/env python3
"""Deterministic, GPU-free tests for `profile_kernel.sh`.

This script had no test of any kind, which the extended module inventory in
`test_script_conventions.py` surfaced once it stopped globbing `*.py` only. It
is not a minor helper: three role prompts (`deep_engineer`, `benchmark_engineer`,
`profile_engineer`) invoke it by name every profiling round, and every GPU
launch it makes is supposed to go through `gpu_lock.sh`.

That last property is the reason this file leads with a *behavioural* fence test
rather than a source scan. A source scan asks "does the text contain
`bash "$GPU_LOCK"`"; what actually matters is whether any process that touches
the GPU can be reached without holding the lock. So the fake lock here exports a
marker, and every stub profiler and the benchmark command itself record whether
they saw it. A future branch that runs a tool directly fails this even if it is
written in a form no `grep` would have anticipated.

Nothing here needs a GPU, a profiler, or ROCm: the script is copied into a temp
directory beside a fake `gpu_lock.sh` (it resolves its sibling through
`BASH_SOURCE`, so a copy re-points it) and the profilers are stubs on `PATH`.
The copy is byte-identical to the real file -- read from it each run rather than
reproduced here, so this suite cannot drift into testing a fossil.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "profile_kernel.sh"

# The fake lock. Mirrors the real contract -- `<ids> <payload...>` -- and does
# nothing but record the call and hand the payload straight on, with a marker in
# the environment so anything downstream can prove it ran inside.
FAKE_LOCK = """#!/bin/bash
printf '%s\\n' "$*" >> "$LOCK_LOG"
shift
export GEAK_TEST_LOCK_HELD=1
exec "$@"
"""

# One stub stands in for every profiler. It records its own name and whether the
# lock marker was in its environment, optionally fails with a chosen code, and
# then runs whatever benchmark command was handed to it so the command's own
# fence record is produced too.
STUB = """#!/bin/bash
me="$(basename "$0")"
printf '%s held=%s\\n' "$me" "${GEAK_TEST_LOCK_HELD:-0}" >> "$TOOL_LOG"
echo "stub $me output"
for t in $GEAK_TEST_FAIL_TOOLS; do
    if [ "$t" = "$me" ]; then exit "${GEAK_TEST_FAIL_CODE:-3}"; fi
done
seen_bash=0
args=()
for a in "$@"; do
    if [ "$seen_bash" = 1 ]; then args+=("$a"); continue; fi
    if [ "$a" = "bash" ]; then seen_bash=1; fi
done
if [ "$seen_bash" = 1 ]; then bash "${args[@]}"; fi
exit 0
"""

TOOLS = ("rocprof-compute", "omniperf", "rocprofv3", "rocprof", "metrix")

# The suite runs on a sanitized PATH holding nothing but these utilities and the
# stubs a test installs. Inheriting the host PATH looked harmless and was not:
# this box has a real `rocprofv3`, so "no profiler is installed" quietly became
# "the real profiler was chosen", one test asserted the wrong branch, and one
# test *launched an actual profiler against the GPU* from what is supposed to be
# a GPU-free unit suite.
SYS_UTILS = ("bash", "env", "seq", "date", "basename", "dirname", "pwd", "mkdir",
             "mv", "cat", "tail", "sed", "find", "sort", "grep", "printf", "cut",
             "tr", "head", "wc", "ls")

# Each ladder branch must point the engineer at the env var that actually
# controls the args it just failed with. A wrong name here is invisible: the
# report still looks like a helpful self-heal note, and the override it suggests
# does nothing.
OVERRIDE_VARS = {
    "rocprof-compute": "RPC_PROFILE_ARGS",
    "omniperf": "RPC_PROFILE_ARGS",
    "rocprofv3": "RPV3_TRACE_ARGS",
    "rocprof": "RPROF_ARGS",
    "metrix": "METRIX_ARGS",
}


class ProfileKernelHarness(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="profkern_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.sysbin = self.tmp / "sysbin"
        self.sysbin.mkdir()
        for util in SYS_UTILS:
            found = shutil.which(util)
            if found:
                (self.sysbin / util).symlink_to(found)
        self.out = self.tmp / "out"
        self.script = self.tmp / "profile_kernel.sh"
        self.script.write_bytes(SCRIPT.read_bytes())
        lock = self.tmp / "gpu_lock.sh"
        lock.write_text(FAKE_LOCK, encoding="utf-8")
        lock.chmod(0o755)
        self.lock_log = self.tmp / "lock.log"
        self.tool_log = self.tmp / "tool.log"
        self.bench_log = self.tmp / "bench.log"

    def install(self, *tools: str):
        for tool in tools:
            path = self.bin / tool
            path.write_text(STUB, encoding="utf-8")
            path.chmod(0o755)

    def run_script(self, *, priority: str, fail: str = "", code: int = 3,
                   warmup: int = 1, env: dict[str, str] | None = None,
                   bench: str | None = None, expect_ok: bool = True):
        cmd = bench or (
            f'printf "bench held=%s arch=%s\\n" "${{GEAK_TEST_LOCK_HELD:-0}}" '
            f'"${{PYTORCH_ROCM_ARCH:-unset}}" >> {self.bench_log}')
        environ = dict(os.environ)
        # Sanitized, not merely prefixed: the suite must behave identically on a
        # box that has rocprofv3 installed and one that does not.
        environ.update({
            "PATH": f"{self.bin}{os.pathsep}{self.sysbin}",
            "LOCK_LOG": str(self.lock_log),
            "TOOL_LOG": str(self.tool_log),
            "GEAK_TEST_FAIL_TOOLS": fail,
            "GEAK_TEST_FAIL_CODE": str(code),
            "PROFILER_PRIORITY": priority,
            "WARMUP_RUNS": str(warmup),
            "KERNEL_ENV_KEEP_ARCH": "1",
        })
        environ.pop("PYTORCH_ROCM_ARCH", None)
        environ.update(env or {})
        proc = subprocess.run(["bash", str(self.script), "7", cmd, str(self.out)],
                              capture_output=True, text=True, env=environ, timeout=120)
        if expect_ok:
            self.assertEqual(0, proc.returncode, proc.stdout + proc.stderr)
            # A utility missing from SYS_UTILS does not fail the run: `dirname`
            # went missing once, SCRIPT_DIR resolved to nothing, every locked
            # call silently no-opped behind `|| true`, and the script still
            # exited 0 with a plausible-looking report. The harness has to catch
            # its own holes or it reports them as findings about the script.
            self.assertNotIn("command not found", proc.stderr, proc.stderr)
        return proc

    def report(self) -> str:
        return (self.out / "profile_report.txt").read_text(encoding="utf-8")

    def lines(self, path: Path) -> list[str]:
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


class GpuFenceTest(ProfileKernelHarness):
    """Nothing reaches the GPU except through `gpu_lock.sh`."""

    def test_every_profiler_and_every_benchmark_run_holds_the_lock(self):
        for priority in ("rocprof-compute", "rocprofv3", "rocprof", "metrix", "omniperf"):
            with self.subTest(profiler=priority):
                self.setUp()
                self.install(priority)
                self.run_script(priority=priority, warmup=2)
                tools = self.lines(self.tool_log)
                self.assertTrue(tools, "the stub profiler never ran at all")
                for line in tools:
                    self.assertIn("held=1", line,
                                  f"{line!r}: a profiler was launched outside gpu_lock.sh")
                bench = self.lines(self.bench_log)
                self.assertTrue(bench, "the benchmark command never ran")
                for line in bench:
                    self.assertIn("held=1", line,
                                  f"{line!r}: the benchmark ran unfenced")

    def test_the_fallback_path_is_fenced_too(self):
        """The one branch that runs the benchmark with no profiler wrapping it,
        and so the one most easily written without the lock."""
        # The marker is on the script's own stdout, not in the report: the
        # report only ever holds what the benchmark itself printed, which for
        # this stub is nothing.
        proc = self.run_script(priority="nonexistent-profiler", warmup=1)
        self.assertIn("benchmark-only", proc.stdout)
        bench = self.lines(self.bench_log)
        self.assertGreaterEqual(len(bench), 2, bench)  # warmup + fallback
        for line in bench:
            self.assertIn("held=1", line)

    def test_the_fence_test_can_fail(self):
        """(55). If the marker were never exported, every assertion above would
        read `held=1` from nowhere and pass vacuously."""
        self.install("rocprofv3")
        (self.tmp / "gpu_lock.sh").write_text(
            '#!/bin/bash\nprintf \'%s\\n\' "$*" >> "$LOCK_LOG"\nshift\nexec "$@"\n',
            encoding="utf-8")
        self.run_script(priority="rocprofv3")
        self.assertTrue(any("held=0" in l for l in self.lines(self.tool_log)),
                        "a lock that sets no marker still produced held=1")


class FenceMutationTest(ProfileKernelHarness):
    """Every single `gpu_lock` call site is individually load-bearing.

    The fence test above passes as long as the paths it happens to exercise are
    locked. This one deletes the wrapper from one call site at a time and
    requires that *something* notices -- which is the difference between "the
    fence is tested" and "the fence is tested everywhere it exists".
    """

    WRAPPER = 'bash "$GPU_LOCK" "$GPU_ID"'

    def test_removing_any_one_lock_wrapper_is_detected(self):
        src = SCRIPT.read_text(encoding="utf-8")
        sites = src.count(self.WRAPPER)
        self.assertGreaterEqual(sites, 6,
                                "fewer lock sites than branches that touch the GPU")
        for i in range(sites):
            with self.subTest(site=i):
                head, sep, tail = "", "", src
                for _ in range(i + 1):
                    a, sep, tail = tail.partition(self.WRAPPER)
                    head += a + (sep if _ < i else "")
                mutated = head + tail
                self.assertNotEqual(src, mutated)
                unfenced = False
                # `None` is the no-profiler run, and it is not optional: the
                # benchmark-only fallback is reachable no other way, so without
                # it site 6 looks deletable and the test reports the fence as
                # untested where it is in fact merely unexercised.
                for tool in TOOLS + (None,):
                    self.setUp()
                    self.script.write_text(mutated, encoding="utf-8")
                    if tool:
                        self.install(tool)
                    self.run_script(priority=tool or "nonexistent", warmup=1)
                    lines = self.lines(self.tool_log) + self.lines(self.bench_log)
                    unfenced = unfenced or any("held=0" in l for l in lines)
                self.assertTrue(
                    unfenced,
                    f"lock site {i} can be deleted and no test notices: either it is "
                    "unreachable dead code or the GPU work behind it is unguarded")


class ProfilerSelectionTest(ProfileKernelHarness):
    def test_the_first_available_profiler_in_the_priority_list_wins(self):
        self.install("rocprofv3", "rocprof")
        self.run_script(priority="rocprofv3 rocprof")
        names = {l.split()[0] for l in self.lines(self.tool_log)}
        self.assertIn("rocprofv3", names)
        self.assertNotIn("rocprof", names)

    def test_priority_order_is_honoured_and_not_hardcoded(self):
        self.install("rocprofv3", "rocprof")
        self.run_script(priority="rocprof rocprofv3")
        names = {l.split()[0] for l in self.lines(self.tool_log)}
        self.assertIn("rocprof", names)
        self.assertNotIn("rocprofv3", names)

    def test_an_unavailable_first_choice_is_skipped_not_fatal(self):
        self.install("rocprof")
        self.run_script(priority="rocprof-compute rocprofv3 rocprof")
        self.assertIn("rocprof", {l.split()[0] for l in self.lines(self.tool_log)})

    def test_no_profiler_at_all_degrades_to_benchmark_only(self):
        proc = self.run_script(priority="rocprofv3 rocprof")
        self.assertIn("Profiler used: benchmark-only", proc.stdout)


class FaultToleranceLadderTest(ProfileKernelHarness):
    """A profiler that fails must say so, and say how to recover."""

    def test_a_failing_profiler_names_its_own_override_variable(self):
        for tool, var in sorted(OVERRIDE_VARS.items()):
            with self.subTest(tool=tool):
                self.setUp()
                self.install(tool)
                self.run_script(priority=tool, fail=tool, code=42)
                report = self.report()
                self.assertIn(f"PROFILER FAILED: {tool} exited 42", report)
                self.assertIn(var, report,
                              "the self-heal note points at some other env var, so the "
                              "override it suggests would change nothing")
                self.assertIn("profiling_guide.md", report,
                              "the recipe pointer is the half an agent can act on")

    def test_no_override_variable_is_shared_by_two_unrelated_ladders(self):
        """rocprof-compute and omniperf are the same tool under two names and
        legitimately share one variable; anything else sharing would mean a
        branch is advertising a knob it does not read."""
        shared = {}
        for tool, var in OVERRIDE_VARS.items():
            shared.setdefault(var, []).append(tool)
        for var, tools in sorted(shared.items()):
            with self.subTest(var=var):
                self.assertTrue(len(tools) == 1 or set(tools) == {"rocprof-compute", "omniperf"},
                                f"{var} is claimed by {sorted(tools)}")

    def test_a_failure_still_leaves_a_report_and_exits_zero(self):
        """The engineer reads `profile_report.txt`; a nonzero exit here would
        abort the calling role before it ever got to read the explanation."""
        self.install("rocprofv3")
        proc = self.run_script(priority="rocprofv3", fail="rocprofv3")
        self.assertEqual(0, proc.returncode)
        self.assertTrue((self.out / "profile_report.txt").exists())

    def test_the_ladder_covers_every_dispatchable_profiler(self):
        """(135) again, in miniature: the `case` at the bottom is the inventory,
        and a name in the default priority list with no branch silently falls
        through to benchmark-only."""
        src = SCRIPT.read_text(encoding="utf-8")
        default = src.split('PROFILER_PRIORITY:-', 1)[1].split('}', 1)[0].strip().strip('"')
        listed = default.split()
        self.assertGreaterEqual(len(listed), 4, listed)
        case_body = src.split('case "$PROFILER" in', 1)[1].split("esac", 1)[0]
        for tool in listed:
            with self.subTest(tool=tool):
                self.assertIn(tool, case_body,
                              f"{tool} is offered in the default priority list but has no "
                              "branch; if it is installed it is chosen and then does nothing")
                self.assertIn(tool, OVERRIDE_VARS, "this test file's own table is stale")


class NonDestructiveTest(ProfileKernelHarness):
    def test_a_stale_workload_directory_is_moved_aside_not_deleted(self):
        """The script says why in a comment -- `rm` prompts and blocks an
        unattended run -- which is exactly the kind of reason that gets edited
        away by someone tidying up."""
        for tool, sub in (("rocprofv3", "rocprofv3"), ("metrix", "metrix"),
                          ("rocprof-compute", "rocprof-compute_workload")):
            with self.subTest(tool=tool):
                self.setUp()
                self.install(tool)
                stale = self.out / sub
                stale.mkdir(parents=True)
                (stale / "keepme.txt").write_text("prior run", encoding="utf-8")
                self.run_script(priority=tool)
                moved = [p for p in self.out.iterdir() if p.name.startswith(f"{sub}.old_")]
                self.assertEqual(1, len(moved), [p.name for p in self.out.iterdir()])
                self.assertEqual("prior run",
                                 (moved[0] / "keepme.txt").read_text(encoding="utf-8"))

    def test_the_script_never_calls_rm(self):
        src = SCRIPT.read_text(encoding="utf-8")
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        for token in (" rm ", " rm -", "rm -rf"):
            with self.subTest(token=token):
                self.assertNotIn(token, code)


class WarmupAndArchTest(ProfileKernelHarness):
    def test_warmup_runs_the_requested_number_of_times_under_the_lock(self):
        for n in (1, 3):
            with self.subTest(warmup=n):
                self.setUp()
                self.run_script(priority="nonexistent", warmup=n)
                # n warmups plus the single benchmark-only fallback run.
                self.assertEqual(n + 1, len(self.lines(self.bench_log)))

    def test_a_caller_set_arch_is_not_overwritten(self):
        self.run_script(priority="nonexistent", warmup=1,
                        env={"PYTORCH_ROCM_ARCH": "gfx000", "KERNEL_ENV_KEEP_ARCH": "0"})
        for line in self.lines(self.bench_log):
            self.assertIn("arch=gfx000", line)

    def test_keep_arch_opts_out_of_the_sniff_entirely(self):
        """The opt-out has to mean "leave it unset", not "sniff anyway": the
        whole point is a caller who wants the build's own multi-arch default."""
        self.run_script(priority="nonexistent", warmup=1,
                        env={"KERNEL_ENV_KEEP_ARCH": "1"})
        for line in self.lines(self.bench_log):
            self.assertIn("arch=unset", line)


class UsageTest(ProfileKernelHarness):
    def test_missing_arguments_fail_closed(self):
        for argv in ([], ["7"], ["7", "true"]):
            with self.subTest(argv=argv):
                proc = subprocess.run(["bash", str(self.script), *argv],
                                      capture_output=True, text=True, timeout=60)
                self.assertNotEqual(0, proc.returncode,
                                    "a missing output dir would scatter artifacts into cwd")

    def test_the_output_directory_is_created_and_holds_the_report(self):
        self.run_script(priority="nonexistent", warmup=1)
        self.assertTrue((self.out / "profile_report.txt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
