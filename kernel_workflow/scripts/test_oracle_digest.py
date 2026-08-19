#!/usr/bin/env python3
"""Regression tests for the prescribed `oracle_digest` snippet -- finding (119).

Round 14 of the BF16 GEMM search ran 20 agents for 99 minutes and then failed
closed with `oracle:digest_drift`: "the immutable oracle changed during the run".
It had not. The pinned digest `a2927b6b...` is still, right now, the digest of
the task dir; the "now" value `b667990b...` is the digest of the eval WORKSPACE,
a 34-file tree whose whole purpose is to be edited by the engineers.

The mechanism is one line, present verbatim in two role prompts:

    cd "$TASK_DIR" || return 1

That reads as a guard and is not one. In bash `cd ""` is a silent no-op that
returns 0, so an unset or empty TASK_DIR does not take the `|| return 1` branch
-- it digests the current directory. And the lane never passed TASK_DIR to the
verify_engineer role at all, while the COMMANDMENT tells every agent to work
from inside the workspace. So the verifier confidently digested the candidate
tree, disagreed with the Director's pin (as it must, on every single run that
reaches verification), and the lane reported it as the oracle being rewritten
mid-run.

Two things are therefore under test here, and both halves matter:

  * the snippet must refuse to digest anything when it has not been told what to
    digest -- silence beats a confident answer about the wrong directory;
  * the lane must actually supply TASK_DIR to every role that runs the snippet,
    or the fixed snippet just converts a false `digest_drift` into a real
    `digest_missing` and the run stalls one gate later.

Nothing here touches a GPU or runs the workflow; the digest is pure filesystem.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROLES = ROOT / "roles"
LANE = ROOT / "kernel_lane.js"

HEX64 = re.compile(r"^[0-9a-f]{64}$")

# The snippet as each role publishes it, dedented. Both roles state in prose that
# their copies must stay identical; test_the_two_role_copies_are_identical is what
# makes that a fact rather than an intention.
SNIPPET_RE = re.compile(
    r"^(?P<ind>[ ]*)oracle_digest\(\) \{\n.*?\n(?P=ind)\}\n(?P=ind)oracle_digest\n",
    re.S | re.M)


def extract_snippet(role: str) -> str:
    text = (ROLES / f"{role}.md").read_text()
    match = SNIPPET_RE.search(text)
    assert match, f"no oracle_digest snippet found in {role}.md"
    return textwrap.dedent(match.group(0))


def run_snippet(snippet: str, cwd: Path, task_dir: str | None
                ) -> subprocess.CompletedProcess:
    """Run the snippet exactly as an agent would: a bash shell, in some working
    directory, with TASK_DIR either exported or absent."""
    prelude = "" if task_dir is None else f"export TASK_DIR={task_dir!r}\n"
    return subprocess.run(["bash", "-c", prelude + snippet],
                          cwd=str(cwd), capture_output=True, text=True, timeout=120)


def make_oracle(root: Path) -> Path:
    """A miniature of the real task dir: a few source files plus the build output
    and caches the snippet is supposed to ignore."""
    task = root / "task"
    (task / "src").mkdir(parents=True)
    (task / "build").mkdir()
    (task / "__pycache__").mkdir()
    (task / "harness_lib.py").write_text("def time_op():\n    pass\n")
    (task / "config.yaml").write_text("tol: 2e-2\n")
    (task / "src" / "rocblas_baseline.cpp").write_text("// oracle\n")
    (task / "build" / "report.json").write_text('{"compile": "PASS"}')
    (task / "__pycache__" / "harness_lib.cpython-311.pyc").write_bytes(b"\x00cached")
    (task / "src" / "kernel.so").write_bytes(b"\x7fELF")
    return task


class SnippetParityTest(unittest.TestCase):
    def test_the_two_role_copies_are_identical(self):
        """Both files say, in prose, that the other side must not author its own
        version -- because a consistency check whose two sides compute different
        functions agrees only by luck. That claim was true of the file list and
        false of the guard the moment one side was edited alone."""
        self.assertEqual(extract_snippet("director"), extract_snippet("verify_engineer"),
                         "the Director and the Verifier are digesting with different code; "
                         "the drift check would then compare two different functions")

    def test_the_snippet_is_valid_bash(self):
        for role in ("director", "verify_engineer"):
            with self.subTest(role=role):
                proc = subprocess.run(["bash", "-n"], input=extract_snippet(role),
                                      capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)


class DigestBehaviourTest(unittest.TestCase):
    def setUp(self):
        self.snippet = extract_snippet("director")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.task = make_oracle(self.root)
        # The decoy stands in for the eval workspace: a real directory, full of
        # plausible files, that the agent is standing in and must NOT digest.
        self.decoy = self.root / "workspace"
        (self.decoy / "src").mkdir(parents=True)
        (self.decoy / "src" / "custom_gemm.hip").write_text("// candidate\n")
        (self.decoy / "harness_lib.py").write_text("def time_op():\n    pass\n")

    def digest_of(self, cwd: Path, task_dir: str | None):
        return run_snippet(self.snippet, cwd, task_dir)

    # -- the round 14 regression ------------------------------------------------

    def test_an_unset_task_dir_does_not_silently_digest_the_current_directory(self):
        proc = self.digest_of(self.decoy, None)
        self.assertNotEqual(proc.returncode, 0,
                            "the snippet answered without being told what to digest")
        self.assertIn("digest_no_task_dir", proc.stderr)
        self.assertFalse([l for l in proc.stdout.splitlines() if HEX64.match(l.strip())],
                         "a digest was emitted for the current directory; that value would "
                         "be compared against the pin and reported as the oracle drifting")

    def test_an_empty_task_dir_is_refused_too_because_cd_to_it_succeeds(self):
        """`cd ""` returns 0. This is the exact case the original `|| return 1`
        was believed to cover and did not."""
        proc = self.digest_of(self.decoy, "")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("digest_no_task_dir", proc.stderr)

    def test_a_task_dir_that_does_not_exist_fails_closed(self):
        proc = self.digest_of(self.decoy, str(self.root / "no_such_dir"))
        self.assertNotEqual(proc.returncode, 0)
        self.assertNotIn("\n" + "0" * 64, proc.stdout)

    def test_a_file_path_is_not_a_task_dir(self):
        proc = self.digest_of(self.decoy, str(self.task / "config.yaml"))
        self.assertNotEqual(proc.returncode, 0)

    # -- the gate still does its job -------------------------------------------

    def test_it_digests_the_task_dir_regardless_of_where_it_is_invoked_from(self):
        a = self.digest_of(self.decoy, str(self.task))
        b = self.digest_of(self.task, str(self.task))
        self.assertEqual(a.returncode, 0, a.stderr)
        self.assertEqual(b.returncode, 0, b.stderr)
        digest = a.stdout.strip().splitlines()[-1]
        self.assertTrue(HEX64.match(digest), a.stdout)
        self.assertEqual(digest, b.stdout.strip().splitlines()[-1],
                         "the answer depends on the caller's cwd, so two roles running "
                         "the same snippet on the same oracle can still disagree")

    def test_it_reports_the_root_and_the_count_it_actually_used(self):
        proc = self.digest_of(self.decoy, str(self.task))
        self.assertIn(f"oracle digest root: {self.task}", proc.stderr)
        self.assertIn("oracle files digested: 3", proc.stderr,
                      "build output, __pycache__ and .so must be outside the file set")

    def test_a_real_change_to_an_oracle_file_moves_the_digest(self):
        before = self.digest_of(self.decoy, str(self.task)).stdout.strip().splitlines()[-1]
        (self.task / "harness_lib.py").write_text("def time_op():\n    return 0\n")
        after = self.digest_of(self.decoy, str(self.task)).stdout.strip().splitlines()[-1]
        self.assertNotEqual(before, after,
                            "the drift gate cannot see a rewritten harness")

    def test_a_new_oracle_file_moves_the_digest(self):
        before = self.digest_of(self.decoy, str(self.task)).stdout.strip().splitlines()[-1]
        (self.task / "src" / "sneaky.h").write_text("#pragma once\n")
        after = self.digest_of(self.decoy, str(self.task)).stdout.strip().splitlines()[-1]
        self.assertNotEqual(before, after)

    def test_rebuilding_does_not_move_the_digest(self):
        """The other failure direction: if build output counted, every run would
        drift and the gate would be turned off within a day."""
        before = self.digest_of(self.decoy, str(self.task)).stdout.strip().splitlines()[-1]
        (self.task / "build" / "report.json").write_text('{"compile": "PASS", "t": 12}')
        (self.task / "build" / "kernel.so").write_bytes(b"\x7fELF\x01\x02")
        (self.task / "__pycache__" / "harness_lib.cpython-311.pyc").write_bytes(b"\x00new")
        after = self.digest_of(self.decoy, str(self.task)).stdout.strip().splitlines()[-1]
        self.assertEqual(before, after)

    def test_an_empty_file_set_is_refused(self):
        empty = self.root / "empty_task"
        empty.mkdir()
        proc = self.digest_of(self.decoy, str(empty))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("digest_empty_fileset", proc.stderr)
        # abcfa6a9... is sha256sum of the empty stream: a well-formed digest that
        # covers zero bytes and never changes.
        self.assertNotIn("abcfa6a9", proc.stdout)


class LaneSuppliesTaskDirTest(unittest.TestCase):
    """A snippet that refuses without TASK_DIR is only half the fix. If the lane
    still does not pass it, the verifier returns no digest, the candidate is
    refused as `oracle:digest_missing`, and the run dies at the next gate instead
    of this one."""

    def setUp(self):
        self.lane = LANE.read_text()

    def digest_running_roles(self) -> set[str]:
        return {p.stem for p in ROLES.glob("*.md")
                if "oracle_digest() {" in p.read_text()}

    def test_every_role_that_runs_the_digest_is_handed_a_task_dir(self):
        roles = self.digest_running_roles()
        self.assertTrue(roles, "no role publishes the digest snippet any more")
        for role in sorted(roles):
            for match in re.finditer(rf"roleAgent\('{role}',\s*'(\w+)'", self.lane):
                action = match.group(1)
                # The input object ends at the first line that closes the
                # roleAgent(...) call at the same nesting the call opened on.
                tail = self.lane[match.start():match.start() + 4000]
                end = tail.find("\n      }),")
                if end == -1:
                    end = tail.find("}),")
                block = tail[:end]
                with self.subTest(role=role, action=action):
                    self.assertIn("TASK_DIR", block,
                                  f"roleAgent('{role}', '{action}') runs the oracle digest "
                                  f"but is never told which directory the oracle is in")

    def test_task_dir_is_always_the_original_task_not_the_workspace(self):
        """Every supplied TASK_DIR must name the frozen original. Pointing it at
        CANONICAL or a seed dir would restore the round 14 failure with a guard
        in place and an explicit value to blame it on."""
        for match in re.finditer(r"TASK_DIR:\s*([A-Za-z_][\w.]*)", self.lane):
            self.assertEqual(match.group(1), "KERNEL_PATH_ORIG",
                             "TASK_DIR must be the immutable original task dir")


if __name__ == "__main__":
    unittest.main()
