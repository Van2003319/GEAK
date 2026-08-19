#!/usr/bin/env python3
"""GPU-free tests for source_hash.py."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("source_hash.py")
SPEC = importlib.util.spec_from_file_location("source_hash", SCRIPT)
assert SPEC and SPEC.loader
QSH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QSH
SPEC.loader.exec_module(QSH)


def _load_sibling(name: str):
    """Load another script in this directory by path, as SCRIPT is loaded above."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TreeHashTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, relative: str, data: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")
        return path

    def test_identical_trees_hash_identically(self):
        self.write("a.py", "print(1)\n")
        self.write("sub/b.py", "print(2)\n")
        first = QSH.tree_hash(self.root)
        second = QSH.tree_hash(self.root)
        self.assertEqual(first, second)
        self.assertEqual(64, len(first))

    def test_content_change_changes_hash(self):
        self.write("a.py", "print(1)\n")
        before = QSH.tree_hash(self.root)
        self.write("a.py", "print(2)\n")
        after = QSH.tree_hash(self.root)
        self.assertNotEqual(before, after)

    def test_rename_changes_hash(self):
        self.write("a.py", "print(1)\n")
        before = QSH.tree_hash(self.root)
        (self.root / "a.py").rename(self.root / "renamed.py")
        after = QSH.tree_hash(self.root)
        self.assertNotEqual(before, after)

    def test_generated_artifacts_never_affect_the_hash(self):
        self.write("a.py", "print(1)\n")
        baseline = QSH.tree_hash(self.root)
        self.write("__pycache__/a.cpython-312.pyc", "garbage")
        self.write(".git/HEAD", "ref: refs/heads/main\n")
        self.write("build/out.o", "binary-ish")
        self.write(".torch_ext/ext.so", "binary-ish")
        self.write(".rocprofv3/counters.csv", "generated")
        self.write("logs/benchmark.log", "timing noise")
        self.write("reports/profile.json", "generated")
        self.write("top-level.o", "binary-ish")
        self.write("extension.so", "binary-ish")
        self.write("run.log", "timing noise")
        self.write("generated.hipify.cpp", "generated")
        self.write("worker_result.json", "generated")
        self.assertEqual(baseline, QSH.tree_hash(self.root))

    def test_extra_excluded_dirs_are_respected(self):
        self.write("a.py", "print(1)\n")
        baseline = QSH.tree_hash(self.root)
        self.write("scratch/notes.txt", "not part of the candidate")
        self.assertNotEqual(baseline, QSH.tree_hash(self.root))
        self.assertEqual(baseline, QSH.tree_hash(self.root, extra_excluded_dirs=["scratch"]))

    def test_nested_excluded_dir_is_excluded_too(self):
        self.write("a.py", "print(1)\n")
        self.write("sub/b.py", "print(2)\n")
        baseline = QSH.tree_hash(self.root)
        self.write("sub/__pycache__/x.pyc", "garbage")
        self.assertEqual(baseline, QSH.tree_hash(self.root))

    def test_symlink_is_hashed_by_target_not_followed(self):
        target = self.write("outside.py", "print('outside')\n")
        candidate = self.root / "candidate"
        candidate.mkdir()
        (candidate / "link.py").symlink_to(target)
        first = QSH.tree_hash(candidate)
        # Changing the symlink's target content must not change the tree hash:
        # the link is hashed by its target path text, never dereferenced.
        target.write_text("print('changed')\n", encoding="utf-8")
        second = QSH.tree_hash(candidate)
        self.assertEqual(first, second)

    def test_traversal_order_does_not_matter(self):
        # Two trees built by writing files in a different order must still
        # hash identically -- the walk sorts entries deterministically.
        self.write("z.py", "1")
        self.write("a.py", "2")
        first = QSH.tree_hash(self.root)
        with tempfile.TemporaryDirectory() as other:
            other_root = Path(other)
            (other_root / "a.py").write_text("2", encoding="utf-8")
            (other_root / "z.py").write_text("1", encoding="utf-8")
            second = QSH.tree_hash(other_root)
        self.assertEqual(first, second)

    def test_single_file_root_is_supported(self):
        f = self.write("only.py", "print(1)\n")
        self.assertEqual(64, len(QSH.tree_hash(f)))

    def test_cli_is_deterministic(self):
        self.write("a.py", "print(1)\n")
        run1 = subprocess.run([sys.executable, str(SCRIPT), str(self.root)],
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        run2 = subprocess.run([sys.executable, str(SCRIPT), str(self.root)],
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        self.assertEqual(run1.stdout, run2.stdout)
        payload = json.loads(run1.stdout)
        self.assertEqual("geak.source-hash/v1", payload["schema"])
        self.assertEqual(64, len(payload["source_hash"]))


# The prose below is copied from the shipped v59 kernel's rasterization comment.
# It is the strongest available negative fixture precisely because it is real: it
# names XCDs six times, quotes `kGroupM = 8` verbatim, and spells out the remap
# arithmetic -- while arguing that the v58 grouping it describes LOST. Any
# rasterization rule that fires on it grounds a mechanism claim on prose that
# rejects the mechanism. A first draft of both rules did exactly that.
V59_PROSE = """
// v59: XCD-aware grouped rasterization.
// gfx942 dispatches consecutive block ids round-robin over 8 XCDs, each with
// its own L2, so blockIdx adjacency is not L2 adjacency: XCD x receives
// exactly the blocks with pid % 8 == x. v58 grouped kGroupM = 8 rows tall and
// 8 is the XCD count, so `in_group % group_m` collapsed to `pid % 8`.
// So un-shuffle the round-robin first -- p = (pid % 8) * chunk + pid / 8 --
// and only then group. v58 did not lose because grouping is wrong.
"""

V59_CODE = """
constexpr int kGroupM = 8;
constexpr int kXcds = 8;
const int chunk = nblocks / kXcds;
const int p = pid < kXcds * chunk ? (pid % kXcds) * chunk + pid / kXcds : pid;
const int group_m = min(kGroupM, tiles_m - first_m);
"""

LINEAR_CODE = """
// plain linear order; we deliberately do not remap across XCD boundaries.
int row0 = blockIdx.y * CTA_M;
int col0 = blockIdx.x * CTA_N;
"""


if __name__ == "__main__":
    unittest.main(verbosity=2)
