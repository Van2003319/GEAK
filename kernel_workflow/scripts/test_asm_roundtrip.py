#!/usr/bin/env python3
"""Guards for the phase2 assembly round trip.

Two halves, deliberately separated.

The FIRST half needs no toolchain: flag reading, the refusals, and the code-object picker. These are
the parts that decide WHICH program gets rebuilt, and getting them wrong produces a plausible object
built from the wrong flags -- which is the failure the module's docstring is about and the one a
GPU-less CI can still catch.

The SECOND half compiles, so it skips without ROCm. It is not optional detail: the whole design rests
on `-fcuda-include-gpubinary` substituting our code object into an otherwise normal host compile, and
the only way to know that still holds on a given toolchain is to do it. A round trip that silently
stopped preserving the instruction stream would make every phase2 measurement attribute the round
trip's own damage to the engineer's edit.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import asm_roundtrip as RT  # noqa: E402

HIPCC = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
HAVE_TOOLCHAIN = (
    Path(HIPCC).is_file()
    and shutil.which("clang") is not None
    and shutil.which("clang-offload-bundler") is not None
    and shutil.which("llvm-objdump") is not None
)

KERNEL = textwrap.dedent("""\
    #include <hip/hip_runtime.h>
    extern "C" __global__ void demo_axpy(const float* a, const float* b, float* c, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) {
        float acc = 0.0f;
        for (int k = 0; k < 4; ++k) acc += a[i] * (float)k + b[i];
        c[i] = acc;
      }
    }
    """)

NINJA = textwrap.dedent("""\
    ninja_required_version = 1.3
    cxx = c++
    nvcc = {hipcc}

    cflags = -DTORCH_EXTENSION_NAME=demo -I/opt/rocm/include -fPIC -std=c++17 -O3
    post_cflags = -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -fPIC
    cuda_cflags = -DWITH_HIP -I/opt/rocm/include -O3 -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -fPIC -std=c++17 --offload-arch={arch} -fno-gpu-rdc -O3
    cuda_post_cflags =
    ldflags = -shared -L/opt/rocm/lib -lamdhip64

    rule compile
      command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags
      depfile = $out.d
      deps = gcc

    rule cuda_compile
      command = $nvcc  $cuda_cflags -c $in -o $out $cuda_post_cflags

    rule link
      command = $cxx $in $ldflags -o $out

    build demo_hip.cuda.o: cuda_compile {src}
    build demo.so: link demo_hip.cuda.o
    """)


def make_tree(root: Path, arch: str = "gfx942") -> Path:
    """A tree shaped like the one torch.utils.cpp_extension.load() leaves behind."""
    src = root / "src"
    ext = root / ".torch_ext" / "demo"
    src.mkdir(parents=True, exist_ok=True)
    ext.mkdir(parents=True, exist_ok=True)
    hip = src / "demo_hip.hip"
    hip.write_text(KERNEL, encoding="utf-8")
    (ext / "build.ninja").write_text(
        NINJA.format(hipcc=HIPCC, arch=arch, src=hip), encoding="utf-8")
    return src


class FlagReadingTest(unittest.TestCase):
    """The arch is READ, never defaulted.

    A wrong -mcpu reassembles against a different instruction encoding, and the failure surfaces at
    link or load time pointing at the engineer's edit rather than at the guess that caused it.
    """

    def test_the_arch_comes_from_the_builds_own_flags(self):
        self.assertEqual("gfx950", RT.arch_from_flags(["-O3", "--offload-arch=gfx950", "-fPIC"]))

    def test_the_older_spelling_is_read_too(self):
        self.assertEqual("gfx90a", RT.arch_from_flags(["--amdgpu-target=gfx90a"]))

    def test_flags_naming_no_arch_return_none_rather_than_a_default(self):
        self.assertIsNone(RT.arch_from_flags(["-O3", "-fPIC", "-std=c++17"]))


class RefusalTest(unittest.TestCase):
    """Every way the build cannot be replayed is a refusal that says which way."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="asmrt_t_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_tree_that_was_never_built_is_a_hole_not_a_guess(self):
        # Deep enough that `find_ninja`'s parent search finds nothing either: it looks in the source
        # root AND its parent by design, because `.torch_ext/` conventionally sits beside `src/`.
        iso = self.tmp / "a" / "b" / "src"
        iso.mkdir(parents=True)
        build, holes = RT.resolve_build(iso, None, None)
        self.assertIsNone(build)
        self.assertTrue(any("ninja:absent" in h for h in holes), holes)

    def test_a_build_naming_no_offload_arch_is_refused(self):
        src = make_tree(self.tmp)
        ninja = self.tmp / ".torch_ext" / "demo" / "build.ninja"
        ninja.write_text(ninja.read_text(encoding="utf-8").replace("--offload-arch=gfx942", ""),
                         encoding="utf-8")
        build, holes = RT.resolve_build(src, "demo_hip.hip", None)
        self.assertIsNone(build)
        self.assertTrue(any("no_offload_arch" in h for h in holes), holes)

    def test_the_default_output_resolves_against_the_build_directory(self):
        """A ninja target is relative to the directory holding build.ninja, not to the caller's cwd.

        Resolving it against cwd wrote the patched object somewhere the build never looks; the link
        then picked up the ORIGINAL object and the round read as "the edit did not help" rather than
        as "the edit was never built". That failure is silent in every log, which is why it is
        pinned here rather than left to the integration.
        """
        src = make_tree(self.tmp)
        build, holes = RT.resolve_build(src, "demo_hip.hip", None)
        self.assertIsNotNone(build, holes)
        out = Path(build["output"])
        self.assertTrue(out.is_absolute(), out)
        self.assertEqual(Path(build["build_dir"]), out.parent,
                         "the object must land beside build.ninja, which is where ninja links from")
        self.assertEqual("demo_hip.cuda.o", out.name)

    def test_a_recorded_build_yields_the_compiler_flags_and_arch_verbatim(self):
        src = make_tree(self.tmp, arch="gfx942")
        build, holes = RT.resolve_build(src, "demo_hip.hip", None)
        self.assertIsNotNone(build, holes)
        self.assertEqual("gfx942", build["arch"])
        self.assertEqual(HIPCC, build["compiler"])
        self.assertIn("--offload-arch=gfx942", build["cuda_cflags"])
        self.assertIn("-fno-gpu-rdc", build["cuda_cflags"],
                      "the flags are replayed as recorded; dropping one compiles a different program")
        self.assertTrue(build["source"].endswith("demo_hip.hip"))


class ProvenanceTest(unittest.TestCase):
    """The stamp is how a verifier learns which build an assembly file belongs to.

    Without it `restore` would have to guess, and this task tree holds the candidate AND the
    immutable rocBLAS oracle -- so a guess produces a real object built from the wrong program,
    which nothing downstream can tell from the right one.
    """

    def test_a_stamp_round_trips_through_the_first_line(self):
        text = RT.stamp_provenance("\t.text\n\ts_endpgm\n", "custom_gemm.hip", "gfx942")
        self.assertTrue(text.startswith("; "), "the stamp must be an AMDGCN comment")
        self.assertEqual({"source_file": "custom_gemm.hip", "arch": "gfx942"},
                         RT.read_provenance(text))

    def test_the_stamp_assembles_to_nothing(self):
        """`;` is AMDGCN's comment character. If that ever stopped being true the round-trip gate
        would fail on an unedited file, which is the loudest possible place for it to surface --
        but the cheap check belongs here."""
        first = RT.stamp_provenance("x", "a.hip", "gfx942").splitlines()[0]
        self.assertEqual([], RT.instruction_lines(first))

    def test_an_unstamped_file_reads_as_none_rather_than_a_default(self):
        self.assertIsNone(RT.read_provenance("\t.text\n\ts_endpgm\n"))

    def test_a_malformed_stamp_is_none_rather_than_a_partial_dict(self):
        self.assertIsNone(RT.read_provenance(f"; {RT.SCHEMA_PROVENANCE} {{not json\n"))
        self.assertIsNone(RT.read_provenance(f'; {RT.SCHEMA_PROVENANCE} {{"arch": "gfx942"}}\n'),
                          "a stamp naming no source_file resolves nothing and must not pass")

    def test_the_stamp_is_not_searched_for_beyond_the_head_of_the_file(self):
        """An unbounded search would find the string inside a `.asciz` and call it provenance."""
        buried = "\n" * (RT._PROVENANCE_SEARCH_LINES + 2) + RT.stamp_provenance("x", "a.hip", "g")
        self.assertIsNone(RT.read_provenance(buried))


class EditDetectionTest(unittest.TestCase):
    """`instruction_lines` answers "was this actually edited", and must not answer "was this
    emitted somewhere else"."""

    def test_directives_labels_and_comments_are_not_content(self):
        self.assertEqual(
            ["s_nop 0", "s_endpgm"],
            RT.instruction_lines('\t.file "/a/b/x.hip"\n.LBB0_1:\n\ts_nop 0 ; pad\n\ts_endpgm\n'))

    def test_the_same_program_from_two_workspaces_compares_equal(self):
        """`hipcc -S` writes the absolute source path into `.file` and the toolchain into
        `.ident`. A byte comparison would report every candidate as edited -- including the
        pristine one this check exists to catch."""
        a = '\t.file "/ws_a/src/x.hip"\n\t.ident "clang 22"\n\ts_endpgm\n'
        b = '\t.file "/ws_b/src/x.hip"\n\t.ident "clang 22"\n\ts_endpgm\n'
        self.assertNotEqual(a, b)
        self.assertEqual(RT.instruction_lines(a), RT.instruction_lines(b))

    def test_an_inserted_instruction_is_seen(self):
        base = "\ts_endpgm\n"
        self.assertNotEqual(RT.instruction_lines(base),
                            RT.instruction_lines("\ts_nop 0\n" + base))


class IncumbentTest(unittest.TestCase):
    """Whether a phase2 round is CONTINUING or STARTING, which is the question that decides whether
    the assembly search can compound at all.

    Before this action existed the engineer was told to `emit`, unconditionally -- and `emit`
    regenerates the assembly FROM SOURCE, so every round discarded the edits the previous round
    banked and restarted from the compiler's output. The search could re-find one window's win
    forever and never stack two. None of this needs a toolchain: it is a read.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="asmrt_i_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.src = make_tree(self.tmp)
        self.hip = self.src / "demo_hip.hip"

    def stamped(self, name="k.s", sha=None, body="\ts_nop 0\n"):
        path = self.src / name
        path.write_text(
            RT.stamp_provenance(body, "demo_hip.hip", "gfx942",
                                RT.source_sha(self.hip) if sha is None else sha),
            encoding="utf-8")
        return path

    def test_a_tree_with_no_assembly_is_a_cold_start_not_a_failure(self):
        """The normal state of the first phase2 round in a cycle. Reporting it as a hole would push
        the engineer into treating it as something to work around."""
        code, out = RT.incumbent(self.src, [self.src], None)
        self.assertEqual(RT.EXIT_OK, code, out)
        self.assertFalse(out["continue_from_assembly"])
        self.assertEqual(0, out["assembly_found"])
        self.assertEqual([], out["holes"], "a cold start is not a hole")

    def test_an_assembly_emitted_from_this_source_is_continued_from(self):
        self.stamped()
        code, out = RT.incumbent(self.src, [self.src], None)
        self.assertEqual(RT.EXIT_OK, code, out)
        self.assertTrue(out["continue_from_assembly"], out)
        row = out["incumbent"][0]
        self.assertTrue(row["usable"]) and self.assertFalse(row["superseded"])

    def test_a_source_edit_since_the_emit_supersedes_the_assembly(self):
        """During phase2 the source is read-only, so this cannot happen mid-phase. It happens when
        the ladder LEAVES phase2: a phase1 or deep round edits the file, and the `.s` becomes a
        description of a program that no longer exists."""
        self.stamped()
        self.hip.write_text(KERNEL + "\n// a phase1 round landed here\n", encoding="utf-8")
        code, out = RT.incumbent(self.src, [self.src], None)
        self.assertEqual(RT.EXIT_OK, code, out)
        self.assertFalse(out["continue_from_assembly"])
        self.assertTrue(out["incumbent"][0]["superseded"], out)

    def test_a_superseded_assembly_is_refused_by_restore_rather_than_rebuilt(self):
        """The quiet outcome is the dangerous one: rebuilding would overwrite the object compiled
        from the NEW source with code generated from the OLD, reverting the source edit, passing
        correctness, and reading as a measurement of the edit it just undid."""
        asm = self.stamped()
        self.hip.write_text(KERNEL + "\n// a phase1 round landed here\n", encoding="utf-8")
        ok, row = RT.restore_one(self.src, asm, "demo_hip.hip", {}, self.tmp / "s")
        self.assertFalse(ok)
        self.assertTrue(row.get("superseded"), row)
        self.assertTrue(any("source_superseded" in h for h in row.get("holes", [])), row)

    def test_an_assembly_with_no_source_sha_still_continues(self):
        """Backward compatibility. A `.s` emitted before the stamp carried a digest cannot be shown
        stale, and refusing it would strand an in-flight lane; unknown is not superseded."""
        self.stamped(sha="")
        code, out = RT.incumbent(self.src, [self.src], None)
        self.assertEqual(RT.EXIT_OK, code, out)
        self.assertTrue(out["continue_from_assembly"], out)
        self.assertFalse(out["incumbent"][0]["superseded"])


class RestoreRefusalTest(unittest.TestCase):
    """Every way `restore` cannot know what to rebuild is a refusal naming which way.

    None of these need a toolchain: they are the decisions that pick WHICH program gets rebuilt,
    and they are exactly the ones whose failure is silent -- a restore that quietly does nothing
    leaves the source-compiled object in place, and the round reports the parent's time as the
    candidate's.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="asmrt_r_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.scratch = self.tmp / "scratch"
        self.scratch.mkdir()

    def test_finding_no_assembly_is_a_hole_not_a_silent_success(self):
        """`restore` is called because the patch carried a `.s`. Finding none means it did not
        land where the scan looks, and returning 0 there would certify the unpatched kernel."""
        src = make_tree(self.tmp)
        code, out = RT.restore(src, [src], None, {}, self.scratch)
        self.assertEqual(RT.EXIT_HOLE, code, out)
        self.assertTrue(any("asm:none_found" in h for h in out["holes"]), out["holes"])

    def test_an_unstamped_assembly_with_no_source_file_is_refused(self):
        src = make_tree(self.tmp)
        (src / "hand.s").write_text("\ts_endpgm\n", encoding="utf-8")
        code, out = RT.restore(src, [src], None, {}, self.scratch)
        self.assertEqual(RT.EXIT_HOLE, code, out)
        self.assertTrue(any("asm:no_provenance" in h for h in out["holes"]), out["holes"])
        self.assertEqual([str(src / "hand.s")], out["unresolved"])

    def test_two_unstamped_files_cannot_share_one_source_file(self):
        src = make_tree(self.tmp)
        (src / "a.s").write_text("\ts_endpgm\n", encoding="utf-8")
        (src / "b.s").write_text("\ts_endpgm\n", encoding="utf-8")
        code, out = RT.restore(src, [src], "demo_hip.hip", {}, self.scratch)
        self.assertEqual(RT.EXIT_HOLE, code, out)
        self.assertTrue(any("ambiguous_source_file" in h for h in out["holes"]), out["holes"])

    def test_unstamped_files_are_reported_rather_than_filtered_away(self):
        """Dropping them would make the one case worth refusing read exactly like a clean tree."""
        src = make_tree(self.tmp)
        (src / "hand.s").write_text("\ts_endpgm\n", encoding="utf-8")
        found, _ = RT.find_assembly([src])
        self.assertEqual(1, len(found))
        self.assertIsNone(found[0][1])

    def test_two_assembly_files_for_one_unit_are_refused_rather_than_ordered(self):
        """The EXPECTED shape, not an edge case: the role tells the engineer to keep a pristine
        copy to diff against, and `git add -A` sweeps up whatever is in the workspace. Both files
        resolve to the same build edge and both rebuilds write the same object, so the winner is
        whichever sorted last -- and with the names this file's own RoundTripTest uses (`e.edit.s`
        and `e.s`) that is the PRISTINE one, which silently reverts the edit and measures the
        candidate as its own parent."""
        src = make_tree(self.tmp)
        (src / "e.edit.s").write_text(
            RT.stamp_provenance("\ts_nop 0\n", "demo_hip.hip", "gfx942"), encoding="utf-8")
        (src / "e.s").write_text(
            RT.stamp_provenance("", "demo_hip.hip", "gfx942"), encoding="utf-8")
        code, out = RT.restore(src, [src], None, {}, self.scratch)
        self.assertEqual(RT.EXIT_HOLE, code, out)
        self.assertTrue(any("multiple_assembly_for_one_unit" in h for h in out["holes"]),
                        out["holes"])
        self.assertEqual(2, len(out["unresolved"]), out)
        self.assertEqual([], out["restored"],
                         "nothing may be rebuilt when the tool cannot tell which file is the edit")

    def test_two_assembly_files_for_DIFFERENT_units_are_both_restored(self):
        """The refusal is per translation unit, not a cap on how many files a patch may carry."""
        src = make_tree(self.tmp)
        (src / "a.s").write_text(RT.stamp_provenance("", "a_hip.hip", "gfx942"), encoding="utf-8")
        (src / "b.s").write_text(RT.stamp_provenance("", "b_hip.hip", "gfx942"), encoding="utf-8")
        code, out = RT.restore(src, [src], None, {}, self.scratch)
        # Both resolve to units this fixture's build does not compile, so each fails on its own
        # terms -- but neither is refused for clashing, which is what this asserts.
        self.assertFalse(any("multiple_assembly_for_one_unit" in h for h in out["holes"]),
                         out["holes"])
        self.assertEqual(2, len(out["restored"]), out)

    def test_an_arch_disagreement_is_refused_rather_than_reconciled(self):
        """`rebuild` reassembles against the BUILD's arch, so a `.s` emitted for another one would
        encode against a different ISA -- and where that does not fail outright it loads."""
        src = make_tree(self.tmp, arch="gfx942")
        asm = src / "k.s"
        asm.write_text(RT.stamp_provenance("\ts_endpgm\n", "demo_hip.hip", "gfx90a"),
                       encoding="utf-8")
        ok, row = RT.restore_one(src, asm, "demo_hip.hip", {}, self.scratch)
        self.assertFalse(ok)
        self.assertTrue(any("arch_mismatch" in h for h in row.get("holes", [])), row)


class CodeObjectPickerTest(unittest.TestCase):
    def test_no_amdgpu_object_reads_as_none_rather_than_empty(self):
        blob = self.tmp / "notelf.bin"
        blob.write_bytes(b"\x00" * 512)
        self.assertIsNone(RT.device_instructions({"llvm-objdump": "llvm-objdump"}, blob, "gfx942"))

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="asmrt_p_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)


@unittest.skipUnless(HAVE_TOOLCHAIN, "ROCm toolchain absent; the compiling half cannot run here")
class RoundTripTest(unittest.TestCase):
    """The claim the whole phase rests on, exercised rather than asserted."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="asmrt_rt_"))
        cls.src = make_tree(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def rt(self, *args):
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "asm_roundtrip.py"), *args,
             "--source-root", str(self.src), "--source-file", "demo_hip.hip"],
            capture_output=True, text=True)
        try:
            return proc.returncode, json.loads(proc.stdout)
        except json.JSONDecodeError:
            self.fail(f"non-JSON output (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}")

    def test_an_unedited_round_trip_reproduces_the_builds_own_device_code(self):
        rc, out = self.rt("verify")
        self.assertEqual(RT.EXIT_OK, rc, out)
        self.assertTrue(out["round_trip_identical"],
                        f"first difference: {out.get('first_difference')}")
        self.assertGreater(out["reference_instructions"], 0)

    def test_the_emitted_assembly_carries_the_declaration_layer(self):
        asm = self.tmp / "demo.s"
        rc, out = self.rt("emit", "--out", str(asm))
        self.assertEqual(RT.EXIT_OK, rc, out)
        # This is what makes `hipcc -S` cheaper than the paper's recovery step: raw disassembly has
        # neither of these, and without them the result is not reassemblable.
        self.assertTrue(out["has_amdhsa_kernel"])
        self.assertTrue(out["has_amdgpu_metadata"])
        self.assertGreater(out["lines"], 20)

    def test_an_edit_reaches_the_rebuilt_host_object(self):
        asm = self.tmp / "e.s"
        rc, _ = self.rt("emit", "--out", str(asm))
        self.assertEqual(RT.EXIT_OK, rc)

        lines, done = [], False
        for line in asm.read_text(encoding="utf-8").splitlines():
            lines.append(line)
            if not done and line.startswith("demo_axpy:"):
                lines.append("\ts_nop 0")
                done = True
        self.assertTrue(done, "the kernel label was not found in the emitted assembly")
        edited = self.tmp / "e.edit.s"
        edited.write_text("\n".join(lines) + "\n", encoding="utf-8")

        obj = self.tmp / "patched.o"
        rc, out = self.rt("rebuild", "--asm", str(edited), "--out", str(obj))
        self.assertEqual(RT.EXIT_OK, rc, out)
        self.assertTrue(obj.is_file())

        tool, missing = RT.tools()
        self.assertFalse(missing, missing)
        insns = RT.device_instructions(tool, obj, "gfx942")
        self.assertTrue(insns, "no device code in the rebuilt object")
        # Position, not count: the code object is padded to alignment, so an inserted instruction is
        # absorbed by the padding and a count-based check reads as "the edit was dropped".
        self.assertEqual("s_nop 0", insns[0],
                         "the injected instruction must be first in the rebuilt kernel; if it is "
                         "not, -fcuda-include-gpubinary did not take our fatbin")

    def test_rebuild_refuses_an_assembly_file_that_does_not_exist(self):
        rc, out = self.rt("rebuild", "--asm", str(self.tmp / "absent.s"))
        self.assertEqual(RT.EXIT_ERROR, rc)
        self.assertIn("--asm", out["error"])

    def test_emit_stamps_the_translation_unit_it_came_from(self):
        asm = self.tmp / "stamped.s"
        rc, out = self.rt("emit", "--out", str(asm))
        self.assertEqual(RT.EXIT_OK, rc, out)
        self.assertEqual("demo_hip.hip", out["provenance"]["source_file"])
        self.assertEqual("gfx942", out["provenance"]["arch"])
        stamp = RT.read_provenance(asm.read_text(encoding="utf-8"))
        self.assertEqual(out["provenance"], stamp,
                         "the stamp in the file must match the one emit reported")


@unittest.skipUnless(HAVE_TOOLCHAIN, "ROCm toolchain absent; the compiling half cannot run here")
class RestoreTest(unittest.TestCase):
    """The verifier's entry point, exercised against a tree shaped like a real candidate.

    This is the guard for the failure the whole module exists to prevent, and it is worth stating
    plainly: NO EDGE IN `build.ninja` COMPILES A `.s`. Before `restore` existed, applying an
    assembly patch and building normally left the assembly unread, compiled the untouched `.hip`,
    and timed the parent -- while every other receipt in the verifier reported a clean pass.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="asmrt_ro_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.src = make_tree(self.tmp)

    def run_restore(self, *extra):
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "asm_roundtrip.py"), "restore",
             "--source-root", str(self.src), *extra],
            capture_output=True, text=True)
        try:
            return proc.returncode, json.loads(proc.stdout)
        except json.JSONDecodeError:
            self.fail(f"non-JSON output (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}")

    def emit_into_tree(self, name="kernel.s") -> Path:
        asm = self.src / name
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "asm_roundtrip.py"), "emit",
             "--source-root", str(self.src), "--source-file", "demo_hip.hip",
             "--out", str(asm)], capture_output=True, text=True)
        self.assertEqual(RT.EXIT_OK, proc.returncode, proc.stdout + proc.stderr)
        return asm

    def test_an_edited_assembly_is_rebuilt_into_the_object_the_build_links(self):
        asm = self.emit_into_tree()
        lines, done = [], False
        for line in asm.read_text(encoding="utf-8").splitlines():
            lines.append(line)
            if not done and line.startswith("demo_axpy:"):
                lines.append("\ts_nop 0")
                done = True
        self.assertTrue(done, "the kernel label was not found in the emitted assembly")
        asm.write_text("\n".join(lines) + "\n", encoding="utf-8")

        rc, out = self.run_restore()
        self.assertEqual(RT.EXIT_OK, rc, out)
        self.assertEqual(1, out["assembly_found"], out)
        row = out["restored"][0]
        self.assertEqual("stamp", row["provenance"],
                         "restore must resolve from the stamp, not from a --source-file guess")
        self.assertTrue(row["edited"], row)

        # The object it wrote is the one the build edge names, so a normal link picks it up with
        # no further step. That path -- not the caller's cwd -- is the whole reason this lands.
        obj = Path(row["object_path"])
        self.assertTrue(obj.is_file(), row)
        self.assertEqual(".torch_ext", obj.parent.parent.name)
        tool, missing = RT.tools()
        self.assertFalse(missing, missing)
        insns = RT.device_instructions(tool, obj, "gfx942")
        self.assertEqual("s_nop 0", insns[0],
                         "the edit is not in the object the build links, so a verifier following "
                         "this path would still have timed the unpatched kernel")

    def test_a_pristine_assembly_is_reported_as_unedited(self):
        """The assembly phase's `hip_twin_sync`. An engineer who saved the file without editing it
        produces a candidate byte-identical to its parent; measured and reported as `no effect`
        that closes a direction nothing ever tried."""
        self.emit_into_tree()
        rc, out = self.run_restore()
        self.assertEqual(RT.EXIT_OK, rc, out)
        self.assertFalse(out["restored"][0]["edited"], out)

    def test_restore_finds_assembly_beside_the_source_root_too(self):
        """`--scan` defaults to the source root AND its parent, because the engineer may save the
        `.s` on either side of `src/` and a patch does not say which."""
        asm = self.emit_into_tree()
        moved = self.src.parent / "kernel.s"
        asm.rename(moved)
        rc, out = self.run_restore()
        self.assertEqual(RT.EXIT_OK, rc, out)
        self.assertEqual(1, out["assembly_found"], out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
