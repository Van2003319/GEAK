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


if __name__ == "__main__":
    unittest.main(verbosity=2)
