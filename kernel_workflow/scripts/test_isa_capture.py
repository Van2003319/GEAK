#!/usr/bin/env python3
"""Tests for `isa_capture.py`.

No GPU, no ROCm, no compiler. The ELF scan is exercised against SYNTHESIZED
headers, and the two ROCm tools are injected as a fake runner, so the parts that
can be wrong offline are all checked offline.

Two of these matter more than the rest.

`ElfScanTest` is the load-bearing one. The scan replaced a `roc-obj-ls` text
parse precisely so it could be tested, and its dangerous failure is not missing an
object -- that shows up immediately as an empty archive -- but ACCEPTING a
`\\x7fELF` that is not an AMDGPU code object, or slicing a valid one at the wrong
length. Either yields a disassembly of garbage, and garbage disassembles into
plausible-looking mnemonics rather than into an error.

`ArchiveContractTest` is the one that keeps the two halves honest: it runs a
captured archive straight into `isa_signals.build_signals` and asserts real
kernels come out. The archive layout is a contract between two files, and a
contract with no test is where the writer starts emitting `0.disasm` while the
reader still globs `*.disasm.txt`.
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import isa_capture as C  # noqa: E402
import isa_signals as S  # noqa: E402
import test_isa_signals as F  # noqa: E402  (single source of fixture truth)

EM_X86_64 = 62
EM_CUDA = 190  # an NVIDIA cubin is an ELF too, and this is why it is not read here

# Tool paths are injected rather than discovered. Without this the `runner` below
# is never reached on a box with no ROCm -- discovery fails first, capture skips
# straight to a HOLE, and every assertion about parsing passes vacuously.
FAKE_TOOLS = {"llvm-objdump": "/fake/llvm-objdump", "llvm-readelf": "/fake/llvm-readelf"}
OBJDUMP_ONLY = {"llvm-objdump": "/fake/llvm-objdump", "llvm-readelf": None}


def fake_code_object(length: int = 256, machine: int = C.EM_AMDGPU, etype: int = 3,
                     ei_class: int = 2, ei_data: int = 1,
                     declared_length: int | None = None) -> bytes:
    """A byte-exact ELF64 header whose declared extent is `declared_length or length`.

    Section table placed at the very end with one 64-byte entry, so
    `e_shoff + e_shentsize * e_shnum` is the object's length -- the same
    arithmetic `elf_object_span` does.
    """
    assert length >= 128
    extent = length if declared_length is None else declared_length
    header = bytearray(64)
    header[0:4] = b"\x7fELF"
    header[4] = ei_class
    header[5] = ei_data
    header[6] = 1
    header[7] = 64  # ELFOSABI_AMDGPU_HSA
    struct.pack_into("<HH", header, 16, etype, machine)
    struct.pack_into("<I", header, 20, 1)
    struct.pack_into("<QQ", header, 32, 0, extent - 64)   # e_phoff, e_shoff
    struct.pack_into("<HHHHH", header, 52, 64, 0, 0, 64, 1)
    return bytes(header) + bytes(length - 64)


def fake_runner(notes_text: str = "", disasm_text: str = "",
                notes_code: int = 0, disasm_code: int = 0):
    """Stand-in for llvm-readelf / llvm-objdump, dispatched on the real argv."""
    def run(argv):
        if "--notes" in argv:
            return notes_code, notes_text, "" if notes_code == 0 else "readelf boom"
        if "-d" in argv:
            return disasm_code, disasm_text, "" if disasm_code == 0 else "objdump boom"
        raise AssertionError(f"unexpected tool invocation: {argv}")
    return run


class ElfScanTest(unittest.TestCase):
    def test_accepts_a_wellformed_amdgpu_object_and_reports_its_declared_length(self):
        blob = fake_code_object(length=512)
        self.assertEqual(C.elf_object_span(blob, 0), (0, 512))

    def test_rejects_a_host_object(self):
        self.assertIsNone(C.elf_object_span(fake_code_object(machine=EM_X86_64), 0))

    def test_rejects_an_nvidia_cubin(self):
        """The mechanical reason this tool is HIP-only and `cuda` is not in the
        lane's ISA_LANGUAGES. An NVIDIA cubin is a perfectly well-formed ELF whose
        `e_machine` is EM_CUDA, so a cuda lane would scan a real fat binary, match
        no code object, and produce an archive reporting no device code — which
        reads exactly like a clean capture of a simple kernel. Nothing downstream
        could tell the difference, which is why the exclusion is a naming decision
        pinned by a probe rather than something left to notice at runtime."""
        self.assertIsNone(C.elf_object_span(fake_code_object(machine=EM_CUDA), 0))
        blob = b"\x00" * 16 + fake_code_object(machine=EM_CUDA, length=256)
        self.assertEqual(list(C.iter_code_objects(blob)), [],
                         "a whole cubin must yield no AMDGPU code object, not a "
                         "mis-sliced one that would disassemble into noise")

    def test_rejects_32_bit_and_big_endian(self):
        self.assertIsNone(C.elf_object_span(fake_code_object(ei_class=1), 0))
        self.assertIsNone(C.elf_object_span(fake_code_object(ei_data=2), 0))

    def test_rejects_an_unexpected_elf_type(self):
        self.assertIsNone(C.elf_object_span(fake_code_object(etype=4), 0))

    def test_rejects_a_truncated_header(self):
        self.assertIsNone(C.elf_object_span(fake_code_object()[:40], 0))

    def test_rejects_a_length_that_runs_past_the_blob(self):
        """The dangerous one. A declared extent longer than what is present would
        slice in trailing host bytes, and llvm-objdump renders those as
        instructions rather than refusing them."""
        blob = fake_code_object(length=256, declared_length=4096)
        self.assertIsNone(C.elf_object_span(blob, 0))

    def test_a_bare_magic_string_is_not_an_object(self):
        self.assertIsNone(C.elf_object_span(b"\x7fELF" + b"junk" * 40, 0))

    def test_finds_two_embedded_objects_past_a_host_header_and_padding(self):
        host = fake_code_object(machine=EM_X86_64, length=128)
        first = fake_code_object(length=256)
        second = fake_code_object(length=192)
        blob = host + b"\x00" * 37 + first + b"pad" + second + b"\x00" * 11
        spans = list(C.iter_code_objects(blob))
        self.assertEqual(len(spans), 2)
        self.assertEqual([length for _off, length in spans], [256, 192])
        (off0, len0), (off1, len1) = spans
        self.assertEqual(blob[off0:off0 + len0], first)
        self.assertEqual(blob[off1:off1 + len1], second)

    def test_spans_do_not_overlap_when_an_object_contains_the_magic_in_its_body(self):
        """The scan resumes AFTER an accepted object, so a `\\x7fELF` byte sequence
        inside a string table cannot be reported as a nested code object."""
        inner = fake_code_object(length=256)
        blob = bytearray(inner)
        blob[100:104] = b"\x7fELF"
        spans = list(C.iter_code_objects(bytes(blob)))
        self.assertEqual(spans, [(0, 256)])

    def test_an_empty_blob_yields_nothing_rather_than_raising(self):
        self.assertEqual(list(C.iter_code_objects(b"")), [])


class ToolResolutionTest(unittest.TestCase):
    def test_path_wins(self):
        with mock.patch.object(C.shutil, "which", return_value="/usr/bin/llvm-objdump"):
            self.assertEqual(C.resolve_tool("llvm-objdump"), "/usr/bin/llvm-objdump")

    def test_falls_back_to_rocm_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool_dir = Path(tmp) / "llvm" / "bin"
            tool_dir.mkdir(parents=True)
            tool = tool_dir / "llvm-readelf"
            tool.write_text("#!/bin/sh\n", encoding="utf-8")
            tool.chmod(0o755)
            with mock.patch.object(C.shutil, "which", return_value=None), \
                    mock.patch.dict(C.os.environ, {"ROCM_PATH": tmp}):
                self.assertEqual(C.resolve_tool("llvm-readelf"), str(tool))

    def test_a_missing_tool_is_none_not_a_guessed_path(self):
        with mock.patch.object(C.shutil, "which", return_value=None), \
                mock.patch.dict(C.os.environ, {"ROCM_PATH": "/nope"}, clear=False):
            self.assertIsNone(C.resolve_tool("llvm-not-a-tool"))

    def test_an_explicit_override_beats_discovery(self):
        """The `--objdump` / `--readelf` flags exist for a container that carries
        ROCm somewhere neither PATH nor /opt/rocm names."""
        with mock.patch.object(C, "resolve_tool", return_value="/discovered/tool"):
            tools = C.resolve_tools(objdump="/pinned/llvm-objdump")
        self.assertEqual(tools["llvm-objdump"], "/pinned/llvm-objdump")
        self.assertEqual(tools["llvm-readelf"], "/discovered/tool")


class ArtifactDiscoveryTest(unittest.TestCase):
    def test_picks_device_bearing_suffixes_and_ignores_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a.so", "b.hsaco", "c.o", "readme.md", "kernel.hip", "d.txt"):
                (root / name).write_bytes(b"x")
            found, holes = C.find_artifacts([root])
        self.assertEqual(sorted(p.name for p in found), ["a.so", "b.hsaco", "c.o"])
        self.assertEqual(holes, [])

    def test_a_missing_scan_root_is_named(self):
        found, holes = C.find_artifacts([Path("/definitely/not/here")])
        self.assertEqual(found, [])
        self.assertTrue(any(h.startswith("scan:missing") for h in holes))

    def test_an_explicit_file_is_accepted_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "libfoo.so"
            target.write_bytes(b"x")
            found, holes = C.find_artifacts([target])
        self.assertEqual([p.name for p in found], ["libfoo.so"])
        self.assertEqual(holes, [])

    def test_the_same_file_reached_twice_is_captured_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.so").write_bytes(b"x")
            found, _holes = C.find_artifacts([root, root / "a.so"])
        self.assertEqual(len(found), 1)


class ArchTest(unittest.TestCase):
    def test_reads_the_target_triple(self):
        self.assertEqual(C.arch_from_text(F.notes()), "gfx942")
        self.assertEqual(
            C.arch_from_text('\t.amdgcn_target "amdgcn-amd-amdhsa--gfx950"'), "gfx950")

    def test_no_triple_is_none_not_a_default(self):
        self.assertIsNone(C.arch_from_text("nothing to see"))


class CaptureTest(unittest.TestCase):
    def _tree(self, tmp: Path, blob: bytes | None = None) -> Path:
        src = tmp / "ws"
        (src / "build").mkdir(parents=True)
        (src / "custom_gemm.hip").write_text("__global__ void k(){}\n", encoding="utf-8")
        if blob is not None:
            (src / "build" / "kernel.so").write_bytes(blob)
        return src

    def test_happy_path_writes_a_complete_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = self._tree(tmp, fake_code_object(length=256))
            manifest = C.capture(tmp / "ir", src, [src], arch="gfx942",
                                 runner=fake_runner(F.notes(), F.DISASM), tools=FAKE_TOOLS)
        self.assertEqual(manifest["exit_code"], C.EXIT_OK, manifest["holes"])
        self.assertEqual(manifest["object_count"], 1)
        self.assertEqual(manifest["arch"], "gfx942")
        self.assertEqual(manifest["arches_observed"], ["gfx942"])
        self.assertRegex(manifest["source_hash"] or "", r"^[0-9a-f]{64}$")
        entry = manifest["objects"][0]
        self.assertEqual(entry["disasm"], "objects/0.disasm.txt")
        self.assertEqual(entry["notes"], "objects/0.notes.txt")
        self.assertEqual(entry["bytes"], 256)

    def test_the_source_hash_tracks_the_source_not_the_binary(self):
        """Two captures of the same tree agree; editing the source changes it. This
        is what ties a signal to a candidate, and finding (144) is the standing
        reminder of what accurate numbers about the wrong object cost."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = self._tree(tmp, fake_code_object())
            run = fake_runner(F.notes(), F.DISASM)
            first = C.capture(tmp / "a", src, [src], runner=run, tools=FAKE_TOOLS)
            second = C.capture(tmp / "b", src, [src], runner=run, tools=FAKE_TOOLS)
            self.assertEqual(first["source_hash"], second["source_hash"])
            (src / "custom_gemm.hip").write_text("__global__ void k(){int q=1;}\n",
                                                 encoding="utf-8")
            third = C.capture(tmp / "c", src, [src], runner=run, tools=FAKE_TOOLS)
        self.assertNotEqual(first["source_hash"], third["source_hash"])

    def test_no_code_object_is_a_hole_not_an_empty_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = self._tree(tmp, b"not an elf at all")
            manifest = C.capture(tmp / "ir", src, [src],
                                 runner=fake_runner(F.notes(), F.DISASM), tools=FAKE_TOOLS)
        self.assertEqual(manifest["exit_code"], C.EXIT_HOLE)
        self.assertTrue(any(h.startswith("capture:nothing") for h in manifest["holes"]))

    def test_a_missing_readelf_is_partial_and_says_budgets_are_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = self._tree(tmp, fake_code_object())
            manifest = C.capture(tmp / "ir", src, [src],
                                 runner=fake_runner("", F.DISASM), tools=OBJDUMP_ONLY)
        self.assertEqual(manifest["exit_code"], C.EXIT_PARTIAL)
        self.assertTrue(any("llvm-readelf missing" in h for h in manifest["holes"]))
        self.assertIsNone(manifest["tools"]["llvm-readelf"])

    def test_a_failing_objdump_is_reported_not_swallowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = self._tree(tmp, fake_code_object())
            manifest = C.capture(tmp / "ir", src, [src],
                                 runner=fake_runner(F.notes(), "", disasm_code=1),
                                 tools=FAKE_TOOLS)
        self.assertEqual(manifest["exit_code"], C.EXIT_HOLE)
        self.assertTrue(any(h.startswith("disasm:failed") for h in manifest["holes"]))

    def test_an_existing_archive_is_not_overwritten_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = self._tree(tmp, fake_code_object())
            out = tmp / "ir"
            out.mkdir()
            (out / "stale.txt").write_text("previous round", encoding="utf-8")
            manifest = C.capture(out, src, [src], runner=fake_runner(F.notes(), F.DISASM),
                                 tools=FAKE_TOOLS)
            self.assertEqual(manifest["exit_code"], C.EXIT_ERROR)
            self.assertTrue(any(h.startswith("archive:not_empty") for h in manifest["holes"]))
            forced = C.capture(out, src, [src], force=True,
                               runner=fake_runner(F.notes(), F.DISASM), tools=FAKE_TOOLS)
        self.assertEqual(forced["exit_code"], C.EXIT_OK, forced["holes"])

    def test_arch_filter_sets_aside_other_targets_and_says_it_did(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = self._tree(tmp, fake_code_object(length=256) + fake_code_object(length=192))
            out = tmp / "ir"
            manifest = C.capture(out, src, [src], arch="gfx950",
                                 runner=fake_runner(F.notes(), F.DISASM), tools=FAKE_TOOLS)
            # Asserted INSIDE the temp directory's lifetime. Outside it every
            # `exists()` is False and an `assertFalse(...exists())` passes for the
            # wrong reason -- which is how the first version of this test read as
            # coverage of the rename while checking a deleted directory.
            self.assertEqual(manifest["object_count"], 0)
            self.assertTrue(any(h.startswith("arch:filtered") for h in manifest["holes"]))
            self.assertFalse((out / "objects" / "0.disasm.txt").exists())
            self.assertTrue((out / "objects" / "0.disasm.other-arch").exists(),
                            "a set-aside dump is renamed, not deleted, so a wrong-target "
                            "capture is diagnosable")
            self.assertTrue((out / "objects" / "1.disasm.other-arch").exists())

    def test_manifest_on_disk_is_canonical_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = self._tree(tmp, fake_code_object())
            out = tmp / "ir"
            C.capture(out, src, [src], runner=fake_runner(F.notes(), F.DISASM),
                      tools=FAKE_TOOLS)
            text = (out / "manifest.json").read_text(encoding="utf-8")
        keys = list(json.loads(text).keys())
        self.assertEqual(keys, sorted(keys))


class ArchiveContractTest(unittest.TestCase):
    """The layout `isa_capture.py` writes is the layout `isa_signals.py` reads."""

    def test_a_captured_archive_reads_back_as_real_kernel_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "ws"
            (src / "build").mkdir(parents=True)
            (src / "k.hip").write_text("x\n", encoding="utf-8")
            (src / "build" / "kernel.so").write_bytes(fake_code_object(length=256))
            out = tmp / "ir"
            manifest = C.capture(out, src, [src], arch="gfx942",
                                 runner=fake_runner(F.notes(scratch=512), F.DISASM),
                                 tools=FAKE_TOOLS)
            self.assertEqual(manifest["exit_code"], C.EXIT_OK, manifest["holes"])
            signals = S.build_signals(out)
            checks = S.run_checks(signals)

        self.assertEqual(signals["exit_code"], S.EXIT_OK, signals["unavailable"])
        self.assertEqual(signals["kernel_count"], 1)
        self.assertEqual(signals["arch"], "gfx942")
        self.assertEqual(signals["source_hash"], manifest["source_hash"],
                         "the reader must pick the writer's source_hash out of the manifest, "
                         "or the signals have no owner")
        kernel = signals["kernels"][0]
        self.assertEqual(kernel["name"], F.KERNEL)
        self.assertEqual(kernel["resources"]["scratch_bytes"], 512)
        self.assertEqual([f["rule"] for f in checks["findings"] if f["severity"] == "high"],
                         ["spill_to_scratch"])

    def test_a_hole_archive_reads_back_as_a_hole_not_as_a_clean_kernel(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "ws"
            src.mkdir()
            (src / "k.hip").write_text("x\n", encoding="utf-8")
            out = tmp / "ir"
            manifest = C.capture(out, src, [src], runner=fake_runner(F.notes(), F.DISASM),
                                 tools=FAKE_TOOLS)
            signals = S.build_signals(out)
        self.assertEqual(manifest["exit_code"], C.EXIT_HOLE)
        self.assertEqual(signals["exit_code"], S.EXIT_HOLE)
        self.assertEqual(signals["kernels"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
