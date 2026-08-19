#!/usr/bin/env python3
"""GPU-free tests for candidate_policy_scan.py."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).with_name("candidate_policy_scan.py")
SPEC = importlib.util.spec_from_file_location("candidate_policy_scan", SCRIPT)
assert SPEC and SPEC.loader
SCAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCAN
SPEC.loader.exec_module(SCAN)


class CandidatePolicyScanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, relative: str, data: str | bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            path.write_text(data, encoding="utf-8")
        return path

    def scan(self, *paths: Path, immutable=()):
        return SCAN.scan(paths or (self.root,), immutable)

    def rules(self, receipt):
        return {item["rule"] for item in receipt["findings"]}

    def test_allowed_hip_mfma_rocwmma_and_ordinary_check(self):
        self.write("kernel.cpp", r'''
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <rocwmma/rocwmma.hpp>
__global__ void kernel() { __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(); }
// check bounds and device status; compiler is hipcc.
''')
        receipt = self.scan()
        self.assertTrue(receipt["passed"], receipt["findings"])
        self.assertEqual([], receipt["findings"])

    def test_all_forbidden_text_families(self):
        cases = {
            "a.cpp": "rocblas_gemm_ex(); hipblasGemmEx(); hipblasLtMatmul(); TensileLibrary; MIOpenTensorOp();",
            "b.cpp": '#include <ck/tensor_operation/gpu/device/device_gemm.hpp>\nnamespace ck { } // Composable Kernel',
            "c.py": "torch.matmul(a,b); torch.mm(a,b); torch.bmm(a,b); torch.nn.functional.linear(a,w)",
            "d.py": "import ctypes; ctypes.CDLL(name); ctypes.cdll.LoadLibrary(name); load_library(name)",
            "link.txt": "-lrocblas -lhipblas -lhipblaslt -lMIOpen -lTensile",
        }
        for name, body in cases.items():
            self.write(name, body)
        rules = self.rules(self.scan())
        self.assertTrue({"rocblas", "hipblas", "hipblaslt", "tensile", "miopen",
                         "composable_kernel", "torch_matmul", "torch_linear",
                         "dynamic_loader"}.issubset(rules), rules)

    def test_immutable_oracle_and_baseline_are_explicit_exceptions(self):
        candidate = self.write("kernel_src/good.cpp", "hipLaunchKernelGGL(k, dim3(1), dim3(1), 0, 0);")
        oracle = self.write("baseline_src/reference.py", "torch.matmul(a, b); ctypes.CDLL('librocblas.so')")
        receipt = self.scan(candidate.parent, oracle.parent, immutable=(oracle.parent,))
        self.assertTrue(receipt["passed"], receipt["findings"])
        self.assertEqual([str(oracle.parent.resolve())], receipt["skipped_immutable"])

    def test_symlink_is_rejected_and_never_followed(self):
        outside = self.write("outside.cpp", "clean text")
        candidate = self.root / "candidate"
        candidate.mkdir()
        (candidate / "escape.cpp").symlink_to(outside)
        receipt = self.scan(candidate)
        self.assertFalse(receipt["passed"])
        self.assertIn("symlink", self.rules(receipt))
        self.assertFalse(any(item.get("sha256") for item in receipt["inspected"]
                             if item["path"].endswith("escape.cpp")))

    def test_non_utf8_binary_strings_are_scanned(self):
        self.write("blob.bin", b"\xff\xfe\x00prefix\x00librocblas.so.0\x00")
        receipt = self.scan()
        self.assertFalse(receipt["passed"])
        self.assertIn("rocblas", self.rules(receipt))
        record = next(item for item in receipt["inspected"] if item["path"].endswith("blob.bin"))
        self.assertEqual("binary", record["format"])

    @unittest.skipUnless(shutil.which("cc") and shutil.which("readelf") and shutil.which("strings"),
                         "host C compiler/binutils required")
    def test_elf_needed_symbol_and_strings_surfaces(self):
        source = self.write("bad.c", 'extern void rocblas_create_handle(void);\nconst char *x="hipblasLtMatmul";\nvoid f(void){rocblas_create_handle();}\n')
        artifact = self.root / "bad.so"
        subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(artifact)], check=True)
        receipt = self.scan(artifact)
        self.assertFalse(receipt["passed"])
        self.assertIn("rocblas", self.rules(receipt))
        self.assertIn("hipblaslt", self.rules(receipt))
        inspections = {item["inspection"] for item in receipt["findings"]}
        self.assertIn("elf-symbols", inspections)
        self.assertIn("binary-strings", inspections)

    @unittest.skipUnless(shutil.which("cc") and shutil.which("readelf") and shutil.which("strings"),
                         "host C compiler/binutils required")
    def test_elf_dt_needed_is_scanned(self):
        library_source = self.write("dep.c", "int rocblas_test_dep(void) { return 1; }\n")
        library = self.root / "librocblas.so"
        subprocess.run(["cc", "-shared", "-fPIC", str(library_source), "-o", str(library)], check=True)
        candidate_source = self.write("candidate.c", "int candidate(void) { return 1; }\n")
        candidate = self.root / "candidate.so"
        subprocess.run(["cc", "-shared", "-fPIC", str(candidate_source),
                        "-L", str(self.root), "-Wl,--no-as-needed", "-lrocblas",
                        "-o", str(candidate)], check=True)
        receipt = self.scan(candidate)
        needed = [item for item in receipt["findings"] if item["inspection"] == "elf-dynamic"]
        self.assertTrue(any(item["rule"] == "rocblas" for item in needed), receipt["findings"])

    @unittest.skipUnless(shutil.which("cc") and shutil.which("readelf") and shutil.which("strings"),
                         "host C compiler/binutils required")
    def test_clean_elf_is_allowed(self):
        source = self.write("good.c", "int device_helper(int x) { return x + 1; }\n")
        artifact = self.root / "good.so"
        subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(artifact)], check=True)
        receipt = self.scan(artifact)
        self.assertTrue(receipt["passed"], receipt["findings"])

    @unittest.skipUnless(shutil.which("cc") and shutil.which("readelf") and shutil.which("strings"),
                         "host C compiler/binutils required")
    def test_elf_tool_failure_fails_closed(self):
        source = self.write("good.c", "int f(void) { return 1; }\n")
        artifact = self.root / "good.so"
        subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(artifact)], check=True)
        real_run = SCAN.subprocess.run

        def fail_readelf(argv, **kwargs):
            if argv[0] == "readelf":
                return subprocess.CompletedProcess(argv, 7, "", "simulated failure")
            return real_run(argv, **kwargs)

        with mock.patch.object(SCAN.subprocess, "run", side_effect=fail_readelf):
            receipt = self.scan(artifact)
        self.assertFalse(receipt["passed"])
        self.assertIn("inspection_error", self.rules(receipt))

    def test_receipt_and_cli_are_deterministic(self):
        self.write("z.cpp", "hipDeviceSynchronize();")
        self.write("a.cpp", "torch.mm(a, b)")
        first = self.scan()
        second = self.scan()
        self.assertEqual(first, second)
        payload = json.dumps(first, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        run = subprocess.run([sys.executable, str(SCRIPT), str(self.root)], text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(1, run.returncode)
        self.assertEqual(payload, run.stdout)
        self.assertEqual("", run.stderr)

    def test_cli_writes_receipt_and_passes_clean_candidate(self):
        candidate = self.write("candidate.cpp", "// normal HIP device kernel")
        receipt_path = self.root / "receipt.json"
        run = subprocess.run([sys.executable, str(SCRIPT), "--candidate", str(candidate),
                              "--output", str(receipt_path)], text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(0, run.returncode, run.stderr)
        self.assertEqual(run.stdout, receipt_path.read_text(encoding="utf-8"))
        self.assertTrue(json.loads(run.stdout)["passed"])


class RuleIsolationTest(CandidatePolicyScanTest):
    """Every rule, and every alternative inside a rule, fires on its own.

    Finding (53) found a rule that had never worked -- the `-l` link-flag form
    of all five libraries -- because the suite asserted the rule set as a union
    over several fixture files, and a sibling fixture happened to match the
    same rule for a different reason. Union assertions cannot distinguish "this
    trigger works" from "some other line in some other file works".

    So: one trigger, one file, exact rule set. `.rsp` throughout, because that
    extension has no comment syntax and therefore no masking to reason about --
    the comment behaviour is covered separately in CommentAdvisoryTest.

    Alternatives inside a single regex get their own case. `composable_kernel`
    is one rule with five branches, and four of them passing tells you nothing
    about the fifth; that is the union problem again, one level down.
    """

    TRIGGERS = (
        ("rocblas", "rocblas_gemm_ex(handle, a, b);"),
        ("rocblas", "#include <rocblas/rocblas.h>"),
        # Deliberately not `dlsym(h, "rocblas_sgemm")`: after (54) that trips
        # dynamic_loader too, which is right, but then it is no longer an
        # isolated trigger for this rule and this class exists for isolation.
        ("rocblas", 'const char* n = "librocblas.so.4";'),
        ("hipblas", "hipblasGemmEx(h, x);"),
        ("hipblas", "#include <hipblas/hipblas.h>"),
        ("hipblaslt", "hipblasLtMatmul(h, d);"),
        ("hipblaslt", "#include <hipblaslt/hipblaslt.h>"),
        ("tensile", "TensileHost(problem);"),
        ("tensile", "libTensile.so.4"),
        ("miopen", "miopenConvolutionForward(h);"),
        ("miopen", "#include <miopen/miopen.h>"),
        ("composable_kernel", "using namespace ck;\nnamespace ck { }"),
        ("composable_kernel", "#include <ck/tensor_operation/gpu/device/device_gemm.hpp>"),
        ("composable_kernel", "#include <ck/utility/data_type.hpp>"),
        ("composable_kernel", "ck::tensor_operation::device::DeviceGemm g;"),
        ("composable_kernel", "ck::utility::foo();"),
        ("composable_kernel", "libck.so.1"),
        ("composable_kernel", "This uses the Composable Kernel library."),
        ("composable_kernel", "composable-kernel headers"),
        ("torch_matmul", "y = torch.matmul(a, b)"),
        ("torch_matmul", "y = torch.mm(a, b)"),
        ("torch_matmul", "y = torch.bmm(a, b)"),
        ("torch_matmul", "y = torch . matmul (a, b)"),
        ("torch_linear", "y = torch.nn.functional.linear(x, w)"),
        ("torch_linear", "y = F.linear(x, w)"),
        ("dynamic_loader", "import ctypes"),
        ("dynamic_loader", "h = dlopen(path, RTLD_NOW);"),
        ("dynamic_loader", "ctypes.cdll.LoadLibrary(p)"),
        ("dynamic_loader", "windll.kernel32"),
        ("dynamic_loader", "pydll.foo"),
        ("dynamic_loader", "load_library(p)"),
        # Added after (54). Each of these scanned clean before that finding.
        ("dynamic_loader", "h = dlmopen(LM_ID_NEW, p, RTLD_NOW);"),
        ("dynamic_loader", "f = dlsym(RTLD_DEFAULT, name);"),
        ("dynamic_loader", "f = dlvsym(h, name, ver);"),
        ("dynamic_loader", 'HMODULE m = LoadLibraryA("x.dll");'),
    )

    def test_the_two_rule_tables_agree_on_dynamic_loading(self):
        """A symbol the ELF table catches must also be caught in source.

        The tables are two views of one policy. `dlmopen` sat in SYMBOL_RULES
        and not in TEXT_RULES, so the identical call was a violation in a built
        artifact and clean in the source that produced it -- and which one the
        scan reaches first is a matter of when in the build it is run.
        """
        for name in ("dlopen", "dlmopen", "dlsym", "dlvsym"):
            with self.subTest(symbol=name):
                symbol_rule = next(r for r, rx in SCAN.SYMBOL_RULES if rx.search(name) and r != "inspection_error")
                self.assertEqual("dynamic_loader", symbol_rule)
                path = self.write("src.rsp", f"x = {name}(a, b);\n")
                receipt = self.scan(path)
                self.assertEqual({"dynamic_loader"}, {f["rule"] for f in receipt["findings"]},
                                 f"{name} is caught in an ELF but not in source")

    #: Rules that exist in source only, with the reason each cannot have a
    #: symbol form. Written out rather than inferred so that adding a rule to
    #: TEXT_RULES and forgetting SYMBOL_RULES is a failure, not a silent pass.
    SOURCE_ONLY = {
        "torch_matmul": "a Python attribute call; the ELF exports no such symbol",
        "torch_linear": "same -- `F.linear` is Python, not a linked entry point",
    }

    def test_the_two_rule_tables_cover_the_same_policy(self):
        """The dlmopen finding, generalised from one family to the table (57).

        The test above pins the four dynamic-loading symbols because those are
        the ones that bit. But the defect was not about `dlmopen`: it was that
        nothing compared the tables at all, so *any* rule could sit in one and
        not the other. Checking only the family that already failed leaves the
        next one exactly as unprotected as this one was.
        """
        text = {r for r, _ in SCAN.TEXT_RULES}
        symbol = {r for r, _ in SCAN.SYMBOL_RULES}
        self.assertEqual(set(), symbol - text,
                         "a rule catches a built artifact but not the source that produced it")
        self.assertEqual(self.SOURCE_ONLY.keys() | symbol, text,
                         "a source rule has no symbol counterpart and no stated reason "
                         "for lacking one")

    def test_every_rule_in_either_table_has_a_trigger_fixture(self):
        # A rule with no fixture is a rule nobody has ever seen fire, and
        # `test_each_trigger_fires_alone_and_alone_fires` -- the check that the
        # rules work at all -- would skip straight past it.
        covered = {rule for rule, _ in self.TRIGGERS}
        declared = {r for r, _ in SCAN.TEXT_RULES} | {r for r, _ in SCAN.SYMBOL_RULES}
        self.assertEqual(set(), declared - covered - {"inspection_error"})

    def test_the_real_candidate_elf_still_scans_clean(self):
        """Tightening a symbol rule must not start failing legitimate builds.

        This is the check (53) taught: the previous false positive survived for
        weeks because nobody re-ran the gate on the artifact it judges. Skipped
        rather than failed when the build is absent, since the tree is not
        guaranteed to carry one.
        """
        built = Path(__file__).resolve().parents[2] / (
            "exp/opt_bf16_20260814/ws_a/.torch_ext/"
            "geak_dense_bf16_gemm_fused_candidate/geak_dense_bf16_gemm_fused_candidate.so")
        if not built.exists():
            self.skipTest(f"no built candidate ELF at {built}")
        if shutil.which("readelf") is None or shutil.which("strings") is None:
            self.skipTest("readelf/strings unavailable")
        receipt = SCAN.scan([built])
        self.assertTrue(receipt["passed"], receipt["findings"])

    def test_each_trigger_fires_alone_and_alone_fires(self):
        for rule, body in self.TRIGGERS:
            with self.subTest(rule=rule, body=body):
                path = self.write("only.rsp", body + "\n")
                receipt = self.scan(path)
                self.assertFalse(receipt["passed"], f"{body!r} did not trip {rule}")
                # Exact, not superset: an over-broad rule that also fires here
                # is a future false positive, and this file is the cheapest
                # place to notice it.
                self.assertEqual({rule}, {f["rule"] for f in receipt["findings"]})

    def test_ordinary_prose_and_allowed_apis_trip_nothing(self):
        """The other half: rules broad enough to catch everything catch nothing.

        `ck` as two free letters and `check`/`black`/`stencil` are the standing
        near-misses; the HIP and MFMA lines are the APIs the policy explicitly
        allows, so a rule firing on them would block every legal candidate.
        """
        benign = (
            "check the stencil bounds; blacklist is empty; ck is not a namespace here",
            "#include <hip/hip_runtime.h>\nhipMalloc(&p, n);\nhipLaunchKernelGGL(k, g, b, 0, 0);",
            "__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);",
            "#include <rocwmma/rocwmma.hpp>\nrocwmma::load_matrix_sync(f, p, ld);",
            "torch.empty(n); torch.zeros_like(x); torch.cuda.synchronize()",
            "int linear_index = row * ld + col;  // not F.linear",
            "static_cast<int>(x); reinterpret_cast<float*>(p);",
        )
        for body in benign:
            with self.subTest(body=body[:40]):
                path = self.write("benign.rsp", body + "\n")
                receipt = self.scan(path)
                self.assertTrue(receipt["passed"], receipt["findings"])


class CommentAdvisoryTest(CandidatePolicyScanTest):
    """v2: a forbidden word inside a comment is advisory, in code it blocks.

    The behaviour these cover is the whole reason v2 exists -- v1 marked the
    shipped kernel `passed: false` forever on two words in explanatory prose,
    and the ledger had a standing note telling readers to ignore the result.
    The dangerous half is not the demotion, it is the scanner that decides
    which text is a comment: if it ever mis-reads code as commented, this gate
    stops working silently. Most of what follows tests that direction.
    """

    def advisory(self, receipt):
        return {item["rule"] for item in receipt["advisory"]}

    def test_word_in_a_comment_is_advisory_and_still_visible(self):
        self.write("k.hip", "// this kernel deliberately does not call rocBLAS\n"
                            "__global__ void k() {}\n")
        receipt = self.scan()
        self.assertTrue(receipt["passed"], receipt["findings"])
        self.assertEqual(set(), self.rules(receipt))
        self.assertEqual({"rocblas"}, self.advisory(receipt))
        self.assertEqual("text-comment", receipt["advisory"][0]["inspection"])

    def test_the_same_word_in_code_still_blocks(self):
        self.write("k.hip", "// we do not call rocBLAS\nvoid f() { rocblas_gemm_ex(); }\n")
        receipt = self.scan()
        self.assertFalse(receipt["passed"])
        self.assertEqual({"rocblas"}, self.rules(receipt))
        # Not reported twice under a softer heading -- that would invite
        # arguing the blocking one away.
        self.assertEqual(set(), self.advisory(receipt))
        self.assertIn("rocblas_gemm_ex", receipt["findings"][0]["evidence"])

    def test_block_comment_and_python_hash_comment(self):
        self.write("a.cpp", "/* Tensile is not linked here */\nint x;\n")
        self.write("b.py", "# this file must not call torch.matmul(a, b)\nx = 1\n")
        receipt = self.scan()
        self.assertTrue(receipt["passed"], receipt["findings"])
        self.assertEqual({"tensile", "torch_matmul"}, self.advisory(receipt))

    def test_double_slash_inside_a_string_does_not_open_a_comment(self):
        # The failure mode: treating the `//` in a URL as a comment start would
        # mask the rest of the line, hiding the real call that follows it.
        self.write("k.hip", 'const char* u = "https://x/y"; void f(){ rocblas_gemm_ex(); }\n')
        receipt = self.scan()
        self.assertFalse(receipt["passed"], receipt)
        self.assertEqual({"rocblas"}, self.rules(receipt))

    def test_library_name_inside_a_string_literal_blocks(self):
        # A string is code: `dlopen("librocblas.so")` is the exact thing the
        # policy exists to catch, and it is not prose.
        self.write("k.hip", 'const char* lib = "librocblas.so.4";\n')
        receipt = self.scan()
        self.assertFalse(receipt["passed"], receipt)
        self.assertEqual({"rocblas"}, self.rules(receipt))

    def test_python_docstring_is_code_not_comment(self):
        self.write("m.py", '"""helper\n\nctypes.CDLL(name)\n"""\n')
        receipt = self.scan()
        self.assertFalse(receipt["passed"], receipt)
        self.assertEqual({"dynamic_loader"}, self.rules(receipt))

    def test_unterminated_block_comment_fails_closed(self):
        self.write("k.hip", "/* rocBLAS mentioned and never closed\nvoid f(){}\n")
        receipt = self.scan()
        self.assertFalse(receipt["passed"], receipt)
        self.assertEqual({"rocblas"}, self.rules(receipt))
        record = next(r for r in receipt["inspected"] if r.get("format") == "text")
        self.assertEqual("none", record["comment_syntax"])

    def test_unterminated_string_literal_fails_closed(self):
        self.write("k.hip", 'const char* s = "oops;\n// rocBLAS in a comment below\n')
        receipt = self.scan()
        self.assertFalse(receipt["passed"], receipt)
        self.assertEqual({"rocblas"}, self.rules(receipt))

    def test_unknown_extension_gets_no_masking(self):
        # A linker response file has no comment syntax this scanner knows, and
        # `-lrocblas` after a `#` in one is still a link against rocBLAS.
        self.write("link.txt", "# flags\n-lrocblas\n")
        self.write("also.rsp", "# -lrocblas\n")
        receipt = self.scan()
        self.assertFalse(receipt["passed"], receipt)
        self.assertEqual({"rocblas"}, self.rules(receipt))
        self.assertEqual(set(), self.advisory(receipt))

    def test_advisory_never_changes_the_exit_code(self):
        # Six role prompts read `passed` and "any finding"; an advisory-only
        # receipt must not make any of them fail closed.
        candidate = self.write("k.hip", "// rocBLAS and Tensile are named here only in prose\n")
        run = subprocess.run([sys.executable, str(SCRIPT), str(candidate)], text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(0, run.returncode, run.stderr)
        receipt = json.loads(run.stdout)
        self.assertEqual([], receipt["findings"])
        self.assertEqual({"rocblas", "tensile"}, {f["rule"] for f in receipt["advisory"]})

    def test_link_flags_are_detected_on_their_own(self):
        """Each `-l` flag must fail its file with nothing else in it.

        `test_all_forbidden_text_families` appeared to cover link flags but
        could not: its `link.txt` was scanned alongside an `a.cpp` full of bare
        API calls, and the rule set is checked as a union over all files. Every
        `-l` flag went undetected from the scanner's first version until
        2026-08-15 for that reason -- the form that actually links a library
        was the one form not caught.
        """
        for flag, rule in (("-lrocblas", "rocblas"), ("-lhipblas", "hipblas"),
                           ("-lhipblaslt", "hipblaslt"), ("-lTensile", "tensile"),
                           ("-lMIOpen", "miopen")):
            with self.subTest(flag=flag):
                path = self.write("only.rsp", f"-O3 {flag} -lm\n")
                receipt = self.scan(path)
                self.assertFalse(receipt["passed"], receipt)
                self.assertEqual({rule}, self.rules(receipt))

    def test_link_flag_still_blocks_from_inside_a_makefile_comment(self):
        # `.cmake` is a hash-comment family, so this is the one place a link
        # flag could be demoted. It should be: a commented-out flag does not
        # link. The point of the test is that it lands in `advisory` and is
        # therefore still reported, not dropped.
        self.write("f.cmake", "# LIBS := -lrocblas\n")
        receipt = self.scan()
        self.assertTrue(receipt["passed"], receipt["findings"])
        self.assertEqual({"rocblas"}, self.advisory(receipt))

    def test_binary_surfaces_are_never_demoted(self):
        # `strings` output has no comment syntax; every match on a binary must
        # stay blocking regardless of what characters surround it.
        self.write("blob.bin", b"\x00\x01// librocblas.so.4\x00")
        receipt = self.scan()
        self.assertFalse(receipt["passed"], receipt)
        self.assertEqual({"rocblas"}, self.rules(receipt))
        self.assertEqual(set(), self.advisory(receipt))


class SummaryBlockTest(CandidatePolicyScanTest):
    """Finding (69): the `summary` block the orchestrator gates on.

    The lane refuses any report claiming `policy_pass: true` whose
    `policy_postbuild` summary is absent, self-contradictory, or shows no
    inspected ELF. That gate is only as good as the block being *derived* here
    rather than typed by an agent, so these tests pin the derivation to the
    receipt it summarises -- a summary that could disagree with its own findings
    list would be exactly the self-report the gate exists to replace.
    """

    def test_summary_agrees_with_the_receipt_it_summarises(self):
        self.write("ok.hip", "__global__ void k() {}\n")
        receipt = self.scan()
        s = receipt["summary"]
        self.assertEqual(s["passed"], receipt["passed"])
        self.assertEqual(s["findings"], len(receipt["findings"]))
        self.assertEqual(s["advisory"], len(receipt["advisory"]))
        self.assertEqual(s["inspected"], len(receipt["inspected"]))
        self.assertEqual(s["schema"], receipt["schema"])

    def test_a_failing_scan_summarises_as_failing_with_a_positive_count(self):
        self.write("bad.hip", '#include <rocblas/rocblas.h>\n')
        receipt = self.scan()
        s = receipt["summary"]
        self.assertFalse(s["passed"], receipt)
        self.assertGreater(s["findings"], 0)
        # The lane refuses `passed` beside nonzero `findings`; the two fields
        # must therefore never be independently settable.
        self.assertEqual(s["passed"], s["findings"] == 0)

    def test_a_source_only_scan_reports_zero_elf(self):
        # This is the field with teeth: `DT_NEEDED` and imported symbols exist
        # only in the binary, so a post-build receipt with elf == 0 is a
        # pre-build receipt and the lane says so.
        self.write("ok.hip", "__global__ void k() {}\n")
        self.assertEqual(0, self.scan()["summary"]["elf"])

    def test_summary_is_present_even_when_nothing_was_inspected(self):
        empty = Path(tempfile.mkdtemp(dir=self.temp.name))
        s = SCAN.scan([empty], ())["summary"]
        # This is the case that showed `inspected` cannot answer the question the
        # lane was asking it: the directory entry itself is inspected, so the
        # count reads 1 for a tree containing nothing. `files` reads 0, and that
        # is the field the orchestrator gates on.
        self.assertEqual(1, s["inspected"])
        self.assertEqual(0, s["files"])
        # `passed` is vacuously true here, which is precisely why the lane
        # additionally requires `files >= 1`.
        self.assertTrue(s["passed"])

    def test_files_counts_opened_files_and_not_directories(self):
        self.write("a/one.hip", "__global__ void k() {}\n")
        self.write("a/two.cmake", "# nothing\n")
        s = self.scan()["summary"]
        self.assertEqual(2, s["files"])
        self.assertGreater(s["inspected"], s["files"],
                           "the root and the subdirectory are inspected entries too")


if __name__ == "__main__":
    unittest.main(verbosity=2)
