#!/usr/bin/env python3
"""Execute this package's node-style JS regression tests on an embedded V8.

Why this exists
---------------
`test_lane_gates.js` is the behavioural guard for the lane's admission gates:
the correctness/metric selector, the oracle pin, the final validation verdict,
and the policy / hipify-twin / ISA receipts. It has never run. Machine after
machine in this project has had no `node`, `nodejs`, `bun`, or `deno`, and the
ledger records the gap honestly as "the only genuinely runtime-blocked item"
across several findings.

Meanwhile finding (56) established the cost of that state directly: an
unreachable gate is also an unexercised one, and the first real invocation of
one such gate turned up six latent defects that no unit fixture would have
contained. `test_lane_gates.js` guards the code path that decides whether a
measured candidate is admitted at all, so leaving it unexecuted is leaving the
lane's trust boundary untested.

`mini-racer` embeds V8 as a Python wheel, no system `node` required. That is
enough to run these tests, because they are pure: they read one source file,
extract functions from it with regexes, and evaluate them in a sandbox. The
only node APIs they touch are `console`, `require('fs'|'path')`, `__dirname`,
and `process.exit`, all of which are shimmed below.

What this is NOT
----------------
Not a node substitute. There is no event loop, no real filesystem, no npm.
`fs.readFileSync` serves exactly the lane source and throws on anything else,
so a test that starts reading other files fails loudly here instead of
silently reading something unintended.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent.parent
LANE = ROOT / "kernel_workflow" / "kernel_lane.js"
WORKFLOW = ROOT / "kernel_workflow" / "kernel_workflow.js"
E2E = ROOT / "e2e_workflow" / "e2e_workflow.js"
E2E_SCRIPTS = ROOT / "e2e_workflow" / "scripts"
E2E_ROLES = ROOT / "e2e_workflow" / "roles"

# Tests this runner knows how to host, with the files each is allowed to read.
# Finding (65) added a second file: the lane produces a validation verdict and
# kernel_workflow.js consumes it, and a guard that can only read the producer
# cannot see the consumer drop it.
JS_TESTS = {
    "test_lane_gates.js": (LANE, WORKFLOW),
    # The CANDIDATE_FLOOR guard reads the lane only. It was runnable under node
    # and nowhere else, which on a box without node means it was not run at all
    # -- the same way test_lane_gates.js went unexecuted before this runner
    # existed.
    "test_candidate_floor.js": (LANE,),
    # The dispatcher guard. Its body is one big async IIFE, which is why it
    # could only be hosted once `run_body` learned to pump the microtask queue
    # and to refuse a suite that never finishes -- before that it "passed" after
    # printing a single section header.
    "test_mode_dispatch.js": (WORKFLOW,),
    # The two e2e guards. They live in a DIFFERENT scripts directory, which is
    # the only reason they were missed twice: a runner that resolves every test
    # against its own directory cannot even name them. Registering them is the
    # generalisation of the last two findings -- the exposure was never "this
    # file", it was "a guard nothing executes", and the way to stop rediscovering
    # it one file at a time is to make the inventory the thing under maintenance.
    "test_expert_skills_off_identical.js": (E2E, LANE),
    "test_analysis_skill_off_identical.js": (
        E2E, E2E_ROLES / "profiler.md", E2E_ROLES / "system_architect.md"),
}

# Where each suite's file lives, and what `__dirname` must resolve to inside it.
# Both suites below compute ROOT as `path.resolve(__dirname, '..', '..')`, so a
# wrong dirname sends every `readFileSync` to a path the shim refuses -- loudly,
# which is the point of finding (66).
SUITE_DIRS = {
    "test_expert_skills_off_identical.js": E2E_SCRIPTS,
    "test_analysis_skill_off_identical.js": E2E_SCRIPTS,
}


class JSRuntimeUnavailable(RuntimeError):
    """No embedded JS engine is installed."""


def _make_context():
    try:
        from py_mini_racer import MiniRacer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise JSRuntimeUnavailable(
            "no embedded JS engine: pip install mini-racer") from exc
    return MiniRacer()


def _prelude(sources: "dict[str, str]", dirname: Path | None = None) -> str:
    """A minimal node surface: console, require('fs'|'path'), __dirname, process.

    `readFileSync` serves an explicit allow-list and refuses everything else. A
    test that reaches for anything beyond the sources it is meant to inspect
    should fail here rather than be quietly handed an empty string -- silently
    reading nothing is how a regex guard turns into a test that passes because
    it matched nothing.

    Finding (66). That was the intent, and `path.join`/`path.resolve` defeated
    it: both returned the single allowed path for ANY arguments, so a test that
    joined its way to a DIFFERENT file was handed the allowed file's text and
    every regex against it silently described the wrong source. The shim failed
    in the direction that fabricates agreement, which is the one direction a
    test harness must never fail in. They now do real (posix, in-memory) path
    arithmetic, so a wrong path resolves to a wrong path and gets refused.
    """
    return f"""
function __norm(p) {{
  var s = String(p), abs = s.charAt(0) === '/', out = [], parts = s.split('/');
  for (var i = 0; i < parts.length; i++) {{
    var seg = parts[i];
    if (seg === '' || seg === '.') continue;
    if (seg === '..') {{ if (out.length) out.pop(); continue; }}
    out.push(seg);
  }}
  return (abs ? '/' : '') + out.join('/');
}}
var __out = [];
var __exit_code = null;
var console = {{
  log: function() {{ __out.push(Array.prototype.join.call(arguments, ' ')); }},
  error: function() {{ __out.push(Array.prototype.join.call(arguments, ' ')); }},
}};
var __SOURCES = {json.dumps(sources)};
var __dirname = {json.dumps(str(dirname or SCRIPTS_DIR))};
var __filename = '';
var module = {{ exports: {{}} }};
var exports = module.exports;
function require(name) {{
  if (name === 'fs') {{
    return {{
      readFileSync: function(p) {{
        var key = __norm(p);
        if (!Object.prototype.hasOwnProperty.call(__SOURCES, key)) {{
          throw new Error('this runner only serves [' + Object.keys(__SOURCES).join(', ') +
                          '], refusing to read ' + key);
        }}
        return __SOURCES[key];
      }},
    }};
  }}
  if (name === 'path') {{
    return {{
      // Real arithmetic, not a constant. See finding (66) in _prelude's docstring.
      join: function() {{ return __norm(Array.prototype.join.call(arguments, '/')); }},
      resolve: function() {{
        var j = Array.prototype.join.call(arguments, '/');
        return __norm(j.charAt(0) === '/' ? j : __dirname + '/' + j);
      }},
      // node's `path.sep`. Absent here, `path.sep + 'x'` silently becomes the
      // string 'undefinedx' and every `endsWith` against it is false -- a shim
      // gap that reads as a failing assertion about the source under test.
      sep: '/',
      // Used only to print a readable heading. Returns the absolute path when
      // `to` is not under `from` rather than climbing with '..': a shim that
      // guesses here would put a wrong-looking but plausible path in a log.
      relative: function(from, to) {{
        var f = __norm(from), t = __norm(to);
        return t.indexOf(f + '/') === 0 ? t.slice(f.length + 1) : t;
      }},
      dirname: function(p) {{
        var n = __norm(p), i = n.lastIndexOf('/');
        return i > 0 ? n.slice(0, i) : (i === 0 ? '/' : '.');
      }},
    }};
  }}
  throw new Error("this runner provides no module '" + name + "'");
}}
var process = {{
  argv: [],
  env: {{}},
  exit: function(code) {{ __exit_code = code | 0; throw {{ __is_exit: true }}; }},
}};
"""


EPILOGUE = """
JSON.stringify({ output: __out.join('\\n'), exit_code: __exit_code });
"""


def strip_shebang(body: str) -> str:
    """node strips `#!`; a bare V8 treats it as a syntax error.

    Blanked rather than dropped so reported line numbers still line up with the
    file on disk.
    """
    return "//" + body[2:] if body.startswith("#!") else body


# How many times to pump the microtask queue waiting for an async suite to
# finish. Each `eval` drains pending microtasks, so a suite whose stubs all
# resolve synchronously completes in a handful; the cap only bounds a genuine
# hang. 200 is far past anything these suites need.
PUMP_LIMIT = 200

# A suite that never reached `process.exit` did not finish. Distinct from 70
# (uncaught throw) because the cause is different and so is the fix.
INCOMPLETE = 71


def run_body(sources: "dict[str, str]", body: str,
             dirname: Path | None = None) -> tuple[int, str]:
    """Host one JS test body against an allow-list of readable sources.

    Shared by the runner and by `test_js_suite.py`'s mutation harness so the two
    cannot drift in how they decide a suite passed -- which matters more than it
    sounds, because the interesting half of that decision is below.

    A suite body may be one big `(async () => { ... })()`, and V8's `eval`
    returns at the first `await`: the body is suspended, no assertion has run,
    `__exit_code` is still null, and the old code read null as 0. That is a
    suite which passes by never executing -- finding (128)'s exact shape in the
    harness itself, and it is how `test_mode_dispatch.js` was found to have
    reported a clean run after printing one section header. So the queue is
    pumped until the suite calls `process.exit`, and a suite that never does is
    reported as INCOMPLETE rather than as a pass.
    """
    ctx = _make_context()
    # `process.exit` unwinds by throwing, so the whole body is wrapped and the
    # sentinel re-thrown only when it is a genuine error.
    ctx.eval(
        _prelude(sources, dirname)
        + "try {\n" + strip_shebang(body) + "\n} catch (e) {\n"
        "  if (!(e && e.__is_exit)) {\n"
        "    __out.push('UNCAUGHT: ' + (e && e.stack ? e.stack : e));\n"
        "    if (__exit_code === null) { __exit_code = 70; }\n"
        "  }\n"
        "}\n"
    )
    pumps = 0
    while ctx.eval("__exit_code") is None and pumps < PUMP_LIMIT:
        ctx.eval("0")  # a no-op eval drains the pending microtask queue
        pumps += 1
    result = json.loads(ctx.eval(EPILOGUE))
    code = result["exit_code"]
    output = result["output"]
    if code is None:
        output += (f"\nINCOMPLETE: the suite never called process.exit after "
                   f"{PUMP_LIMIT} microtask pumps -- it did not run to the end, "
                   f"so the absence of failures says nothing")
        return INCOMPLETE, output
    return int(code), output


def run_js_test(name: str) -> tuple[int, str]:
    """Run one JS test file. Returns (exit_code, captured console output)."""
    if name not in JS_TESTS:
        raise KeyError(f"unknown JS test {name!r}; known: {sorted(JS_TESTS)}")
    source_paths = JS_TESTS[name]
    sources = {str(p): p.read_text(encoding="utf-8") for p in source_paths}
    here = SUITE_DIRS.get(name, SCRIPTS_DIR)
    return run_body(sources, (here / name).read_text(encoding="utf-8"), here)


def main(argv: list[str]) -> int:
    names = argv[1:] or sorted(JS_TESTS)
    worst = 0
    for name in names:
        try:
            code, output = run_js_test(name)
        except JSRuntimeUnavailable as exc:
            print(f"SKIP {name}: {exc}", file=sys.stderr)
            return 77
        print(f"===== {name} =====")
        print(output)
        worst = max(worst, code)
    return worst


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
