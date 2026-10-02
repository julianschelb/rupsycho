"""Measure what ``import rupsycho`` costs in a fresh interpreter, current tree versus the old one.

Every measurement starts a new Python process (one discarded warm-up run first, so that bytecode
caches exist) and times two steps:

1. ``import rupsycho``. In the current tree this only registers the public names; they are
   imported on first access (PEP 562), so the command line starts instantly. The old tree
   imported everything here.
2. The first use of the public API (``rupsycho.ExperimentDocument``), which is where the current
   tree loads the experiment machinery (pydantic, pandas, LangChain, ...). The sum of both is the
   time until rupsycho is ready to use, the number to compare between the trees.

Besides the times (``--runs`` samples, median and min) the report lists which heavy modules
(``torch``, ``transformers``, provider SDKs, ...) ended up in ``sys.modules`` after each step.

The old tree (commit ``94f99d7`` by default) is extracted with ``git archive`` and put on
``PYTHONPATH``. It imported every back-end eagerly, so it cannot even be imported in an
environment without them; such a run is reported as a failure together with the error.

``langchain_core`` imports ``transformers`` (and thereby ``torch``) on its own whenever they are
installed, so the benefit of the optional extras only shows in an environment that has just the
core dependencies. Create one and pass it with ``--python``::

    uv venv --python 3.12 /tmp/minimal
    uv pip install --python /tmp/minimal/bin/python -e .

    python benchmarks/bench_import.py --python "full=$(which python)" --python "core only=/tmp/minimal/bin/python"
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from synthetic import (
    LEGACY_COMMIT,
    REPO_ROOT,
    extract_legacy_tree,
    format_table,
    git_revision,
    library_versions,
    machine_description,
)

HEAVY_MODULES = (
    "torch",
    "transformers",
    "langchain_huggingface",
    "langchain_openai",
    "langchain_ollama",
    "langchain_google_genai",
    "langchain_deepseek",
    "openai",
    "scipy",
    "IPython",
    "streamlit",
)
"""Modules whose presence in ``sys.modules`` is reported (none is needed to run a fake model)."""
MARKER = "BENCH-RESULT-JSON:"
CHILD = f"""
import json, sys, time
heavy = {list(HEAVY_MODULES)!r}
start = time.perf_counter()
import rupsycho
imported = time.perf_counter()
after_import = [name for name in heavy if name in sys.modules]
modules_after_import = len(sys.modules)
rupsycho.ExperimentDocument  # first use: lazy in the current tree, already loaded in the old one
ready = time.perf_counter()
print({MARKER!r} + json.dumps({{
    "import_s": imported - start,
    "first_use_s": ready - imported,
    "file": rupsycho.__file__,
    "loaded_after_import": after_import,
    "loaded_after_use": [name for name in heavy if name in sys.modules],
    "modules_after_import": modules_after_import,
    "modules_after_use": len(sys.modules),
}}))
"""


def run_child(
    python: str, src: Path, *, importtime: bool = False
) -> tuple[dict[str, Any] | None, float, str]:
    """Import ``rupsycho`` from ``src`` in a new process of ``python`` and use it once.

    Returns:
        ``(result, process_seconds, stderr)``; ``result`` is ``None`` if the import failed.
    """
    env = {**os.environ, "PYTHONPATH": str(src), "PYTHONWARNINGS": "ignore"}
    command = [python, *(["-X", "importtime"] if importtime else []), "-c", CHILD]
    start = time.perf_counter()
    completed = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    process_seconds = time.perf_counter() - start
    if completed.returncode != 0:
        return None, process_seconds, completed.stderr
    for line in completed.stdout.splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER) :]), process_seconds, completed.stderr
    return None, process_seconds, completed.stderr or completed.stdout


def last_error_line(stderr: str) -> str:
    """The final non-empty line of a traceback, e.g. ``ModuleNotFoundError: No module named 'x'``."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return lines[-1] if lines else "unknown error"


def top_packages(stderr: str, count: int = 5) -> list[tuple[str, float]]:
    """Top-level packages ranked by the time spent in their own import code (``-X importtime``)."""
    own: collections.Counter[str] = collections.Counter()
    for line in stderr.splitlines():
        if not line.startswith("import time:") or "self [us]" in line:
            continue
        parts = line.removeprefix("import time:").split("|")
        if len(parts) != 3:
            continue
        try:
            microseconds = int(parts[0])
        except ValueError:
            continue
        package = parts[2].strip().split(".")[0]
        own[package] += microseconds
    return [(name, total / 1e6) for name, total in own.most_common(count)]


def measure(
    label: str, tree: str, python: str, src: Path, runs: int, importtime: bool
) -> dict[str, Any]:
    """Measure one (environment, tree) combination."""
    record: dict[str, Any] = {"environment": label, "tree": tree, "src": str(src)}
    result, _, stderr = run_child(python, src)  # warm-up: creates the bytecode caches
    if result is None:
        record["error"] = last_error_line(stderr)
        return record
    imports, uses, process = [], [], []
    for _ in range(runs):
        result, process_seconds, stderr = run_child(python, src)
        if result is None:
            record["error"] = last_error_line(stderr)
            return record
        imports.append(result["import_s"])
        uses.append(result["first_use_s"])
        process.append(process_seconds)
    ready = [i + u for i, u in zip(imports, uses)]
    record.update(
        {
            "import_median_s": statistics.median(imports),
            "first_use_median_s": statistics.median(uses),
            "ready_median_s": statistics.median(ready),
            "ready_min_s": min(ready),
            "process_median_s": statistics.median(process),
            "modules_after_import": result["modules_after_import"],
            "modules_after_use": result["modules_after_use"],
            "loaded_after_import": result["loaded_after_import"],
            "loaded_after_use": result["loaded_after_use"],
            "file": result["file"],
        }
    )
    if importtime:
        _, _, stderr = run_child(python, src, importtime=True)
        record["top_packages"] = top_packages(stderr)
    return record


def parse_environment(text: str) -> tuple[str, str]:
    """Split ``LABEL=PATH`` (argparse type)."""
    label, separator, path = text.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError(f"expected LABEL=PYTHON_EXECUTABLE, got {text!r}")
    return label, path


def main(argv: Sequence[str] | None = None) -> int:
    """Print the import-time table."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("Create one and pass")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="To measure an environment that has only the core dependencies:\n"
        "  uv venv --python 3.12 /tmp/minimal\n"
        "  uv pip install --python /tmp/minimal/bin/python -e .\n"
        '  python benchmarks/bench_import.py --python "full=$(which python)" '
        '--python "core only=/tmp/minimal/bin/python"',
    )
    parser.add_argument(
        "--runs", type=int, default=7, help="measured runs per combination (default 7)"
    )
    parser.add_argument(
        "--python",
        type=parse_environment,
        action="append",
        metavar="LABEL=PATH",
        help="interpreter to test (repeatable; default: the current one)",
    )
    parser.add_argument("--legacy-src", type=Path, help="already extracted src/ of the old tree")
    parser.add_argument("--legacy-commit", default=LEGACY_COMMIT, help=f"default {LEGACY_COMMIT}")
    parser.add_argument("--no-legacy", action="store_true", help="only measure the current tree")
    parser.add_argument(
        "--importtime",
        action="store_true",
        help="also list the packages that take longest to import",
    )
    parser.add_argument("--json", metavar="FILE", help="also write the raw results as JSON")
    args = parser.parse_args(argv)
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")

    environments = args.python or [("current environment", sys.executable)]
    print("import rupsycho in a fresh interpreter, then first use of the public API")
    print(f"machine : {machine_description()}")
    print(f"libs    : {library_versions()}")
    print(f"current : {REPO_ROOT / 'src'}, git {git_revision()}")
    print(f"{args.runs} runs per row after one warm-up run\n")

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rupsycho-legacy-") as scratch:
        trees = [("current", REPO_ROOT / "src")]
        if not args.no_legacy:
            trees.append(
                ("legacy", args.legacy_src or extract_legacy_tree(scratch, args.legacy_commit))
            )
        for label, python in environments:
            for tree, src in trees:
                print(f"measuring {tree} in {label} ...", file=sys.stderr, flush=True)
                records.append(measure(label, tree, python, src, args.runs, args.importtime))

    rows = []
    for record in records:
        if "error" in record:
            rows.append([record["environment"], record["tree"], "fails", "-", "-", "-", "-", "-"])
            continue
        rows.append(
            [
                record["environment"],
                record["tree"],
                f"{record['import_median_s']:.3f}",
                f"{record['first_use_median_s']:.3f}",
                f"{record['ready_median_s']:.2f}",
                f"{record['ready_min_s']:.2f}",
                f"{record['process_median_s']:.2f}",
                f"{record['modules_after_import']} / {record['modules_after_use']}",
            ]
        )
    print(
        format_table(
            [
                "Environment",
                "Tree",
                "import (s)",
                "+ first use (s)",
                "ready, median (s)",
                "ready, min (s)",
                "whole process, median (s)",
                "Modules after import / use",
            ],
            rows,
            "llrrrrrr",
        )
    )
    print("\nHeavy modules in sys.modules (after the import statement -> after first use)")
    for record in records:
        if "error" in record:
            detail = record["error"]
        else:
            after_import = ", ".join(record["loaded_after_import"]) or "none"
            after_use = ", ".join(record["loaded_after_use"]) or "none"
            detail = after_use if after_import == after_use else f"{after_import} -> {after_use}"
        print(f"  {record['tree']:7s} in {record['environment']}: {detail}")
    for record in records:
        if record.get("top_packages"):
            top = ", ".join(f"{name} {seconds:.2f} s" for name, seconds in record["top_packages"])
            print(
                f"\nlargest self-import times, {record['tree']} in {record['environment']}: {top}"
            )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "machine": machine_description(),
                    "revision": git_revision(),
                    "runs": args.runs,
                    "records": records,
                },
                handle,
                indent=2,
            )
        print(f"\nraw results written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
