"""Compare the rewritten run loop with the pre-restructure implementation on the same workload.

The old source tree (commit ``94f99d7`` by default) is extracted with ``git archive`` into a
temporary directory and imported by a *subprocess* with ``PYTHONPATH`` pointing at it; the
current tree runs in a second subprocess. Both subprocesses

* build the same synthetic experiment (``synthetic.py``) with the same deterministic fake model,
* run it sequentially (the old release has no concurrency),
* have their progress-bar output (stderr) discarded while timing, and
* report a fingerprint of all answers, which must be identical.

The old ``run()`` has no ``show_progress`` argument and always draws a progress bar, so the
current tree is timed twice: with its default progress bar (the like-for-like comparison) and
with ``show_progress=False``. Old and new subprocesses alternate for ``--rounds`` rounds so that
slow drifts of the machine hit both. An old experiment can only be run once, so every
repetition builds a fresh experiment; building is not timed.

Examples::

    python benchmarks/bench_legacy.py
    python benchmarks/bench_legacy.py --items 50 --personas 100 --seeds 3 --rounds 2 --repeat 3
    python benchmarks/bench_legacy.py --legacy-src /path/to/extracted/src
"""

from __future__ import annotations

import argparse
import contextlib
import gc
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
    RecordingCallback,
    answers_digest,
    answers_table,
    build_experiment,
    extract_legacy_tree,
    format_table,
    git_revision,
    grid_size,
    library_versions,
    machine_description,
    parse_grid,
)

DEFAULT_GRIDS = ("10x10x1", "20x10x3", "50x20x3")
FIXED_GRID = "1x1x1"
"""A single call: its time is the fixed cost of one ``run()``."""
RESULT_MARKER = "BENCH-RESULT-JSON:"


# ------------------- Worker: runs inside the subprocess -------------------


def worker(args: argparse.Namespace) -> None:
    """Time ``run()`` on the grids with whichever ``rupsycho`` is first on ``sys.path``."""
    import rupsycho

    expected = os.environ.get("BENCH_EXPECT_SRC")
    if expected and not Path(rupsycho.__file__).resolve().is_relative_to(Path(expected).resolve()):
        raise SystemExit(
            f"imported rupsycho from {rupsycho.__file__}, expected it below {expected}"
        )

    # The old run() takes no show_progress argument and always shows a bar; the new one can do both
    modes = (
        {"default": {}}
        if args.tree == "legacy"
        else {"default": {}, "quiet": {"show_progress": False}}
    )
    kind = args.model_kind
    results: dict[str, Any] = {}
    with open(os.devnull, "w") as sink, contextlib.redirect_stderr(sink):
        build_experiment(4, 2, 1, kind=kind).run(**modes["default"])  # warm-up

        gc_times = []
        for _ in range(3):
            start = time.perf_counter()
            gc.collect()
            gc_times.append(time.perf_counter() - start)

        for label in args.grids.split(","):
            grid = parse_grid(label)
            samples: dict[str, list[float]] = {mode: [] for mode in modes}
            digests = set()
            for repetition in range(args.repeat):
                # Alternate the order of the modes so that neither always runs first
                sequence = list(modes.items())[:: -1 if repetition % 2 else 1]
                for mode, kwargs in sequence:
                    experiment = build_experiment(*grid, kind=kind)
                    gc.collect()
                    start = time.perf_counter()
                    experiment.run(**kwargs)
                    samples[mode].append(time.perf_counter() - start)
                    digests.add(answers_digest(answers_table(experiment)))  # not timed

            # One untimed run that also records the order in which callbacks are triggered
            experiment, recorder = build_experiment(*grid, kind=kind), RecordingCallback()
            experiment.run(callbacks=[recorder], **modes["default"])
            results[label] = {
                "calls": grid_size(*grid),
                "samples": samples,
                "answers_digest": sorted(digests),
                "order_digest": answers_digest(recorder.rows),
            }

    payload = {
        "tree": args.tree,
        "rupsycho_file": rupsycho.__file__,
        "gc_collect_s": min(gc_times),
        "grids": results,
    }
    print(RESULT_MARKER + json.dumps(payload))


# ------------------- Parent: orchestrates the subprocesses -------------------


def run_worker(
    tree: str, src: Path, grids: Sequence[str], repeat: int, kind: str
) -> dict[str, Any]:
    """Run one worker subprocess against the source tree ``src`` and return its results."""
    env = {
        **os.environ,
        "PYTHONPATH": str(src),
        "PYTHONWARNINGS": "ignore",
        "BENCH_EXPECT_SRC": str(src),
    }
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--tree",
        tree,
        "--grids",
        ",".join(grids),
        "--repeat",
        str(repeat),
        "--model-kind",
        kind,
    ]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"the {tree} worker failed:\n{completed.stderr[-4000:]}")
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            return json.loads(line[len(RESULT_MARKER) :])  # type: ignore[no-any-return]
    raise SystemExit(f"the {tree} worker printed no result:\n{completed.stdout[-2000:]}")


def merge(rounds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Merge the results of several rounds of the same tree (samples are concatenated)."""
    merged: dict[str, Any] = {
        "rupsycho_file": rounds[0]["rupsycho_file"],
        "gc_collect_s": min(r["gc_collect_s"] for r in rounds),
        "grids": {},
    }
    for label, first in rounds[0]["grids"].items():
        cell: dict[str, Any] = {
            "calls": first["calls"],
            "samples": {},
            "digests": set(),
            "orders": set(),
        }
        for result in rounds:
            entry = result["grids"][label]
            for mode, values in entry["samples"].items():
                cell["samples"].setdefault(mode, []).extend(values)
            cell["digests"].update(entry["answers_digest"])
            cell["orders"].add(entry["order_digest"])
        merged["grids"][label] = cell
    return merged


def summary_of(samples: Sequence[float]) -> tuple[float, float]:
    """``(median, min)`` of the samples."""
    return statistics.median(samples), min(samples)


def compare(
    legacy: dict[str, Any], current: dict[str, Any], grids: Sequence[str]
) -> dict[str, Any]:
    """Print the comparison tables and return the numbers; the key ``identical`` is the verdict."""
    best_rows, median_rows, records, identical = [], [], {}, True
    for label in grids:
        old, new = legacy["grids"][label], current["grids"][label]
        calls = old["calls"]
        old_median, old_min = summary_of(old["samples"]["default"])
        new_median, new_min = summary_of(new["samples"]["default"])
        quiet_median, quiet_min = summary_of(new["samples"]["quiet"])
        same = (
            len(old["digests"] | new["digests"]) == 1
            and len(old["orders"]) == 1
            and old["orders"] == new["orders"]
        )
        identical &= same
        best_rows.append(
            [
                label,
                f"{calls:,}",
                f"{old_min:.3f}",
                f"{new_min:.3f}",
                f"{old_min / new_min:.2f}x",
                f"{quiet_min:.3f}",
                f"{old_min / quiet_min:.2f}x",
                "yes" if same else "NO",
            ]
        )
        median_rows.append(
            [
                label,
                f"{calls:,}",
                f"{old_median:.3f}",
                f"{new_median:.3f}",
                f"{old_median / new_median:.2f}x",
                f"{quiet_median:.3f}",
                f"{old_median / quiet_median:.2f}x",
            ]
        )
        records[label] = {
            "calls": calls,
            "legacy": {"median": old_median, "min": old_min},
            "current": {"median": new_median, "min": new_min},
            "current_no_bar": {"median": quiet_median, "min": quiet_min},
            "identical": same,
        }

    print("Best time of run() (seconds)\n")
    print(
        format_table(
            [
                "Grid",
                "Calls",
                "Legacy",
                "Current",
                "Speed-up",
                "Current, no bar",
                "Speed-up",
                "Same answers and callback order",
            ],
            best_rows,
        )
    )
    print("\nMedian time of run() (seconds)\n")
    print(
        format_table(
            ["Grid", "Calls", "Legacy", "Current", "Speed-up", "Current, no bar", "Speed-up"],
            median_rows,
        )
    )

    fixed_old = min(legacy["grids"][FIXED_GRID]["samples"]["default"])
    fixed_new = min(current["grids"][FIXED_GRID]["samples"]["default"])
    fixed_quiet = min(current["grids"][FIXED_GRID]["samples"]["quiet"])
    largest = max(grids, key=lambda label: legacy["grids"][label]["calls"])
    calls = legacy["grids"][largest]["calls"]
    per_old = (records[largest]["legacy"]["min"] - fixed_old) / calls * 1e6
    per_new = (records[largest]["current"]["min"] - fixed_new) / calls * 1e6
    per_quiet = (records[largest]["current_no_bar"]["min"] - fixed_quiet) / calls * 1e6
    print(
        f"\nFixed and per-call cost (best times; per call = (time of {largest} minus the fixed "
        f"cost) / {calls:,} calls)\n"
    )
    print(
        format_table(
            ["", "Legacy", "Current", "Current, no bar"],
            [
                [
                    "Fixed cost of one run() (1 call), ms",
                    f"{fixed_old * 1000:.1f}",
                    f"{fixed_new * 1000:.1f}",
                    f"{fixed_quiet * 1000:.1f}",
                ],
                [
                    "gc.collect() in the worker process, ms",
                    f"{legacy['gc_collect_s'] * 1000:.1f}",
                    f"{current['gc_collect_s'] * 1000:.1f}",
                    f"{current['gc_collect_s'] * 1000:.1f}",
                ],
                [
                    "Cost per call, us",
                    f"{per_old:,.1f}",
                    f"{per_new:,.1f}",
                    f"{per_quiet:,.1f}",
                ],
            ],
        )
    )
    return {
        "grids": records,
        "fixed_s": {"legacy": fixed_old, "current": fixed_new, "current_no_bar": fixed_quiet},
        "gc_collect_s": {"legacy": legacy["gc_collect_s"], "current": current["gc_collect_s"]},
        "us_per_call": {"legacy": per_old, "current": per_new, "current_no_bar": per_quiet},
        "identical": identical,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the comparison; the exit code is 1 if the two trees produced different answers."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("Examples::")[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--items", type=int, help="items; any of --items/--personas/--seeds selects one custom grid"
    )
    parser.add_argument("--personas", type=int, help="personas (default 10 for a custom grid)")
    parser.add_argument("--seeds", type=int, help="seeds (default 1 for a custom grid)")
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="timed runs per grid inside one subprocess (default 5)",
    )
    parser.add_argument(
        "--rounds", type=int, default=2, help="alternating old/new subprocess rounds (default 2)"
    )
    parser.add_argument(
        "--model-kind",
        choices=("llm", "chat"),
        default="chat",
        help="fake model type (default chat)",
    )
    parser.add_argument(
        "--legacy-src",
        type=Path,
        help="already extracted src/ of the old tree (default: git archive)",
    )
    parser.add_argument(
        "--legacy-commit",
        default=LEGACY_COMMIT,
        help=f"commit to extract (default {LEGACY_COMMIT})",
    )
    parser.add_argument("--json", metavar="FILE", help="also write the raw results as JSON")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tree", choices=("legacy", "current"), help=argparse.SUPPRESS)
    parser.add_argument("--grids", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        worker(args)
        return 0
    if args.repeat < 1 or args.rounds < 1:
        raise SystemExit("--repeat and --rounds must be at least 1")

    if any(value is not None for value in (args.items, args.personas, args.seeds)):
        grids = [f"{args.items or 20}x{args.personas or 10}x{args.seeds or 1}"]
    else:
        grids = list(DEFAULT_GRIDS)
    worker_grids = [FIXED_GRID, *grids]

    with tempfile.TemporaryDirectory(prefix="rupsycho-legacy-") as scratch:
        legacy_src = args.legacy_src or extract_legacy_tree(scratch, args.legacy_commit)
        current_src = REPO_ROOT / "src"
        print(
            f"Legacy ({args.legacy_commit}) vs current run loop, sequential, fake {args.model_kind} model"
        )
        print(f"machine : {machine_description()}")
        print(f"libs    : {library_versions()}")
        print(f"legacy  : {legacy_src}")
        print(f"current : {current_src}, git {git_revision()}")
        print(
            f"{args.rounds} alternating rounds x {args.repeat} runs = "
            f"{args.rounds * args.repeat} samples per cell; stderr (progress bars) discarded\n"
        )
        legacy_rounds, current_rounds = [], []
        for round_index in range(args.rounds):
            print(f"round {round_index + 1}/{args.rounds} ...", file=sys.stderr, flush=True)
            trees = [("legacy", legacy_src), ("current", current_src)]
            for tree, src in trees[:: -1 if round_index % 2 else 1]:  # alternate who goes first
                result = run_worker(tree, src, worker_grids, args.repeat, args.model_kind)
                (legacy_rounds if tree == "legacy" else current_rounds).append(result)

    legacy, current = merge(legacy_rounds), merge(current_rounds)
    result = compare(legacy, current, grids)
    if not result["identical"]:
        print("ERROR: the two trees produced different answers or callback order", file=sys.stderr)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "machine": machine_description(),
                    "revision": git_revision(),
                    "args": {k: str(v) for k, v in vars(args).items()},
                    "result": result,
                },
                handle,
                indent=2,
            )
        print(f"\nraw results written to {args.json}")
    return 0 if result["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
