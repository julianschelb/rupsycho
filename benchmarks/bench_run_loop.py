"""Benchmark the run loop of ``ExperimentDocument.run`` with scripted fake models.

No network and no model downloads are needed: the model is a deterministic fake (see
``synthetic.py``) that optionally sleeps to imitate an API. Four modes (``--mode``):

``overhead`` (default)
    What does the framework cost per model call? A zero-latency model is asked every
    (item, persona, seed) of several grid sizes with ``max_concurrency=1``. Besides the total time
    the report splits the cost into the fixed part of one ``run()`` and the part per call, and
    into what LangChain's ``chain.invoke`` costs on its own and what the loop adds on top.

``latency``
    What does ``max_concurrency`` buy for API models? Every call sleeps ``--latency-ms``; the
    same grid is run with each ``--max-concurrency`` value. Reports the measured speed-up
    against the ideal one and verifies that answers and callback order are identical.

``batch``
    Why does the run loop not use ``chain.batch``? Measures the speed of ``chain.batch`` and the
    thread pool for text-completion (LLM-type) and chat models, and how many answers are lost
    when one prompt fails.

``seeding``
    What does it cost per call to hand the seed to the model via ``model.bind(seed=...)`` (the
    pre-restructure release) compared with a copy of the model that has the seed set
    (``seed_model``, current)? Also times building the prompt variables of one call.

Examples::

    python benchmarks/bench_run_loop.py
    python benchmarks/bench_run_loop.py --items 50 --personas 100 --seeds 3 --repeat 5 --profile
    python benchmarks/bench_run_loop.py --mode latency --latency-ms 20 --max-concurrency 1,2,4,8,16
    python benchmarks/bench_run_loop.py --mode batch
    python benchmarks/bench_run_loop.py --mode seeding
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import json
import pstats
import random
import statistics
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any
from unittest import mock

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from synthetic import (
    ModelKind,
    RecordingCallback,
    answers_digest,
    answers_table,
    build_experiment,
    format_table,
    git_revision,
    grid_size,
    library_versions,
    machine_description,
    parse_grid,
)

import rupsycho
from rupsycho.seeding import seed_model

MODEL_ID = "fake"
OVERHEAD_GRIDS = ("4x2x1", "10x10x1", "20x10x3", "50x20x3", "50x100x3")
"""Default grids of the overhead mode: 8 up to 15 000 calls."""
LATENCY_GRID = "16x8x2"
"""Default grid of the latency mode: 256 calls, 8 personas to run side by side per item."""
BATCH_GRID = "8x4x1"
"""Default grid of the batch mode: 32 prompts."""
BURST_GRID = "20x50x1"
"""Grid of the short bursts that estimate the per-call cost (1 000 calls)."""
LOOP_GRID = "50x100x3"
"""Grid for the loop-only measurement: with an instant chain 15 000 calls take a fraction of a second."""
BURST_MAX_CALLS = 2000
"""A custom grid is used for the bursts only if it is not larger than this."""
SEEDING_GRID = "50x20x1"
"""Default grid of the seeding mode: 1 000 calls per sample."""
FAILING_PROMPT = r"Person2 .*\(item 3\)"
"""Pattern that makes exactly one prompt of a synthetic experiment fail (item 3, persona 2)."""


# ------------------- Measuring -------------------


@dataclass
class RunResult:
    """One timed ``run()``: seconds, the experiment (for its answers) and the recorded callbacks."""

    seconds: float
    experiment: Any
    recorder: RecordingCallback | None


class _InstantChain:
    """Stand-in for the LangChain chain that answers immediately (isolates the loop's own cost)."""

    def invoke(self, inputs: dict[str, Any]) -> str:
        return '{answer: "3"}'


@contextmanager
def stubbed_chain() -> Iterator[None]:
    """Replace the chain of the run loop by an instant stand-in while the context is active.

    This measures what the run loop itself costs per call, without LangChain. It relies on the
    private ``_get_chain`` method of the processing mixin.
    """
    from rupsycho.mixins.experiment_processing import ExperimentProcessingMixin

    with mock.patch.object(
        ExperimentProcessingMixin, "_get_chain", lambda self, *args, **kwargs: _InstantChain()
    ):
        yield


def loop_only_available() -> bool:
    """Whether the run loop still has the ``_get_chain`` hook that ``stubbed_chain`` patches."""
    from rupsycho.mixins.experiment_processing import ExperimentProcessingMixin

    return hasattr(ExperimentProcessingMixin, "_get_chain")


def timed_run(
    grid: tuple[int, int, int],
    *,
    kind: ModelKind = "chat",
    latency: float = 0.0,
    max_concurrency: int = 1,
    cumulative: bool = False,
    record: bool = False,
    stub: bool = False,
    fail_on: str | None = None,
) -> RunResult:
    """Build a fresh synthetic experiment and time one ``run()`` (building is not timed)."""
    experiment = build_experiment(
        *grid, kind=kind, latency=latency, fail_on=fail_on, model_id=MODEL_ID
    )
    recorder = RecordingCallback() if record else None
    gc.collect()
    with stubbed_chain() if stub else nullcontext():
        start = time.perf_counter()
        experiment.run(
            callbacks=[recorder] if recorder else [],
            cumulative=cumulative,
            max_concurrency=max_concurrency,
            on_error="ignore",
            show_progress=False,
        )
        seconds = time.perf_counter() - start
    return RunResult(seconds, experiment, recorder)


def chain_inputs(experiment: Any) -> list[dict[str, str]]:
    """Prompt variables of every (item, persona) pair, built from the public data models."""
    questionnaire = experiment.questionnaire
    options = questionnaire.default_answer_options.join_options()
    return [
        {
            "general_instruction": questionnaire.general_instruction,
            "persona_description": persona.get_profile_desc(),
            "question": item.question,
            "answer_options": options,
        }
        for item in questionnaire.instruction_items
        for persona in experiment.demographic_profiles.values()
    ]


def make_chain(experiment: Any, seed: int | None = None) -> Any:
    """The chain the run loop builds: ``passthrough | prompt | (seeded) model | parser``."""
    model = experiment.runnable_models[MODEL_ID]
    if seed is not None:
        model = seed_model(model, seed)
    chain = RunnablePassthrough() | experiment.runnable_prompt | model | StrOutputParser()
    return chain.with_config(run_name=MODEL_ID)


def timed_bare_chain(grid: tuple[int, int, int], *, kind: ModelKind = "chat") -> float:
    """Seconds for a plain loop of ``chain.invoke`` over the grid: LangChain without rupsycho."""
    experiment = build_experiment(*grid, kind=kind, model_id=MODEL_ID)
    inputs = chain_inputs(experiment)
    chains = [make_chain(experiment, int(seed)) for seed in experiment.parameters.seeds]
    gc.collect()
    start = time.perf_counter()
    for chain in chains:
        for values in inputs:
            chain.invoke(values)
    return time.perf_counter() - start


def gc_seconds() -> float:
    """Best of three ``gc.collect()`` calls in the current process."""
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        gc.collect()
        best = min(best, time.perf_counter() - start)
    return best


def stats_of(times: Sequence[float]) -> dict[str, float]:
    """Median, minimum and maximum of the samples."""
    return {"median": statistics.median(times), "min": min(times), "max": max(times)}


def grid_label(grid: tuple[int, int, int]) -> str:
    return "x".join(str(part) for part in grid)


def sleep_seconds(latency: float, samples: int = 10) -> float:
    """Median real duration of ``time.sleep(latency)``: sleeping overshoots, especially on macOS."""
    durations = []
    for _ in range(samples):
        start = time.perf_counter()
        time.sleep(latency)
        durations.append(time.perf_counter() - start)
    return statistics.median(durations)


def header(title: str, args: argparse.Namespace, latency: float | None = None) -> None:
    """Print what is measured and on which machine."""
    print(title)
    print(f"machine : {machine_description()}")
    print(f"libs    : {library_versions()}")
    print(f"rupsycho: {rupsycho.__version__} ({rupsycho.__file__}), git {git_revision()}")
    print(f"fake {args.model_kind} model, repeat={args.repeat} (median / min), no progress bar")
    if latency:
        print(
            f"time.sleep({latency * 1000:g} ms) really takes "
            f"{sleep_seconds(latency) * 1000:.1f} ms on this machine (median of 10)"
        )
    print()


# ------------------- Mode: overhead -------------------


def mode_overhead(args: argparse.Namespace) -> dict[str, Any]:
    """Framework overhead per call with a zero-latency model, for several grid sizes."""
    grids = [custom_grid(args)] if custom_grid(args) else [parse_grid(g) for g in OVERHEAD_GRIDS]
    kind: ModelKind = args.model_kind
    use_stub = loop_only_available()
    header(
        f"Run-loop overhead{' (cumulative)' if args.cumulative else ''}, max_concurrency=1", args
    )

    timed_run((4, 2, 1), kind=kind)  # warm-up: lazy imports and caches of LangChain
    fixed = min(
        timed_run((1, 1, 1), kind=kind, stub=use_stub).seconds for _ in range(max(args.repeat, 5))
    )
    print(
        f"fixed cost of one run() with a ready model (1 call, instant chain): {fixed * 1000:.1f} ms\n"
        f"for comparison, gc.collect() takes {gc_seconds() * 1000:.1f} ms in this process "
        "(paid once after a lazily loaded model)\n"
    )

    rows, records = [], []
    for grid in grids:
        calls = grid_size(*grid)
        total = stats_of(
            [
                timed_run(grid, kind=kind, cumulative=args.cumulative).seconds
                for _ in range(args.repeat)
            ]
        )
        rows.append(
            [
                grid_label(grid),
                f"{calls:,}",
                f"{total['median']:.3f}",
                f"{total['min']:.3f}",
                f"{calls / total['min']:,.0f}",
                f"{total['min'] / calls * 1e6:,.0f}",
            ]
        )
        records.append({"grid": grid_label(grid), "calls": calls, "run": total})
    print(
        format_table(
            [
                "Grid (items x personas x seeds)",
                "Calls",
                "run() median (s)",
                "run() min (s)",
                "Calls/s (best)",
                "us/call (best)",
            ],
            rows,
        )
    )
    print("\nus/call = best time / calls, so it contains the fixed cost of the run.")

    # Per-call cost from short bursts: a burst is far more likely to run undisturbed than a long
    # run, so the best of many interleaved bursts is a robust estimate
    burst = custom_grid(args)
    if burst is None or grid_size(*burst) > BURST_MAX_CALLS:
        burst = parse_grid(BURST_GRID)
    calls = grid_size(*burst)
    samples = max(args.repeat * 3, 15)
    runs: list[float] = []
    chains: list[float] = []
    for _ in range(samples):  # interleaved, so that drifts of the machine hit both alike
        runs.append(timed_run(burst, kind=kind, cumulative=args.cumulative).seconds)
        if not args.cumulative:
            chains.append(timed_bare_chain(burst, kind=kind))
    # With an instant chain a long run is cheap, and its many calls make the small per-call cost
    # of the loop itself measurable
    loop_grid = parse_grid(LOOP_GRID)
    loop_calls = grid_size(*loop_grid)
    loops = (
        [
            timed_run(loop_grid, kind=kind, cumulative=args.cumulative, stub=True).seconds
            for _ in range(samples)
        ]
        if use_stub
        else []
    )
    per_call = {
        "run": (min(runs) - fixed) / calls * 1e6,
        "chain": min(chains) / calls * 1e6 if chains else None,
        "loop": (min(loops) - fixed) / loop_calls * 1e6 if loops else None,
    }
    print(
        f"\nPer-call cost: best of {samples} interleaved bursts of {grid_label(burst)} = "
        f"{calls:,} calls for run() and the plain chain loop, best of {samples} runs of "
        f"{grid_label(loop_grid)} = {loop_calls:,} calls for the instant chain "
        "(run() and instant-chain runs minus the fixed cost)\n"
    )
    labels = {
        "run": "run()",
        "chain": "chain.invoke in a plain loop (LangChain alone, no rupsycho)",
        "loop": "run() with an instant chain (rupsycho's own loop)",
    }
    print(
        format_table(
            ["Measurement", "us per call"],
            [
                [labels[key], f"{value:,.1f}" if value is not None else "n/a"]
                for key, value in per_call.items()
            ],
        )
    )
    if args.profile:
        profile_run(max(grids, key=lambda g: grid_size(*g)), kind=kind, cumulative=args.cumulative)
    return {
        "fixed_seconds": fixed,
        "grids": records,
        "burst": {
            "grid": grid_label(burst),
            "samples": samples,
            "run": stats_of(runs),
            "chain": stats_of(chains) if chains else None,
            "loop_grid": grid_label(loop_grid),
            "loop": stats_of(loops) if loops else None,
        },
        "us_per_call": per_call,
    }


def custom_grid(args: argparse.Namespace) -> tuple[int, int, int] | None:
    """The grid given by ``--items/--personas/--seeds``, or ``None`` if none was given."""
    if args.items is None and args.personas is None and args.seeds is None:
        return None
    return (args.items or 20, args.personas or 10, args.seeds or 1)


def profile_run(
    grid: tuple[int, int, int], *, kind: ModelKind, cumulative: bool, top: int = 25
) -> None:
    """Profile one ``run()`` with cProfile and print the heaviest functions plus a share table."""
    experiment = build_experiment(*grid, kind=kind, model_id=MODEL_ID)
    gc.collect()
    profiler = cProfile.Profile()
    profiler.enable()
    experiment.run(cumulative=cumulative, show_progress=False)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats("cumulative")
    print(f"\n--- cProfile of one run() on {grid_label(grid)} (top {top} by cumulative time) ---")
    stats.print_stats(top)

    def cumulative_of(function: str, filename: str) -> float:
        return sum(
            entry[3]
            for (path, _, name), entry in stats.stats.items()  # type: ignore[attr-defined]
            if name == function and path == filename
        )

    total = cumulative_of("run", "experiment_processing.py")
    shares = [
        ("chain.invoke per call (_invoke)", cumulative_of("_invoke", "experiment_processing.py")),
        (
            "record + callbacks + progress (_handle_result)",
            cumulative_of("_handle_result", "experiment_processing.py"),
        ),
        (
            "memory cleanup (_cleanup_memory, gc.collect)",
            cumulative_of("_cleanup_memory", "experiment_processing.py"),
        ),
    ]
    if total:
        print("Share of run() (profiling inflates the call-heavy parts):")
        for label, seconds in shares:
            print(f"  {seconds / total:6.1%}  {label}")


# ------------------- Mode: latency -------------------


def mode_latency(args: argparse.Namespace) -> dict[str, Any]:
    """Speed-up of the thread pool for a model with fixed latency, and result identity."""
    grid = custom_grid(args) or parse_grid(LATENCY_GRID)
    latency = (args.latency_ms if args.latency_ms is not None else 20.0) / 1000
    workers = sorted(set(args.max_concurrency or [1, 2, 4, 8, 16]) | {1})
    calls = grid_size(*grid)
    # Calls that can be in flight at once: a whole seed pass, or one item (all personas) in
    # cumulative mode, where each persona has to wait for its own previous answer
    parallel_calls = grid[1] if args.cumulative else grid[0] * grid[1]
    header(
        f"Thread-pool scaling{' (cumulative)' if args.cumulative else ''}: {grid_label(grid)} = "
        f"{calls} calls, {latency * 1000:g} ms per call",
        args,
        latency,
    )

    timed_run((4, 2, 1), kind=args.model_kind)  # warm-up
    reference: tuple[str, str] | None = None
    records, rows, all_identical = [], [], True
    baseline = 0.0
    for k in workers:
        times, identical = [], True
        for _ in range(args.repeat):
            result = timed_run(
                grid,
                kind=args.model_kind,
                latency=latency,
                max_concurrency=k,
                cumulative=args.cumulative,
                record=True,
            )
            times.append(result.seconds)
            assert result.recorder is not None
            fingerprint = (
                answers_digest(answers_table(result.experiment)),
                answers_digest(result.recorder.rows),
            )
            reference = reference or fingerprint
            identical &= fingerprint == reference
        all_identical &= identical
        stat = stats_of(times)
        baseline = baseline or stat["median"]
        speedup = baseline / stat["median"]
        ideal = min(k, parallel_calls)
        rows.append(
            [
                k,
                f"{stat['median']:.3f}",
                f"{stat['min']:.3f}",
                f"{calls / stat['median']:,.0f}",
                f"{speedup:.2f}x",
                f"{ideal}x",
                f"{speedup / ideal:.0%}",
                "yes" if identical else "NO",
            ]
        )
        records.append(
            {"max_concurrency": k, "time": stat, "speedup": speedup, "identical": identical}
        )

    print(
        format_table(
            [
                "max_concurrency",
                "run() median (s)",
                "min (s)",
                "Calls/s",
                "Speed-up",
                "Ideal",
                "Efficiency",
                "Same answers and callback order",
            ],
            rows,
        )
    )
    print(
        "\nSpeed-up = median time of max_concurrency=1 divided by the median time of the row; "
        "ideal = number of workers (at most the calls that can run side by side).\n"
        "Every run() also pays its fixed cost (see overhead mode)."
    )
    if not all_identical:
        print("ERROR: results differ between concurrency settings", file=sys.stderr)
    return {
        "grid": grid_label(grid),
        "latency_s": latency,
        "runs": records,
        "identical": all_identical,
    }


# ------------------- Mode: batch -------------------


def _median_time(function: Callable[[], Any], repeat: int) -> float:
    """Median wall time of ``function`` over ``repeat`` calls (garbage collected beforehand)."""
    samples = []
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _batch_speed(
    kind: ModelKind, grid: tuple[int, int, int], latency: float, k: int, repeat: int
) -> dict[str, float]:
    """Time the same prompts with a sequential loop, ``chain.batch`` and the run loop."""
    experiment = build_experiment(*grid, kind=kind, latency=latency, model_id=MODEL_ID)
    inputs, chain = chain_inputs(experiment), make_chain(experiment)
    return {
        "sequential chain.invoke loop": _median_time(
            lambda: [chain.invoke(values) for values in inputs], repeat
        ),
        f"chain.batch(max_concurrency={k})": _median_time(
            lambda: chain.batch(inputs, config={"max_concurrency": k}), repeat
        ),
        "chain.batch() without a limit": _median_time(lambda: chain.batch(inputs), repeat),
        # timed_run times only run() itself, not the construction of the experiment
        f"rupsycho run(max_concurrency={k})": statistics.median(
            timed_run(grid, kind=kind, latency=latency, max_concurrency=k).seconds
            for _ in range(repeat)
        ),
    }


def _batch_lost(kind: ModelKind, grid: tuple[int, int, int], k: int) -> dict[str, int]:
    """Count the answers lost when exactly one prompt (item 3, persona 2) fails."""
    experiment = build_experiment(*grid, kind=kind, fail_on=FAILING_PROMPT, model_id=MODEL_ID)
    inputs, chain = chain_inputs(experiment), make_chain(experiment)

    def fails(values: dict[str, str]) -> bool:
        try:
            chain.invoke(values)
        except RuntimeError:
            return True
        return False

    lost = {"prompts that really fail": sum(fails(values) for values in inputs)}
    for label, config in (
        (f"chain.batch(max_concurrency={k}, return_exceptions=True)", {"max_concurrency": k}),
        ("chain.batch(return_exceptions=True), no limit", None),
    ):
        outputs = chain.batch(inputs, config=config, return_exceptions=True)
        lost[label] = sum(isinstance(output, Exception) for output in outputs)
    summary = experiment.run(max_concurrency=k, on_error="ignore", show_progress=False)
    lost[f"rupsycho run(max_concurrency={k})"] = summary.n_failed
    return lost


class _ArrivalTimes:
    """Callback that records when each answer reaches the callbacks."""

    def __init__(self) -> None:
        self.times: list[float] = []

    def save_answer(self, *args: Any) -> None:
        self.times.append(time.perf_counter())


def mode_batch(args: argparse.Namespace) -> dict[str, Any]:
    """Compare ``chain.batch`` with the thread pool of the run loop."""
    grid = custom_grid(args) or parse_grid(BATCH_GRID)
    latency = (args.latency_ms if args.latency_ms is not None else 20.0) / 1000
    k = (args.max_concurrency or [8])[0]
    prompts = grid[0] * grid[1]
    header(
        f"chain.batch vs the run loop: {grid_label(grid)}, {prompts} prompts per seed, "
        f"{latency * 1000:g} ms per call, {k} workers",
        args,
        latency,
    )
    kinds: list[ModelKind] = ["llm", "chat"]
    timed_run((4, 2, 1), kind="chat")  # warm-up

    speed = {kind: _batch_speed(kind, grid, latency, k, args.repeat) for kind in kinds}
    labels = list(speed["llm"])
    rows = [
        [label]
        + [
            f"{speed[kind][label]:.3f} s ({speed[kind][labels[0]] / speed[kind][label]:.1f}x)"
            for kind in kinds
        ]
        for label in labels
    ]
    print("Time for all prompts (speed-up over the sequential loop in brackets)\n")
    print(format_table(["Approach", "LLM-type model", "Chat model"], rows))
    print(
        "\nThe run() row includes everything a run does besides the model calls: progress bar, "
        "summary, recording the answers."
    )

    lost = {kind: _batch_lost(kind, grid, k) for kind in kinds}
    rows = [[label, lost["llm"][label], lost["chat"][label]] for label in lost["llm"]]
    print(f"\nAnswers lost when one of {prompts} prompts fails (item 3, persona 2)\n")
    print(format_table(["Approach", "LLM-type model", "Chat model"], rows))

    arrivals = _ArrivalTimes()
    experiment = build_experiment(*grid, kind="chat", latency=latency, model_id=MODEL_ID)
    start = time.perf_counter()
    experiment.run(callbacks=[arrivals], max_concurrency=k, show_progress=False)
    total = time.perf_counter() - start
    first = arrivals.times[0] - start
    print(
        f"\nrupsycho hands the first answer to the callbacks after {first * 1000:.0f} ms "
        f"(run total {total * 1000:.0f} ms);\nchain.batch returns nothing before the whole batch "
        "has finished."
    )
    return {"grid": grid_label(grid), "speed_s": speed, "lost": lost, "first_answer_s": first}


# ------------------- Mode: seeding -------------------


def mode_seeding(args: argparse.Namespace) -> dict[str, Any]:
    """Per-call cost of ``model.bind(seed=...)`` versus a model copy with the seed set."""
    grid = custom_grid(args) or parse_grid(SEEDING_GRID)
    samples = max(args.repeat * 4, 12)
    header(
        f"Cost of handing the seed to the model ({grid_size(*grid):,} calls per sample, "
        f"grid {grid_label(grid)})",
        args,
    )

    experiment = build_experiment(*grid, kind=args.model_kind, model_id=MODEL_ID)
    inputs = chain_inputs(experiment)
    model = experiment.runnable_models[MODEL_ID]
    chains = {
        "model.bind(seed=...)  (pre-restructure release)": (
            RunnablePassthrough()
            | experiment.runnable_prompt
            | model.bind(seed=1)
            | StrOutputParser()
        ).with_config(run_name=MODEL_ID),
        "seed_model(model, seed)  (current)": make_chain(experiment, 1),
    }
    chains["model.bind(seed=...)  (pre-restructure release)"].invoke(inputs[0])  # warm-up
    times: dict[str, list[float]] = {label: [] for label in chains}
    order = list(chains) * samples
    random.Random(0).shuffle(order)  # random order, so that drifts of the machine hit both alike
    for label in order:
        gc.collect()
        start = time.perf_counter()
        for values in inputs:
            chains[label].invoke(values)
        times[label].append((time.perf_counter() - start) / len(inputs) * 1e6)

    build_times = []
    for _ in range(samples):
        start = time.perf_counter()
        chain_inputs(experiment)
        build_times.append((time.perf_counter() - start) / len(inputs) * 1e6)

    current = min(times["seed_model(model, seed)  (current)"])
    rows = [
        [
            label,
            f"{min(values):,.1f}",
            f"{statistics.median(values):,.1f}",
            f"{min(values) - current:+.1f}",
        ]
        for label, values in times.items()
    ]
    rows.append(
        [
            "building the prompt variables of one call (done once per item and persona now)",
            f"{min(build_times):,.1f}",
            f"{statistics.median(build_times):,.1f}",
            "",
        ]
    )
    print(
        format_table(
            ["Per call", "best (us)", "median (us)", "best vs current (us)"],
            rows,
        )
    )
    print(f"\n{samples} samples per row, each over {len(inputs):,} calls, random order.")
    return {
        "grid": grid_label(grid),
        "us_per_call": {label: stats_of(values) for label, values in times.items()},
        "build_inputs_us": stats_of(build_times),
    }


# ------------------- Command line -------------------


def int_list(text: str) -> list[int]:
    """Parse ``"1,2,4"`` into ``[1, 2, 4]`` (argparse type)."""
    try:
        values = [int(part) for part in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {text!r}"
        ) from error
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("values must be at least 1")
    return values


def build_parser() -> argparse.ArgumentParser:
    """The command line of the script."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("Examples::")[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python benchmarks/bench_run_loop.py --repeat 5\n"
        "  python benchmarks/bench_run_loop.py --items 50 --personas 100 --seeds 3 --profile\n"
        "  python benchmarks/bench_run_loop.py --mode latency --max-concurrency 1,2,4,8,16\n"
        "  python benchmarks/bench_run_loop.py --mode batch\n"
        "  python benchmarks/bench_run_loop.py --mode seeding",
    )
    parser.add_argument(
        "--mode", choices=("overhead", "latency", "batch", "seeding"), default="overhead"
    )
    parser.add_argument(
        "--items",
        type=int,
        help="questionnaire items; any of --items/--personas/--seeds selects one custom grid",
    )
    parser.add_argument(
        "--personas", type=int, help="personas (default 10 when a custom grid is used)"
    )
    parser.add_argument("--seeds", type=int, help="seeds (default 1 when a custom grid is used)")
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="repetitions per measurement; median and min are reported (default 3)",
    )
    parser.add_argument(
        "--model-kind",
        choices=("llm", "chat"),
        default="chat",
        help="fake model type (default chat)",
    )
    parser.add_argument(
        "--cumulative", action="store_true", help="run with cumulative=True (response memory)"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="overhead mode: cProfile one run() of the largest grid",
    )
    parser.add_argument(
        "--latency-ms",
        type=float,
        help="latency/batch mode: seconds each call sleeps, in milliseconds (default 20)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int_list,
        help="latency mode: comma-separated worker counts (default 1,2,4,8,16); batch mode: the first value is used (default 8)",
    )
    parser.add_argument("--json", metavar="FILE", help="also write the raw results as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected mode; returns a non-zero exit code if results are not identical."""
    args = build_parser().parse_args(argv)
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    modes = {
        "overhead": mode_overhead,
        "latency": mode_latency,
        "batch": mode_batch,
        "seeding": mode_seeding,
    }
    result = modes[args.mode](args)
    if args.json:
        payload = {
            "mode": args.mode,
            "machine": machine_description(),
            "libraries": library_versions(),
            "revision": git_revision(),
            "args": vars(args),
            "result": result,
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nraw results written to {args.json}")
    return 0 if result.get("identical", True) else 1


if __name__ == "__main__":
    sys.exit(main())
