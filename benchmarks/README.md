# Benchmarks

Reproducible measurements behind the claims of the
[Performance & Scaling](../docs/tutorials/performance.md) page. Nothing here needs network
access or model downloads: the "model" is a deterministic fake that answers instantly or sleeps
for a configurable time to imitate an API call.

| Script | Question | Default run time |
|---|---|---|
| `synthetic.py` | Builds experiments of any size and the fake models (module, plus a small CLI to inspect one). | seconds |
| `bench_run_loop.py` | What does the framework cost per model call (`--mode overhead`)? What does `max_concurrency` buy for API models (`--mode latency`)? Why is `chain.batch` not used (`--mode batch`)? What does the old `model.bind(seed=...)` layer cost (`--mode seeding`)? | 10-45 s per mode |
| `bench_legacy.py` | Is the rewritten run loop faster than the pre-restructure one (commit `94f99d7`)? Do both produce identical answers? | about 1 min |
| `bench_import.py` | How long does `import rupsycho` take, and which heavy modules does it load? | about 1.5 min |

Every script prints GitHub-flavoured Markdown tables, so results can be pasted into the docs,
and takes `--help` and `--json FILE` (raw numbers).

## Setup

```bash
pip install -e ".[dev]"      # run-loop benchmarks: core dependencies are enough
pip install -e ".[models]"   # additionally needed to import the old tree (bench_legacy, bench_import)
```

The old release imported every model back-end (`torch`, `transformers`, `langchain-huggingface`,
`langchain-openai`, ...) as soon as `rupsycho` was imported, so `bench_legacy.py` and
`bench_import.py` only work in an environment that has them. `bench_legacy.py` and
`bench_import.py` extract the old source tree with `git archive 94f99d7 src` into a temporary
directory; in a shallow clone pass an already extracted tree with `--legacy-src DIR/src`.

## What each script does

### `bench_run_loop.py --mode overhead` (default)

Asks a zero-latency fake model every (item, persona, seed) of grids from 8 to 15 000 calls with
`max_concurrency=1`. Besides the total times it separates three things:

* the **fixed cost** of one `run()` (a 1-call run with an instant stand-in for the chain),
* what LangChain's `chain.invoke` costs per call **on its own** (the same chain in a plain loop,
  without rupsycho), and
* what rupsycho's own loop adds per call (`run()` with an instant chain).

The per-call costs are taken from the best of many short, interleaved bursts, because a short
burst is far more likely to run undisturbed on a busy machine than one long run.

```bash
python benchmarks/bench_run_loop.py                                    # five grids, 8 .. 15 000 calls
python benchmarks/bench_run_loop.py --items 50 --personas 100 --seeds 3 --repeat 5 --profile
python benchmarks/bench_run_loop.py --cumulative --items 50 --personas 20   # response-memory mode
python benchmarks/bench_run_loop.py --model-kind llm                   # text-completion instead of chat fake
```

`--profile` runs one `run()` of the largest grid under `cProfile` and prints the heaviest
functions plus how the time is shared between the model call, the bookkeeping and the memory
clean-up.

### `bench_run_loop.py --mode latency`

Every call sleeps `--latency-ms` (default 20). The same grid (default 16 items x 8 personas x 2
seeds = 256 calls) runs once per `--max-concurrency` value; the report shows the measured
speed-up next to the ideal one and checks that **answers and callback order are identical** for
all worker counts (non-zero exit code otherwise).

```bash
python benchmarks/bench_run_loop.py --mode latency --latency-ms 20 --max-concurrency 1,2,4,8,16
python benchmarks/bench_run_loop.py --mode latency --latency-ms 0 --max-concurrency 1,8   # CPU-bound: no gain
python benchmarks/bench_run_loop.py --mode latency --cumulative --max-concurrency 1,4,8,16
```

### `bench_run_loop.py --mode batch`

Compares `chain.batch` with the thread pool of the run loop for a text-completion (`LLM`) fake and
a chat-model fake, and counts how many answers are lost when exactly one prompt fails.

```bash
python benchmarks/bench_run_loop.py --mode batch
python benchmarks/bench_run_loop.py --mode batch --items 16 --personas 8 --repeat 5   # longer run
```

### `bench_run_loop.py --mode seeding`

The pre-restructure release handed the seed to the model with `model.bind(seed=...)`, an extra
`RunnableBinding` layer on every call; `seed_model` sets the seed on a copy of the model instead.
This mode times the same chain both ways in random order, and also times how long it takes to
build the prompt variables of one call (now done once per item and persona, not per call).

```bash
python benchmarks/bench_run_loop.py --mode seeding
```

### `bench_legacy.py`

Runs the same synthetic experiment with the same fake model in two subprocesses, one importing
the old tree (`PYTHONPATH` points at the `git archive` of `94f99d7`), one importing the current
tree. Both run sequentially (the old release has no concurrency), progress-bar output is
discarded, and the old and new subprocess alternate for `--rounds` rounds. Every run is
fingerprinted: the answers stored on the questionnaire items and the order of the callbacks must
be identical, otherwise the script exits with an error.

The old `run()` has no `show_progress` argument and always draws a progress bar, so the current
code is timed twice: with its default progress bar (like for like) and with
`show_progress=False`.

```bash
python benchmarks/bench_legacy.py                                       # three grids, up to 3 000 calls
python benchmarks/bench_legacy.py --items 50 --personas 100 --seeds 3 --rounds 2 --repeat 3
python benchmarks/bench_legacy.py --legacy-src /path/to/extracted/src   # without git history
```

### `bench_import.py`

Starts a fresh interpreter per sample, imports `rupsycho` and reports the wall time of the
import statement, of the first use of the public API (`rupsycho.ExperimentDocument`) and of both
together (median and minimum of `--runs`, default 7, after one discarded warm-up), the time of
the whole process, the number of modules and which heavy modules (`torch`, `transformers`,
provider SDKs, `IPython`, ...) ended up in `sys.modules`, for the current and the old tree.

The current tree imports lazily (PEP 562): `import rupsycho` only registers the public names and
the experiment machinery is loaded on first use, so the sum of both steps is what to compare
with the old release, which loaded everything in the `import` statement.

`langchain_core` imports `transformers` (and with it `torch`) by itself whenever they are
installed, so the benefit of the optional extras only shows in an environment that has just the
core dependencies. Create one and pass it with `--python`:

```bash
uv venv --python 3.12 /tmp/rupsycho-minimal
uv pip install --python /tmp/rupsycho-minimal/bin/python -e .      # core dependencies only

python benchmarks/bench_import.py
python benchmarks/bench_import.py --importtime \
    --python "full=$(which python)" --python "core only=/tmp/rupsycho-minimal/bin/python"
```

### `synthetic.py`

`synthetic_config(n_items, n_personas, n_seeds, n_options)` returns a configuration dictionary
of any size with `"models": {}` (same prompt, instruction text and answer-option layout as the
bundled BFI example). `build_experiment(...)` adds a `ScriptedLLM` or `ScriptedChatModel`, which
answers `{answer: "<k>"}` with `k` derived from the seed and the full prompt, so identical
answers prove identical prompts and seeds. `python benchmarks/synthetic.py --show-prompt` prints
one assembled prompt.

## Reading the numbers

* **Noise.** Timings on a shared laptop vary with whatever else runs. The reports therefore show
  the **minimum** (the run least disturbed by other work; used for all per-call figures) and the
  **median** (the typical run), alternate the variants that are compared, and take several
  repetitions. For numbers you want to quote, close other programs and increase `--repeat`
  (`--rounds` for the legacy comparison).
* **Fixed versus per-call cost.** A `run()` with ready models (what these benchmarks use) has
  almost no fixed cost; a model that is loaded lazily from the configuration is released after
  its turn with a `gc.collect()`, whose duration grows with the number of objects the process
  has imported (the overhead report prints it). The old release paid such a `gc.collect()` on
  every run, which is why small experiments gain most in `bench_legacy.py`. The per-call
  figures come from large grids or many short bursts, where the constant does not matter.
* **`time.sleep` overshoots**, especially on macOS under load. The latency report prints how long
  a "20 ms" sleep really takes here; speed-ups are measured against the measured sequential time
  and are unaffected.
* **What the fake model leaves out.** The fake answers in microseconds, so the numbers isolate
  framework cost. With a real model the call itself (tens of milliseconds for an API, seconds
  for a local model) dominates; see the docs page for how to use these numbers.

## Commands behind the numbers in the docs

The tables of the performance page were produced with:

```bash
python benchmarks/bench_run_loop.py --repeat 5
python benchmarks/bench_run_loop.py --mode latency --repeat 3
python benchmarks/bench_run_loop.py --mode batch --items 16 --personas 8 --repeat 5
python benchmarks/bench_run_loop.py --mode seeding --repeat 5
python benchmarks/bench_legacy.py --rounds 3 --repeat 5
python benchmarks/bench_legacy.py --items 50 --personas 100 --seeds 3 --rounds 2 --repeat 3
python benchmarks/bench_import.py --importtime \
    --python "full=$(which python)" --python "core only=/tmp/rupsycho-minimal/bin/python"
```
