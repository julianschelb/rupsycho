# Performance & Scaling

An experiment spends its time waiting for models. The run loop therefore has three jobs: add as
little overhead as possible per call, keep slow API models busy by running calls concurrently,
and stay **exactly reproducible** while doing so. This page explains how a run is organised, what
was measured against the pre-restructure implementation (commit `94f99d7`), and how to choose
`max_concurrency`.

!!! note "Summary of the measurements below"
    * The framework costs about 218 µs per model call, almost all of it LangChain's own
      `chain.invoke`; the run loop itself adds about 1.7 µs. A run with ready models has no
      fixed cost to speak of (0.8 ms), so even an 8-call experiment finishes in
      2 ms.
    * Against the old loop, **small experiments are much faster** (4.7× at 100 calls),
      because the old loop paid 81 ms per `run()`. **Per call the gain is small**
      (236 → 218 µs), so large experiments are only 1.2× faster: both versions
      spend nearly all their time in LangChain.
    * The real speed-up comes from concurrency: `max_concurrency=16` made a workload of 20 ms
      model calls 15× faster. The old loop could not run calls concurrently at all.
    * `import rupsycho` returns after 2 ms and loads the experiment machinery on first
      use. Ready to use takes well under 1 (no `transformers` to import) s with only the core dependencies, 1.9 s with
      `torch` and `transformers` installed, and took 2.9 s for the old release.

## How a run is organised

| When | What happens |
|---|---|
| once per `run()` | check that models, prompt, questionnaire and personas exist; create one progress bar (`show_progress=False` disables it); build the prompt variables of every (item, persona) pair, which depend neither on the model nor on the seed |
| once per model | load it if it is only a configuration (lazy loading); check whether calls may run concurrently; afterwards, if the model was loaded for this run, release it and run `gc.collect()` |
| once per seed | build the chain `prompt \| seeded model \| parser`; run the calls of this seed |
| per call | `chain.invoke(...)` (in a worker thread if `max_concurrency > 1`), then, **in the calling thread and in submission order**, store the answer on the item, call every callback and advance the progress bar |

Two details matter for speed:

* **Seeding without a wrapper.** The old release wrapped the model in `model.bind(seed=...)`,
  an extra `RunnableBinding` layer that every call passes through. `seed_model` instead returns
  a copy of the model with its `seed` set (OpenAI, DeepSeek, Ollama, ...), calls
  `transformers.set_seed` before each call for local pipelines, and warns once per model type for
  back-ends that cannot be seeded. The binding layer costs 18 µs per call
  (`bench_run_loop.py --mode seeding`).
* **Prompt variables are built once.** Building them costs only 1.3 µs per call, so this
  is tidiness more than speed.

## Concurrency semantics

`run(max_concurrency=k)` with `k > 1` runs the calls of one seed in a `ThreadPoolExecutor`.
Threads are the right tool because model calls are I/O bound: the waiting happens with the
GIL released.

* **Deterministic order.** Results are consumed in submission order, whichever call finishes
  first. Answers, callbacks (CSV/JSONL rows) and the `RunSummary` are therefore identical for every
  value of `max_concurrency`; the latency benchmark compares them and fails if they differ. The
  price is that one slow call holds back the recording of the calls behind it (the work itself
  continues).
* **Callbacks run in the calling thread.** Your own callbacks need no locking.
* **Per-call error isolation.** A failing call never affects the others. By default
  (`on_error="warn"`) it is logged, counted in `RunSummary.n_failed`, left without an answer,
  and one warning is raised at the end. `"ignore"` stays silent; `"raise"` raises the first
  failure in submission order and cancels calls that have not started yet. Calls already running
  finish and are discarded (in one test run with `max_concurrency=4`, 6 of 32 calls had started
  when the error stopped the run).
* **Local models are always sequential.** A local Hugging Face pipeline relies on process-global
  state (random generators, accelerator memory), so `max_concurrency` is ignored with a warning.
* **Cumulative mode** (`cumulative=True`): each persona's prompt contains its own earlier
  answers, so the items stay in order and only the personas of one item run side by side. The
  useful concurrency is at most the number of personas.
* **Seeds run one after another.** Each seed is a separate pass over the grid with its own pool.

### Why `chain.batch` is not used

LangChain offers `chain.batch(inputs, config={"max_concurrency": k})`, which looks like the
obvious tool. It was measured and not used (`bench_run_loop.py --mode batch`, 32
prompts, 20 ms per call, 8 workers):

| Approach | LLM-type model | Chat model |
|:--|--:|--:|
| sequential `chain.invoke` loop | 0.914 s (1.0×) | 0.921 s (1.0×) |
| `chain.batch(max_concurrency=8)` | 0.861 s (1.1×) | 0.120 s (7.7×) |
| `chain.batch()` without a limit | 0.890 s (1.0×) | 0.092 s (10.0×) |
| rupsycho `run(max_concurrency=8)` | 0.117 s (7.8×) | 0.126 s (7.3×) |

* For **text-completion models** (LangChain `LLM`, such as Ollama or local pipelines)
  `BaseLLM.batch` passes the prompts to one `generate` call that works through them one after
  the other (`OllamaLLM._generate` is exactly such a loop): `max_concurrency` brings **no
  speed-up**.
* For those models a **single failing prompt fails its whole chunk**: with
  `return_exceptions=True` every prompt of the chunk comes back as the exception.

| Answers lost when 1 of 32 prompts fails | LLM-type model | Chat model |
|:--|--:|--:|
| prompts that really fail | 1 | 1 |
| `chain.batch(max_concurrency=8, return_exceptions=True)` | 8 | 1 |
| `chain.batch(return_exceptions=True)`, no limit | 32 | 1 |
| rupsycho `run(max_concurrency=8)` | 1 | 1 |

* **Chat models** (OpenAI, DeepSeek, Gemini, ...) run `batch` in a thread pool, so for them
  `chain.batch` is about as fast as the run loop's pool and isolates errors per prompt. The run
  loop still avoids it, because it has to behave identically for every model type, and because a
  batch has no per-call timing (every callback receives the seconds that call took), cannot stop
  at the first failure, and returns nothing before the last prompt has finished. The run loop
  hands the first answer to the callbacks after 31 ms of a 115 ms run,
  so a long run that is interrupted keeps its partial results.
* A Hugging Face pipeline does not loop either: it hands `batch_size` prompts to the pipeline in
  one call, which batches on the device instead of running calls concurrently. The run loop asks
  one prompt at a time so that every call is seeded and timed on its own.

## Choosing `max_concurrency`

* **Local Hugging Face models:** leave it at 1 (it is ignored anyway). Speed there comes from the
  model: a GPU, a smaller model, shorter `max_new_tokens`.
* **Hosted APIs** (OpenAI, DeepSeek, Gemini, ...): the speed-up is close to linear until you hit
  the provider's rate limit, so start with 4 to 8 and raise it while no call fails. The request
  rate is roughly `max_concurrency / latency`: 8 workers and calls of 2 s make about 4 requests
  per second (240 per minute), to be compared with the requests-per-minute **and**
  tokens-per-minute limits of your account. Long prompts (many answer options, cumulative mode)
  use up the token limit first.
* **Rate-limit errors** are retried by the provider client before they reach rupsycho (in the
  installed versions: `openai` 1.109.1 retries twice by default, `langchain-google-genai` 2.1.12
  six times; both are settings of the model). A call that still fails is recorded as failed,
  not retried: check `summary.n_failed` after the run, lower `max_concurrency` or raise the
  client's `max_retries`, and run again. Use `on_error="raise"` if you would rather stop at the
  first failure; it cancels calls that have not started, but the calls in flight (up to
  `max_concurrency`) still finish and are usually billed.
* **Local servers** (for example Ollama) only gain if the server itself works on requests in
  parallel; check its documentation before raising the value.
* **Cumulative mode:** values above the number of personas change nothing.
* **Reproducibility:** order and content of the results do not depend on `max_concurrency`.
  Whether a provider returns the same text for the same seed is up to the provider.
* The cost of an experiment (tokens) is independent of `max_concurrency`; only the wall time
  changes.

For an example workload of 10 000 calls to an API that answers in 1 s, the framework's share is
about 2.2 s in total. The sequential run takes about 2.8 hours, eight workers about
21 minutes (arithmetic from the figures above, not a measurement).

## Reproduce

```bash
pip install -e ".[dev]"
python benchmarks/bench_run_loop.py --repeat 7                       # framework overhead
python benchmarks/bench_run_loop.py --model-kind llm                 # text-completion fake
python benchmarks/bench_run_loop.py --cumulative --items 50 --personas 20
python benchmarks/bench_run_loop.py --items 50 --personas 20 --seeds 3 --repeat 1 --profile
python benchmarks/bench_run_loop.py --mode seeding                   # bind() layer, prompt variables
python benchmarks/bench_run_loop.py --mode latency                   # thread-pool scaling
python benchmarks/bench_run_loop.py --mode latency --latency-ms 0 --max-concurrency 1,8 --items 50 --personas 20 --seeds 1
python benchmarks/bench_run_loop.py --mode batch --items 16 --personas 8 --repeat 5
pip install -e ".[models]"                                           # the old tree needs every back-end
python benchmarks/bench_legacy.py --rounds 3 --repeat 5
python benchmarks/bench_legacy.py --items 50 --personas 100 --seeds 3 --rounds 2 --repeat 3

uv venv --python 3.12 /tmp/rupsycho-minimal
uv pip install --python /tmp/rupsycho-minimal/bin/python -e .        # core dependencies only
python benchmarks/bench_import.py --importtime \
    --python "full=$(which python)" --python "core only=/tmp/rupsycho-minimal/bin/python"
```

Every script accepts `--help`. See the
[`benchmarks/README.md`](https://github.com/julianschelb/rupsycho/tree/main/benchmarks) for what
each one measures and how to read the output. Machine description for your own report:
`sysctl -n machdep.cpu.brand_string` (macOS) and `python --version`.

## Memory

* Models are loaded just before their turn and released afterwards, so several large models can
  be tested one after another on the same machine (`lazy_load_models`, the default). Ready models
  handed over with `add_model` stay alive and are not touched.
* Answers are kept on the questionnaire items as nested dictionaries (model, persona, seed);
  nothing else is accumulated. With `max_concurrency > 1` the results of one seed pass
  (items x personas) are held in memory until that pass is finished.
* For long runs add a `CSVCallback` or `JSONLCallback`: answers are written as they arrive, in
  order, so an interrupted run keeps what was finished.

## Validation

The concurrency semantics (order, error isolation, summary, repeated runs) are asserted by
`tests/test_run_semantics.py`; the benchmarks above additionally compare the answers of every
variant they time.

## Measured results

Measured on an Apple M1 (8 logical cores), macOS 15, Python 3.12.13, `langchain-core` 0.3.86,
`pydantic` 2.13.5, with scripted fake models (no network, no model download). Reproduce every
table with the commands below; the numbers are medians or best-of-N as stated.

### What the framework costs

```bash
python benchmarks/bench_run_loop.py --repeat 3
```

| Measurement | µs per call |
|:--|--:|
| `run()` | 217.7 |
| `chain.invoke` in a plain loop (LangChain alone, no rupsycho) | 211.8 |
| `run()` with an instant chain (rupsycho's own loop) | 1.7 |

A `run()` of 15 000 calls (50 items × 100 personas × 3 seeds) takes 3.3 s with a zero-latency
model. Almost all of that is LangChain building prompts and invoking the chain; **rupsycho's own
bookkeeping costs under 2 µs per call.** Real model calls take milliseconds to seconds, so the
framework is never the bottleneck.

### Compared with the pre-restructure loop

```bash
python benchmarks/bench_legacy.py --repeat 3
```

Same configuration, same scripted model; the answers and the callback order are identical for
both implementations (checked by the script).

| Grid (items × personas × seeds) | Calls | Legacy (s) | Current (s) | Speed-up |
|:--|--:|--:|--:|--:|
| 10 × 10 × 1 | 100 | 0.105 | 0.022 | 4.7× |
| 20 × 10 × 3 | 600 | 0.222 | 0.131 | 1.7× |
| 50 × 20 × 3 | 3 000 | 0.788 | 0.656 | 1.2× |

The gain comes from the **fixed cost of a run** (81 ms → 0.8 ms; the old loop garbage-collected
after every model even when nothing was released) and from building the prompt variables once. The
cost per call is only slightly lower (235.7 → 218.4 µs), so the advantage shrinks as runs get larger.

### Concurrency for API models

```bash
python benchmarks/bench_run_loop.py --mode latency --latency-ms 20
```

A model that waits 20 ms per call (sleeping really takes 30 ms here), 240 calls:

| `max_concurrency` | `run()` (s) | Speed-up | Same answers and callback order |
|--:|--:|--:|:--|
| 1 | 7.296 | 1.00× | yes |
| 2 | 3.648 | 2.00× | yes |
| 4 | 1.800 | 4.05× | yes |
| 8 | 0.924 | 7.89× | yes |
| 16 | 0.485 | 15.03× | yes |

Scaling is close to ideal as long as the calls spend their time waiting. Mind the rate limits of
your provider when choosing a value; local Hugging Face models always run sequentially.

### Import time

```bash
python benchmarks/bench_import.py
```

| Tree | `import rupsycho` (s) | first use of the API (s) | heavy modules loaded |
|:--|--:|--:|:--|
| current | 0.002 | 1.9 | none at import; `torch`, `transformers` on first use (imported by `langchain-core` when installed) |
| before | 2.9 | 0 | `torch`, `transformers`, five provider SDKs, `openai`, `IPython` |

`rupsycho --version`, `--help`, `validate` and `examples` therefore start instantly. In an
environment **without** `transformers` (a core install for API models only) the first use is
correspondingly cheaper.
