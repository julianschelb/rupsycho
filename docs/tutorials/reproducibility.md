# Reproducibility

An experiment is reproducible when somebody else can run the same configuration and get the same
answers. R.U.Psycho supports this in three ways: the whole experiment is **one configuration**,
every call carries a **seed**, and the **model version** can be pinned. This page explains what is
guaranteed and what is not.

## Seeds

`parameters.seeds` lists the repetitions of an experiment. Every model is asked every question once
per seed. Integers and numeric strings are accepted (`[1, 2, 3]` and `["1", "2", "3"]` are the same);
the seeds are stored as strings because they are keys of the stored answers.

!!! warning "Set seeds explicitly"
    If `seeds` is omitted, **one random seed is drawn per experiment**. Runs are only reproducible
    if you write the seeds into the configuration.

The seed of a call is applied in the way each back-end supports it:

| Back-end | How the seed is applied |
| --- | --- |
| Local Hugging Face (`local_huggingface`) | `transformers.set_seed(seed)` immediately before every call (Python, NumPy and PyTorch generators) |
| OpenAI, DeepSeek | the `seed` parameter of the API (best effort on the provider's side) |
| Ollama | the `seed` option of the request |
| Hugging Face endpoint (`remote_huggingface`) | the `seed` parameter of the chat completion request |
| Google Gemini | **cannot be seeded**: a `UserWarning` is emitted once per model type; repetitions are independent samples |
| any other LangChain model | not seeded (warning); register a strategy with `rupsycho.seeding.register_seeder` |

The seed is the same for every call of a repetition, so the result of a call depends only on the
model, the prompt and the seed — **not** on the order of the calls or on `max_concurrency`.

!!! note "What older versions did"
    Before this behaviour was introduced, local Hugging Face models ignored the seed (the same seed
    gave different text) and every Ollama call failed because the seed was passed as an unsupported
    keyword. Results produced with those versions are not reproducible with seeds.

## What can still differ

Seeds fix the random numbers, not the arithmetic. Outputs can differ when

- the hardware or the kernels differ (GPU vs CPU, different GPUs, MPS) or non-deterministic kernels are used,
- the versions of `torch`, `transformers` or the tokenizer differ,
- the batch size or padding differs (the run loop asks one prompt at a time),
- an API provider changes or retires a model, or does not honour seeds strictly.

Therefore: **record your environment** (`rupsycho.__version__`, `pip freeze`, hardware) next to the
results, and treat seeded API calls as "mostly" reproducible.

## Pin the model version

For Hugging Face models pin `revision` to a commit hash. It is applied to the model and the tokenizer:

```json
"models": {
  "smollm": {
    "type": "local_huggingface",
    "name_or_path": "HuggingFaceTB/SmolLM-1.7b-Instruct",
    "revision": "<commit hash>",
    "parameters": {"max_new_tokens": 64, "do_sample": true, "return_full_text": false}
  }
}
```

For API models use the dated model name your provider offers (for example `gpt-4o-mini-2024-07-18`).

## Share the experiment

```python
experiment.export_to_file("experiment.json")          # configuration + answers
experiment.export_to_file("definition.json", include_answers=False)
```

The export contains only the configuration (and the answers). API keys are masked and are read from
the environment (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`)
when the file is loaded again. The default prompt template is part of the contract: its exact text
is pinned by the test-suite, so experiments that rely on it do not change silently between releases.

## Check it yourself

```python
a = rup.experiment_from_file("config.json"); a.run()
b = rup.experiment_from_file("config.json"); b.run()
assert a.get_answers_as_dataframe().equals(b.get_answers_as_dataframe())
```

`examples/reproducibility_check.py` runs this check offline with a tiny local model.
