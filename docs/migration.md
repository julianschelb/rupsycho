# Migrating from the initial release

This page lists what changed since the first public version of the package (the state of the
`main` branch before the repository was restructured). Behaviour that was a bug is described as
fixed; breaking changes are marked.

## Installation

- The core package no longer installs PyTorch, Transformers or the provider SDKs. Install the
  extras you need: `huggingface`, `openai`, `ollama`, `google`, `deepseek`, or `models` for all.
  A missing extra raises an `ImportError` that names it.
- `scipy`, `nltk`, `protobuf`, `langchain`, `langchain-community` and `ipywidgets` are no longer
  dependencies (`ipywidgets` is the `notebook` extra).
- Python 3.10 – 3.14 are supported.

## Results

- **Seeds are applied** for local Hugging Face models, Ollama and Hugging Face endpoints (see
  [Reproducibility](tutorials/reproducibility.md)). Results of earlier versions cannot be
  reproduced with seeds; re-run if you need reproducibility.
- The generation parameters of the **default model** (used when a configuration has no `models`
  key) are now applied; they were silently dropped, so answers contained the whole prompt.
- `revision`, `cache_dir` and `huggingfacehub_api_token` of local models are now honoured.
- The **default prompt** keeps its exact original text (a formatter had changed two spaces).

## API changes

| Before | Now |
| --- | --- |
| `run()` returned `None` | returns a `RunSummary`; new arguments `max_concurrency`, `on_error`, `show_progress` |
| failed calls were printed and skipped | logged, counted in the summary, one `RuntimeWarning` at the end (`on_error="raise"` stops) |
| an experiment could only be run once | experiments can be run repeatedly |
| `experiment_from_file` returned `None` for invalid files; `experiment_from_dict` raised an empty `RuntimeError` | both raise `ValueError` (`FileNotFoundError` for a missing file) with a useful message |
| `experiments_from_files` found nothing → `RuntimeError` | `ValueError`; files are loaded in alphabetical order |
| `experiment.model_dump()` (the documented trimmed dump never ran) | `experiment.to_config()` / `export_to_file()`; API keys are masked |
| `api_key` fields were plain strings | `SecretStr`; the DeepSeek and Google keys of a config are used; environment fallbacks work |
| seeds had to be strings | integers or numeric strings; non-integers are rejected when the config is loaded |
| `add_model` / `add_persona` with an existing identifier kept the old entry (model: half replaced) | replace it and warn |
| `PromptRemovalCleaner.parse` returned a dict | returns the cleaned string |
| `PostprocessingPipeline` printed progress, ignored default answer options | logs, falls back to the questionnaire's default options, adds a `valid` column, returns the frame |
| `JSONLCallback`, `PrintCallback`, `PrintTableCallback` dropped every answer (missing `time` argument) | work; `JSONLCallback` rows no longer embed all answers collected so far |
| `import rupsycho` took seconds | takes milliseconds; names and sub-packages load on first use |
| `rupsycho.parser` (copy of `rupsycho.parsers.parser`) | `BasicParser` is an alias of `BasicCleaner` |
| `example_experiment_bfi` | deprecated, use `load_example_experiment("bfi", models={})` |

Removed: `rupsycho.chains`, `rupsycho.utils.plots`, `rupsycho.models.instruction`,
`rupsycho.models.profile`, `get_default_model`, the unused alternative prompt templates in
`rupsycho.prompts`, and the dead `ExperimentExportMixin.model_dump`.

## New

`rupsycho` command, `rupsycho.scoring`, `rupsycho.seeding`, bundled examples
(`rup.list_examples()`), `ExperimentDocument.assemble_prompt`, `to_config`, `RunSummary`, a
configurator that keeps reverse-keying flags and answer weights on import, benchmarks and a much
larger test-suite.
