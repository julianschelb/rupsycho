# Examples

All examples live in the [`examples/`](https://github.com/julianschelb/rupsycho/tree/main/examples) folder of the repository. Start with the **quickstart notebook**: it is executed, needs no model download and no API key, and walks through the whole workflow in about ten seconds.

## Notebooks

| File | What it shows | Requirements |
| --- | --- | --- |
| [`quickstart.ipynb`](https://github.com/julianschelb/rupsycho/blob/main/examples/quickstart.ipynb) | Executed notebook: bundled example, assembled prompt, adding a model, running with a CSV callback, `RunSummary`, `on_error`, `max_concurrency`, seeds and reproducibility, cleaning / validating / judging answers, scores from the answer option weights, the CLI | none (offline, scripted stand-in model) |
| [`Basics.ipynb`](https://github.com/julianschelb/rupsycho/blob/main/examples/Basics.ipynb) | Load a configuration, inspect it, run it with a real local model, collect and save the answers | `huggingface` extra, downloads a model |
| [`Callbacks.ipynb`](https://github.com/julianschelb/rupsycho/blob/main/examples/Callbacks.ipynb) | Stream answers to the console, CSV and JSONL while the experiment runs; the file formats; a custom callback | `huggingface` extra, downloads a model |
| [`Parsers.ipynb`](https://github.com/julianschelb/rupsycho/blob/main/examples/Parsers.ipynb) | Every cleaner, validator and judge on its own, parsers as the output parser of an experiment | none; the two model-based parsers need the `huggingface` extra and a download |
| [`Postprocessing.ipynb`](https://github.com/julianschelb/rupsycho/blob/main/examples/Postprocessing.ipynb) | The `PostprocessingPipeline` on the sample results in `examples/data/output/` | none (offline) |

The notebooks are committed with their outputs, so you can read them on GitHub. Cells that load a real model are tagged `skip-execution` and have no stored output. Install the model back-ends with an [extra](getting-started.md), for example `pip install "rupsycho[huggingface,notebook]"`.

## Scripts

Run them from the repository root, for example `python examples/reproducibility_check.py`. All of them have `--help`.

| File | What it shows | Requirements |
| --- | --- | --- |
| [`run_experiment.py`](https://github.com/julianschelb/rupsycho/blob/main/examples/run_experiment.py) | Run a configuration with a local Hugging Face model and stream the answers into a CSV file; `--dry-run` validates and previews without loading anything | `huggingface` extra, **downloads a model** (except with `--dry-run`) |
| [`reproducibility_check.py`](https://github.com/julianschelb/rupsycho/blob/main/examples/reproducibility_check.py) | Same seeds give the same answers; what `rupsycho.seeding` does for models with and without seed support, including a tiny local Hugging Face model built on the fly | none (offline); the local-model part needs the `huggingface` extra |
| [`custom_callback.py`](https://github.com/julianschelb/rupsycho/blob/main/examples/custom_callback.py) | A callback that stores every answer, failed calls included, in a SQLite database | none (offline) |
| [`api_models.py`](https://github.com/julianschelb/rupsycho/blob/main/examples/api_models.py) | Configuration snippets and environment variables for OpenAI, Ollama, Google Gemini and DeepSeek; `--run` tries one of them | none by default; `--run` needs the extra, a key and network access |

The scripts that run offline use small *scripted stand-in models* that are defined in the script and marked as such. They are not language models; they only make the behaviour of the package observable without a download.

## Minimal end-to-end script

```python
import rupsycho as rup
from rupsycho.callbacks import CSVCallback
from rupsycho.parsers.cleaners import BasicCleaner
from rupsycho.parsers.judges import MultipleChoiceJudge
from rupsycho.parsers.validators import ValidatorParser
from rupsycho.postprocessing import PostprocessingPipeline

CONFIG = "examples/data/bfi_demo_config.json"

# 1. Load and validate the configuration: questionnaire, personas, models, prompt, seeds
experiment = rup.experiment_from_file(CONFIG)

# 2. Look at exactly what the model will see
experiment.print_assembled_prompt(item_idx=0, persona_idx=0)

# 3. Run every model x seed x persona x question; answers stream into a CSV file
#    (callbacks append: delete results.csv before you run this a second time)
summary = experiment.run(callbacks=[CSVCallback("results.csv")], on_error="warn")
print(summary)  # N model calls in T seconds

# 4. Clean, validate and judge the free-text answers
options = experiment.questionnaire.default_answer_options.get_options_as_list()
processed = PostprocessingPipeline(
    CONFIG,
    "results.csv",
    cleaner=BasicCleaner(),
    validator=ValidatorParser(),
    judge=MultipleChoiceJudge(options),
    output_path="processed.csv",
).run()
print(processed[["profile_id", "cleaned_answer", "valid", "decision"]])
```

The demo configuration loads `Qwen/Qwen2.5-0.5B-Instruct` (about 1 GB) on first use. To try the same code without a download, replace the models of the configuration and add a scripted one, as the quickstart notebook does:

```python
config = rup.load_example_config("bfi")
config["models"] = {}
experiment = rup.experiment_from_dict(config)
experiment.add_model(my_model, identifier="my-model")  # any LangChain model
```

## Experiment configurations

[`examples/data/`](https://github.com/julianschelb/rupsycho/tree/main/examples/data) holds ready-made configurations, all validated by the test suite. The demo has a twin inside the package: `rup.load_example_experiment("bfi")` loads the bundled copy. The others are the larger study configurations (their names read "Experiment 1" to "Experiment 5"); they have hundreds of personas and models up to 72B parameters, so look at them with `rupsycho run --dry-run` before you start one.

| File | Questionnaire | Questions | Personas | Models | Model calls |
| --- | --- | ---: | ---: | ---: | ---: |
| `bfi_demo_config.json` | Big Five Inventory, 5 questions | 5 | 2 | 1 | 10 |
| `bfi_small_and_mid.json` | Big Five Inventory, impact of model size | 44 | 250 | 5 | 55 000 |
| `bdi_qwen72.json` | Beck's Depression Inventory, prompt order | 21 | 250 | 1 | 5 250 |
| `trolley_qwen72.json` | Trolley dilemma, prior knowledge | 3 | 250 | 1 | 750 |
| `gsdb_new_qwen.json` | Gender/Sex Diversity Beliefs Scale, bias detection | 23 | 250 | 1 | 5 750 |
| `rfq_json_format_small.json`, `rfq_model_friendly_small.json`, `rfq_natural_language_small.json` | Regulatory Focus Questionnaire in three prompt formats | 11 | 250 | 1 | 2 750 each |
| `example_config_for_demographics_judge.json` | Demographic questions for the `DemographicsJudge` (3 seeds) | 2 | 1 | 1 | 6 |

`examples/data/output/` contains sample result files in the format of `CSVCallback` and `JSONLCallback` and the output of the post-processing pipeline. They were created offline with a scripted model (`examples/make_sample_outputs.py`), not by a language model. The materials of the user study are in `examples/user_testing/`.

## Command line

Everything above that does not need Python code is also available from the shell: `rupsycho validate`, `rupsycho prompt`, `rupsycho run --dry-run`, `rupsycho run -o results.csv`, `rupsycho postprocess`. See the [CLI reference](cli.md).

## Benchmarks

The [`benchmarks/`](https://github.com/julianschelb/rupsycho/tree/main/benchmarks) folder contains scripts that need no model download. `bench_run_loop.py` (`poe bench`) times the run loop with scripted models: the framework overhead per call and what `max_concurrency` buys for slow API models. `bench_import.py` measures the cost of `import rupsycho`, and `bench_legacy.py` compares the current implementation with the one from before the restructuring. The docstring of each script lists its options; [Running experiments](tutorials/running-experiments.md) explains the run options they exercise.
