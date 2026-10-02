# Examples

Notebooks, runnable scripts and ready-made experiment configurations for R.U.Psycho. The
documentation page [Examples](https://julianschelb.github.io/rupsycho/examples/) describes the
same material.

**Start with [`quickstart.ipynb`](quickstart.ipynb).** It is executed, runs completely offline in
about ten seconds and walks through the whole workflow with a scripted stand-in model.

## Notebooks

| Notebook | What it shows | Needs | Stored outputs |
| --- | --- | --- | --- |
| [`quickstart.ipynb`](quickstart.ipynb) | The whole workflow: bundled example, assembled prompt, adding a model, run with a CSV callback, `RunSummary`, `on_error`, `max_concurrency`, seeds and reproducibility, cleaning, validating and judging answers, scores from answer option weights, the CLI | core package only, **offline** | yes, everything |
| [`Basics.ipynb`](Basics.ipynb) | Load a configuration, inspect it, run it with a real local model, collect and save the results | `huggingface` extra, **downloads a model** | up to the model cells |
| [`Callbacks.ipynb`](Callbacks.ipynb) | Stream answers to the console, CSV and JSONL while the experiment runs; the file formats; writing your own callback | `huggingface` extra, **downloads a model** | up to the model cells |
| [`Parsers.ipynb`](Parsers.ipynb) | Every cleaner, validator and judge on its own, parsers inside an experiment | core package; the two model-based parsers need the `huggingface` extra and a download | yes, except the two model-based cells |
| [`Postprocessing.ipynb`](Postprocessing.ipynb) | The `PostprocessingPipeline` on the sample results in `data/output/` | core package only, **offline** | yes, everything |

Cells that load a real model are tagged `skip-execution` and have no stored output (their
execution count is empty). Everything else was executed, so you can read the notebooks on GitHub
without running them. Install the model back-ends with an extra, for example
`pip install "rupsycho[huggingface,notebook] @ git+https://github.com/julianschelb/rupsycho.git"`.

## Scripts

Run them from the repository root, for example `python examples/reproducibility_check.py`.

| Script | What it shows | Needs |
| --- | --- | --- |
| [`run_experiment.py`](run_experiment.py) | Run a configuration (the bundled example or your own file) with a local Hugging Face model and stream the answers into a CSV file; `--dry-run` validates and previews without loading anything | `huggingface` extra, **downloads a model** (except with `--dry-run`) |
| [`reproducibility_check.py`](reproducibility_check.py) | Same seeds give the same answers, other seeds other answers, and what `rupsycho.seeding` does for models with and without seed support, including a tiny local Hugging Face model built on the fly | offline, about 6 s; the local-model part needs the `huggingface` extra and is skipped without it |
| [`custom_callback.py`](custom_callback.py) | A callback that stores every answer in a SQLite database (standard library only), incl. failed calls | offline, about 3 s |
| [`api_models.py`](api_models.py) | Configuration snippets and environment variables for OpenAI, Ollama, Google Gemini and DeepSeek; validates them without sending anything. `--run PROVIDER` runs the bundled example against one of them | offline by default; `--run` needs the extra, a key and network |
| [`make_sample_outputs.py`](make_sample_outputs.py) | Maintenance: regenerates `data/output/` (see below) | offline |
| [`execute_notebooks.py`](execute_notebooks.py) | Maintenance: executes the notebooks offline and, with `--write`, stores the outputs | `dev` extra (nbclient, ipykernel) |

All scripts have `--help`. The scripts that run offline use small *scripted stand-in models* that
are defined in the script and marked as such; they are not language models and exist only to make
the behaviour of the package observable without a download.

## Experiment configurations

[`data/`](data) holds ready-made configurations. All of them are validated by the test suite
(`tests/test_examples.py`). The first one is the small demo that the notebooks use; its twin ships
inside the package as the bundled example `bfi` (`rup.load_example_experiment("bfi")`). The others
are the larger study configurations (their names read "Experiment 1" to "Experiment 5"); they
have hundreds of personas and models up to 72B parameters, so look at them with
`python examples/run_experiment.py CONFIG --dry-run` (or `rupsycho run CONFIG --dry-run`) before
you start one.

| File | Questionnaire | Questions | Personas | Models | Model calls |
| --- | --- | ---: | ---: | ---: | ---: |
| `bfi_demo_config.json` | Big Five Inventory, 5 questions | 5 | 2 | 1 | 10 |
| `bfi_small_and_mid.json` | Big Five Inventory, impact of model size | 44 | 250 | 5 | 55 000 |
| `bdi_qwen72.json` | Beck's Depression Inventory, prompt order | 21 | 250 | 1 | 5 250 |
| `trolley_qwen72.json` | Trolley dilemma, prior knowledge | 3 | 250 | 1 | 750 |
| `gsdb_new_qwen.json` | Gender/Sex Diversity Beliefs Scale, bias detection | 23 | 250 | 1 | 5 750 |
| `rfq_json_format_small.json`, `rfq_model_friendly_small.json`, `rfq_natural_language_small.json` | Regulatory Focus Questionnaire in three prompt formats | 11 | 250 | 1 | 2 750 each |
| `example_config_for_demographics_judge.json` | Demographic questions for `DemographicsJudge` (3 seeds) | 2 | 1 | 1 | 6 |

## Sample results

[`data/output/`](data/output) contains result files in the exact format of `CSVCallback` and
`JSONLCallback`, plus the output of the post-processing pipeline. They were **not** produced by a
language model: [`make_sample_outputs.py`](make_sample_outputs.py) runs the demo configuration
through the real run loop with a scripted model (model id `scripted`) whose hand-written replies
imitate typical free-text answers. They serve [`Postprocessing.ipynb`](Postprocessing.ipynb) and
show the file formats. `tests/test_examples.py` checks that they are what the current callbacks
and pipeline write. If those change, run `python examples/make_sample_outputs.py` and then
`python examples/execute_notebooks.py --write`, which also rewrites the processed file.

## Re-executing the notebooks

```bash
pip install -e ".[dev]"
python examples/execute_notebooks.py            # run them all in a scratch copy, change nothing
python examples/execute_notebooks.py --write    # run in place and store the outputs
```

The script skips `skip-execution` cells, drops execution timings and merges stream chunks, so
re-executing an unchanged notebook gives an unchanged file. The test suite re-executes
`quickstart.ipynb` and runs the model cells of the other notebooks against a stand-in model.

## User study

[`user_testing/`](user_testing) contains the instructions, data and notebook of the user study
that was used to test the package.
