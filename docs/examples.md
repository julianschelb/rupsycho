# Examples

Notebooks in [`examples/`](https://github.com/julianschelb/rupsycho/tree/main/examples).
They use paths such as `./data/bfi_demo_config.json`, so open them from the `examples/`
directory of a clone of the repository.

| Notebook                                                                                         | Shows                                          |
| ------------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| [Basics](https://github.com/julianschelb/rupsycho/blob/main/examples/Basics.ipynb)               | Load, run and inspect an experiment            |
| [Callbacks](https://github.com/julianschelb/rupsycho/blob/main/examples/Callbacks.ipynb)         | Stream answers while the experiment runs       |
| [Parsers](https://github.com/julianschelb/rupsycho/blob/main/examples/Parsers.ipynb)             | Cleaners, validators and judges                |
| [Postprocessing](https://github.com/julianschelb/rupsycho/blob/main/examples/Postprocessing.ipynb) | End-to-end postprocessing pipeline           |

Ready-made configurations in
[`examples/data/`](https://github.com/julianschelb/rupsycho/tree/main/examples/data):

| File                                         | Questionnaire and setup |
| -------------------------------------------- | ----------------------- |
| `bfi_demo_config.json`                       | Big Five Inventory (BFI): 4 items, 2 personas, one small local model. The demo of the [Getting Started](getting-started.md) guide |
| `bfi_small_and_mid.json`                     | BFI with 44 items, 250 personas and five Qwen2.5 models from 0.5B to 14B parameters (named "Experiment 2" in the file) |
| `bdi_qwen72.json`                            | Beck's Depression Inventory: 21 items, 250 personas, Qwen2.5-72B ("Experiment 5") |
| `trolley_qwen72.json`                        | Trolley dilemma: 3 items, 250 personas, Qwen2.5-72B ("Experiment 4") |
| `gsdb_new_qwen.json`                         | Gender/Sex Diversity Beliefs Scale (GSDB): 23 items, 250 personas, Qwen2.5-72B ("Experiment 3") |
| `rfq_*_small.json`                           | Regulatory Focus Questionnaire (Higgins et al., 2001): 11 items, 250 personas, Qwen2.5-0.5B, in three prompt formats: `json_format`, `model_friendly` and `natural_language` ("Experiment 1") |
| `example_config_for_demographics_judge.json` | Demographic questions (gender, age) for the `DemographicsJudge`, three seeds, Qwen2.5-7B |

The configurations with 250 personas and the 72B models are large studies that need
substantial GPU memory; start with `bfi_demo_config.json`. `examples/data/output/` contains
sample results of the demo (CSV, JSONL and the postprocessed CSV) that the Postprocessing
notebook reads.

The materials from the user study are in
[`examples/user_testing/`](https://github.com/julianschelb/rupsycho/tree/main/examples/user_testing):
step-by-step tasks and a notebook around a small preferences survey.
