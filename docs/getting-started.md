# Getting Started

## Installation

R.U.Psycho requires Python 3.10 – 3.13.

```bash
pip install git+https://github.com/julianschelb/rupsycho.git
```

The package depends on PyTorch and Transformers, so the installation is large. For a
CPU-only environment, install PyTorch from the CPU index first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Optional extras:

| Extra          | Installs                                                      |
| -------------- | ------------------------------------------------------------- |
| `configurator` | Streamlit, pypdf and OpenAI – for the `rup-configurator` app  |
| `notebook`     | ipywidgets – notebook progress bars                           |
| `quantization` | bitsandbytes – 4/8-bit loading of local models                |
| `all`          | all of the above (bitsandbytes is skipped on macOS)           |

```bash
pip install "rupsycho[configurator] @ git+https://github.com/julianschelb/rupsycho.git"
```

The `dev`, `docs` and `test` extras are for contributors, see [Development](development.md).

## Your first experiment

An experiment is described by a JSON file. The repository ships several in
[`examples/data/`](https://github.com/julianschelb/rupsycho/tree/main/examples/data),
for instance the Big Five Inventory (BFI). These files are not part of the installed
package, so download the demo configuration first (or clone the repository and use
`examples/data/bfi_demo_config.json`):

```bash
curl -O https://raw.githubusercontent.com/julianschelb/rupsycho/main/examples/data/bfi_demo_config.json
```

```python
import rupsycho as rup

# 1. Load and validate the configuration
experiment = rup.experiment_from_file("bfi_demo_config.json")

# 2. Inspect the exact prompt the model will see
experiment.print_assembled_prompt(item_idx=0, persona_idx=0)

# 3. Run every model × seed × persona × item combination
experiment.run()

# 4. Collect the answers
df = experiment.get_answers_as_dataframe()
print(df.head())
```

The demo has 4 items, 2 personas, 1 seed and 1 model, so `run()` sends 8 prompts. The
result has one row per generated answer with the columns `Instruction ID`,
`Instruction Question`, `Model ID`, `Persona ID`, `Run Seed` and `Answer`.

!!! note "Model downloads"
    The demo configuration runs `HuggingFaceTB/SmolLM-1.7b-Instruct` (about 1.7 billion
    parameters) on the CPU. The model is downloaded from the Hugging Face Hub when `run()`
    first needs it, not when the configuration is loaded. See [Models](tutorials/models.md)
    for hosted and local alternatives.

## Building an experiment in Python

You do not need a JSON file. Dictionaries work the same way, and models can be added
programmatically:

```python
import rupsycho as rup
from langchain_core.language_models.fake import FakeListLLM

config = {
    "name": "Tiny demo",
    "parameters": {"seeds": ["1"]},
    "models": {},  # no config-defined models; we add one in code below
    "prompt_template": {
        "type": "chat",
        "messages": [
            {"role": "system", "content": "Act as {persona_description}. {general_instruction}"},
            {"role": "user", "content": "Question: {question}\nAnswer Options: {answer_options}\nAnswer:"},
        ],
    },
    "demographic_profiles": {
        "Anna": {"attributes": {"name": "Anna", "age": 30}, "template": "{name}, {age} years old"}
    },
    "questionnaire": {
        "name": "Mini",
        "general_instruction": "Rate the statement.",
        "default_answer_options": {
            "1": {"text": "1. Disagree", "weight": 1},
            "2": {"text": "2. Agree", "weight": 2},
        },
        "instruction_items": [{"question": "I like tests."}],
    },
}

experiment = rup.experiment_from_dict(config)
experiment.add_model(FakeListLLM(responses=["2"]), identifier="fake")  # any LangChain model
experiment.run()
```

!!! note "Default model"
    If the configuration has **no `models` key at all**, R.U.Psycho adds a small default
    model (`HuggingFaceTB/SmolLM-1.7b-Instruct`, on CPU) under the id `default_model` so
    that an experiment can be run out of the box. Use `"models": {}` to opt out; models
    added with `add_model` run in addition to the default model.

!!! warning "Keys a configuration must contain"
    Always include the `parameters` key (an empty object `{}` is fine): a configuration
    without it cannot be loaded. To get any answers you also need a `questionnaire` with
    answer options, at least one entry in `demographic_profiles` and at least one model.
    All keys are listed in the [Configuration Reference](configuration.md).

## Next steps

- Understand how the pieces fit together in [Concepts](concepts.md).
- Look up every field in the [Configuration Reference](configuration.md).
- Run, inspect and export experiments: [Running Experiments](tutorials/running-experiments.md).
- Clean and score the raw answers: [Parsing & Postprocessing](tutorials/postprocessing.md).
