# Getting Started

## Installation

R.U.Psycho requires Python 3.10 – 3.13.

```bash
pip install git+https://github.com/julianschelb/rupsycho.git
```

Optional extras:

| Extra          | Installs                                               |
| -------------- | ------------------------------------------------------ |
| `configurator` | Streamlit, pypdf, OpenAI – for the `rup-configurator` app |
| `notebook`     | ipywidgets – notebook progress bars                    |
| `quantization` | bitsandbytes – 4/8-bit loading of local models         |
| `all`          | everything above                                       |

```bash
pip install "rupsycho[configurator] @ git+https://github.com/julianschelb/rupsycho.git"
```

## Your first experiment

An experiment is described by a JSON file. The repository ships several in
[`examples/data/`](https://github.com/julianschelb/rupsycho/tree/main/examples/data),
for instance the Big Five Inventory (BFI):

```python
import rupsycho as rup

# 1. Load and validate the configuration
experiment = rup.experiment_from_file("examples/data/bfi_demo_config.json")

# 2. Inspect the exact prompt the model will see
experiment.print_assembled_prompt(item_idx=0, persona_idx=0)

# 3. Run every model × seed × persona × item combination
experiment.run()

# 4. Collect the answers
df = experiment.get_answers_as_dataframe()
print(df.head())
```

The result has one row per combination with the columns `Instruction ID`,
`Instruction Question`, `Model ID`, `Persona ID`, `Run Seed` and `Answer`.

!!! note "Model downloads"
    The demo configuration uses a small local Hugging Face model. It is downloaded on
    first use. See [Models](tutorials/models.md) for hosted and local alternatives.

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
    model (`HuggingFaceTB/SmolLM-1.7b-Instruct`, on CPU) so that an experiment can be run
    out of the box. Use `"models": {}` to opt out.

## Next steps

- Understand how the pieces fit together in [Concepts](concepts.md).
- Look up every field in the [Configuration Reference](configuration.md).
- Clean and score the raw answers: [Parsing & Postprocessing](tutorials/postprocessing.md).
