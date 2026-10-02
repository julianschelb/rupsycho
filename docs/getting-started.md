# Getting Started

## Installation

R.U.Psycho requires Python 3.10 or newer (tested on 3.10 – 3.14).

```bash
pip install git+https://github.com/julianschelb/rupsycho.git
```

The core package is light: configurations, the run loop, parsers, scoring and the command line.
Model back-ends are optional extras:

| Extra          | Installs                                                                 |
| -------------- | ------------------------------------------------------------------------ |
| `huggingface`  | PyTorch, Transformers, accelerate, sentencepiece, langchain-huggingface – local models, endpoints, model-based parsers |
| `openai`       | langchain-openai                                                         |
| `ollama`       | langchain-ollama                                                         |
| `google`       | langchain-google-genai                                                   |
| `deepseek`     | langchain-deepseek                                                       |
| `models`       | all of the above                                                         |
| `configurator` | Streamlit, pypdf and OpenAI – for the `rup-configurator` app             |
| `notebook`     | ipywidgets – notebook progress bars                                      |
| `quantization` | bitsandbytes – 4/8-bit loading of local models                           |
| `all`          | everything (bitsandbytes is skipped on macOS)                            |

For a CPU-only environment install PyTorch from the CPU index first
(`pip install torch --index-url https://download.pytorch.org/whl/cpu`), then:

```bash
pip install "rupsycho[huggingface,configurator] @ git+https://github.com/julianschelb/rupsycho.git"
```

The `dev`, `docs` and `test` extras are for contributors, see [Development](development.md).

## Your first experiment

An experiment is described by a JSON file or dictionary. A ready-to-run Big Five Inventory
example (five items, two personas) is bundled with the package:

```python
import rupsycho as rup

# 1. Load and validate the bundled example (rup.list_examples() lists them)
experiment = rup.load_example_experiment("bfi", seeds=[1, 2, 3])

# 2. Inspect the exact prompt the model will see
experiment.print_assembled_prompt(item_idx=1, persona_idx=0)

# 3. Run every model × seed × persona × item combination
summary = experiment.run()
print(summary)                      # "30 model calls in 41.2s"

# 4. Collect the answers
df = experiment.get_answers_as_dataframe()
print(df.head())
```

Or from the shell: `rupsycho examples copy bfi bfi.json`, then `rupsycho run bfi.json -o results.csv`
(see the [CLI reference](cli.md)). To use your own configuration, load it with
`rup.experiment_from_file("config.json")`.

The example has 5 items, 2 personas and 3 seeds, so `run()` sends 30 prompts. The result has one
row per generated answer with the columns `Instruction ID`, `Instruction Question`, `Model ID`,
`Persona ID`, `Run Seed` and `Answer`.

!!! note "Model downloads"
    The example runs `Qwen/Qwen2.5-0.5B-Instruct` (about 500 million parameters) on the CPU. It is
    downloaded from the Hugging Face Hub when `run()` first needs it, not when the experiment is
    loaded, and needs the `huggingface` extra. See [Models](tutorials/models.md) for hosted and
    local alternatives, or pass `models={}` to `load_example_experiment` and add your own.

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

!!! tip "What a configuration needs"
    To get answers you need a `questionnaire` with answer options, at least one entry in
    `demographic_profiles` and at least one model. `parameters` is optional (set `seeds` for
    reproducible runs). All keys are listed in the [Configuration Reference](configuration.md).

## Next steps

- Understand how the pieces fit together in [Concepts](concepts.md).
- Look up every field in the [Configuration Reference](configuration.md).
- Run, inspect and export experiments: [Running Experiments](tutorials/running-experiments.md).
- Clean the raw answers: [Parsing & Postprocessing](tutorials/postprocessing.md), then compute scores: [Scoring](tutorials/scoring.md).
- Make your results reproducible: [Reproducibility](tutorials/reproducibility.md).
