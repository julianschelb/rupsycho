# R.U.Psycho

**R.U.Psycho** (*Robust Unified Psychometric Testing of Language Models*) is a Python
framework for designing and running **reproducible psychometric experiments on
generative language models**, with limited coding expertise required.

It accompanies the paper
[*R.U.Psycho? Robust Unified Psychometric Testing of Language Models*](https://arxiv.org/abs/2503.10229)
(Schelb, Borin, Garcia & Spitz, 2025).

## Why?

Psychometric testing of language models is easy to get wrong and hard to reproduce.
Model outputs are unstable, results are sensitive to prompt design and generation
parameters, and a large number of model versions are in circulation. All of this
increases the documentation burden for anyone who wants their results to be
comparable.

R.U.Psycho addresses this by making a whole experiment **one declarative,
shareable configuration**:

| You declare …                          | … in the config section |
| -------------------------------------- | ----------------------- |
| the questionnaire, items, answer scales | `questionnaire`         |
| who the model should answer *as*       | `demographic_profiles`  |
| which model(s) and generation settings | `models`                |
| how the question is phrased            | `prompt_template`       |
| which random seeds to use              | `parameters`            |

The package runs every combination of *model × seed × persona × item*, collects the
free-text answers, and provides cleaners, validators and judges to turn them into
valid, scorable responses.

## Features

- **Declarative experiments** – questionnaire, personas, models, prompt and seeds live
  in a single JSON file that doubles as documentation.
- **Many model back-ends** – local and remote Hugging Face, Ollama, OpenAI, Google
  Gemini, DeepSeek, or any LangChain runnable.
- **Systematic robustness checks** – vary seeds, prompts, personas and models without
  touching code.
- **Answer post-processing** – cleaners, refusal / "as an AI" validators and
  rule-based or model-based judges map free text to answer options.
- **Callbacks** – stream answers to JSONL / CSV or the console as they are generated.
- **Configurator app** – a Streamlit GUI (`rup-configurator`) to build configs, with
  optional LLM-assisted questionnaire import from PDF.

## Quick start

```bash
pip install git+https://github.com/julianschelb/rupsycho.git
```

```python
import rupsycho as rup

experiment = rup.experiment_from_file("examples/data/bfi_demo_config.json")
experiment.run()

answers = experiment.get_answers_as_dataframe()
```

Continue with [Getting Started](getting-started.md), or read about the
[core concepts](concepts.md).

!!! warning "Under active development"
    The API may change between releases. See the [changelog](changelog.md).
