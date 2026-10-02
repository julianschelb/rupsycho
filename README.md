# R.U.Psycho

[![CI](https://github.com/julianschelb/rupsycho/actions/workflows/ci.yml/badge.svg)](https://github.com/julianschelb/rupsycho/actions/workflows/ci.yml)
[![Docs](https://github.com/julianschelb/rupsycho/actions/workflows/docs.yml/badge.svg)](https://julianschelb.github.io/rupsycho/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://github.com/julianschelb/rupsycho/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/julianschelb/rupsycho/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2503.10229-b31b1b.svg)](https://arxiv.org/abs/2503.10229)

**R.U.Psycho** (*Robust Unified Psychometric Testing of Language Models*) is a framework for
designing and running **robust and reproducible psychometric experiments on generative language
models**, with limited coding expertise required.

Documentation: <https://julianschelb.github.io/rupsycho/>

Paper: [*R.U.Psycho? Robust Unified Psychometric Testing of Language Models*](https://arxiv.org/abs/2503.10229)
(Schelb, Borin, Garcia & Spitz, 2025)

## Why?

Instabilities in model outputs, sensitivity to prompt design and generation parameters, and the
sheer number of model versions make psychometric studies of language models hard to reproduce.
R.U.Psycho turns a whole study into **one declarative configuration**: the questionnaire, the
personas the model answers as, the models and their parameters, the prompt template and the
random seeds. The package runs every *model × seed × persona × item* combination, turns the
free-text answers into scorable responses and computes item and scale scores.

## Features

- **Declarative experiments** in a single, shareable JSON file (secrets are never exported)
- **Seeds that work**: every back-end receives the seed in the way it supports (see
  [Reproducibility](https://julianschelb.github.io/rupsycho/tutorials/reproducibility/))
- **Many back-ends:** local & remote Hugging Face, Ollama, OpenAI, Google, DeepSeek, any LangChain runnable
- **Robust runs:** a `RunSummary` instead of silent failures, an error policy, opt-in concurrency for
  API models with a deterministic result order, re-runnable experiments
- **Post-processing and scoring:** cleaners, refusal / "as an AI" validators, rule- and model-based
  judges, then weights, reverse-keyed items and per-trait scale scores
- **Command line:** `rupsycho run | validate | prompt | postprocess | examples | configurator`
- **Configurator app** with LLM-assisted questionnaire import from PDF
- **Light core:** `import rupsycho` takes milliseconds; model back-ends are optional extras

## Installation

```bash
pip install git+https://github.com/julianschelb/rupsycho.git          # core: configs, run loop, parsers, scoring
pip install "rupsycho[huggingface] @ git+https://github.com/julianschelb/rupsycho.git"   # + local Hugging Face models
```

Extras: `huggingface` (PyTorch, Transformers), `openai`, `ollama`, `google`, `deepseek`, `models`
(all back-ends), `configurator` (Streamlit app), `notebook`, `quantization`, `all`. Using a
back-end whose extra is missing raises an error that names the extra to install.

Requires Python 3.10 or newer (tested on 3.10 – 3.14).

## Quick start

```python
import rupsycho as rup

# A bundled example: five Big Five items answered by two personas
experiment = rup.load_example_experiment("bfi", seeds=[1, 2, 3])
experiment.print_assembled_prompt(item_idx=1, persona_idx=0)   # exactly what the model sees

summary = experiment.run()                      # needs the huggingface extra for the example model
print(summary)                                  # e.g. "30 model calls in 41.2s"
answers = experiment.get_answers_as_dataframe() # one row per model x seed x persona x item
```

Use your own model instead of the example's:

```python
from langchain_openai import ChatOpenAI          # pip install "rupsycho[openai]"

experiment = rup.load_example_experiment("bfi", models={}, seeds=[1, 2, 3])
experiment.add_model(ChatOpenAI(model="gpt-4o-mini"), identifier="gpt-4o-mini")
experiment.run(max_concurrency=8)                # API calls run in parallel, order is preserved
```

From the shell:

```bash
rupsycho examples copy bfi bfi.json
rupsycho validate bfi.json
rupsycho run bfi.json -o results.csv --seeds 1 2 3
rupsycho postprocess bfi.json results.csv -o processed.csv
```

Then score the judged answers:

```python
import pandas as pd
from rupsycho import scoring

processed = pd.read_csv("processed.csv")
scored = scoring.score_answers(processed, experiment)
print(scoring.scale_scores(scored))              # mean score per model, persona, seed and trait
```

See the [Getting Started guide](https://julianschelb.github.io/rupsycho/getting-started/), the
[tutorials](https://julianschelb.github.io/rupsycho/tutorials/running-experiments/) and the
notebooks in [`examples/`](https://github.com/julianschelb/rupsycho/tree/main/examples).

## Development

```bash
pip install -e ".[dev,models,configurator]"
pre-commit install --hook-type pre-commit --hook-type commit-msg

poe check              # lint + format check + mypy + offline tests
poe docs               # serve the docs locally
poe test-integration   # tests that download real models
```

See [CONTRIBUTING.md](https://github.com/julianschelb/rupsycho/blob/main/CONTRIBUTING.md) and the
[development guide](https://julianschelb.github.io/rupsycho/development/).

## Citation

If you use R.U.Psycho, please cite the paper (see also [CITATION.cff](https://github.com/julianschelb/rupsycho/blob/main/CITATION.cff)):

```bibtex
@misc{schelb2025rupsycho,
  title         = {R.U.Psycho? Robust Unified Psychometric Testing of Language Models},
  author        = {Schelb, Julian and Borin, Orr and Garcia, David and Spitz, Andreas},
  year          = {2025},
  eprint        = {2503.10229},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2503.10229}
}
```

## License

[MIT](https://github.com/julianschelb/rupsycho/blob/main/LICENSE)
