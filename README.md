# R.U.Psycho

[![CI](https://github.com/julianschelb/rupsycho/actions/workflows/ci.yml/badge.svg)](https://github.com/julianschelb/rupsycho/actions/workflows/ci.yml)
[![Docs](https://github.com/julianschelb/rupsycho/actions/workflows/docs.yml/badge.svg)](https://julianschelb.github.io/rupsycho/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2503.10229-b31b1b.svg)](https://arxiv.org/abs/2503.10229)

**R.U.Psycho** (*Robust Unified Psychometric Testing of Language Models*) is a framework for
designing and running **robust and reproducible psychometric experiments on generative
language models**, with limited coding expertise required.

Documentation: <https://julianschelb.github.io/rupsycho/>

Paper: [*R.U.Psycho? Robust Unified Psychometric Testing of Language Models*](https://arxiv.org/abs/2503.10229)
(Schelb, Borin, Garcia & Spitz, 2025)

## Why?

Instabilities in model outputs, sensitivity to prompt design and generation parameters, and the
sheer number of model versions make psychometric studies of language models hard to reproduce.
R.U.Psycho turns a whole study into **one declarative configuration**: the questionnaire, the
personas the model answers as, the models and their parameters, the prompt template and the
random seeds. The package then runs every *model × seed × persona × item* combination and
helps you turn the free-text answers into scorable responses.

## Features

- **Declarative experiments** in a single, shareable JSON file
- **Many back-ends:** local & remote Hugging Face, Ollama, OpenAI, Google, DeepSeek, or any LangChain chat model or LLM
- **Personas** via demographic profile templates
- **Post-processing:** cleaners, refusal / "as an AI" validators, rule- and model-based judges
- **Callbacks** that stream answers to JSONL / CSV / console while the experiment runs
- **Configurator app** (`rup-configurator`) with LLM-assisted questionnaire import from PDF or pasted text

## Installation

```bash
pip install git+https://github.com/julianschelb/rupsycho.git

# with the Streamlit configurator app
pip install "rupsycho[configurator] @ git+https://github.com/julianschelb/rupsycho.git"
```

Requires Python 3.10 – 3.13. The package depends on PyTorch and Transformers, so the installation
is large. For a CPU-only environment, install PyTorch from the CPU index first
(`pip install torch --index-url https://download.pytorch.org/whl/cpu`).

## Quick start

The example configurations live in the repository and are not part of the installed package.
Download the demo configuration (or clone the repository and use
`examples/data/bfi_demo_config.json`):

```bash
curl -O https://raw.githubusercontent.com/julianschelb/rupsycho/main/examples/data/bfi_demo_config.json
```

```python
import rupsycho as rup

# Load a configuration (questionnaire, personas, models, prompt, seeds)
experiment = rup.experiment_from_file("bfi_demo_config.json")

# Inspect exactly what the model will see
experiment.print_assembled_prompt(item_idx=0, persona_idx=0)

# Run every model × seed × persona × item combination
experiment.run()

# One row per generated answer
answers = experiment.get_answers_as_dataframe()
```

The demo configuration runs `HuggingFaceTB/SmolLM-1.7b-Instruct` on the CPU. The model (several GB)
is downloaded from the Hugging Face Hub when `run()` first needs it.

See the [Getting Started guide](https://julianschelb.github.io/rupsycho/getting-started/) and the
notebooks in [`examples/`](examples) for more.

## Using the Configurator App

The configurator is a Streamlit app for building the personas and the questionnaire of a
configuration without writing JSON, including an LLM-assisted import of a questionnaire from a PDF
or from pasted text. Install the `configurator` extra (see above) and start it with

```bash
rup-configurator
```

The [Configurator App guide](https://julianschelb.github.io/rupsycho/tutorials/configurator/)
describes the tabs and workflows.

## Development

```bash
pip install -e ".[dev,configurator]"
pre-commit install --hook-type pre-commit --hook-type commit-msg

poe check              # lint + format check + mypy + offline tests
poe docs               # serve the docs locally
poe test-integration   # tests that download real models
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[development guide](https://julianschelb.github.io/rupsycho/development/).

## Citation

If you use R.U.Psycho, please cite the paper (see also [CITATION.cff](CITATION.cff)):

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

[MIT](LICENSE)
