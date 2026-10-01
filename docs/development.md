# Development

## Setup

```bash
git clone https://github.com/julianschelb/rupsycho.git
cd rupsycho
python -m venv .venv && source .venv/bin/activate

# CPU-only PyTorch avoids a multi-GB CUDA download (optional)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev,configurator]"
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

## Tasks

All tasks are defined with [poethepoet](https://poethepoet.natn.io/):

| Command                  | What it does                                            |
| ------------------------ | ------------------------------------------------------- |
| `poe test`               | Offline tests (a fake LLM; no model downloads)          |
| `poe test-cov`           | Tests with coverage                                     |
| `poe test-integration`   | Tests that download and run real models                 |
| `poe lint` / `poe format`| Ruff lint / format                                      |
| `poe typecheck`          | mypy                                                    |
| `poe check`              | lint + format check + typecheck + tests                 |
| `poe docs`               | Serve the documentation locally                         |
| `poe docs-build`         | Build the documentation with `--strict`                 |

## Project layout

```
src/rupsycho/                   the library
  experiment.py                 ExperimentDocument (pydantic + LangChain)
  mixins/                       run loop, model / persona / prompt management, export
  models/                       pydantic data models of the configuration
  parsers/                      cleaners, validators, judges
  callbacks/                    answer-saving callbacks
  reader.py                     JSON → ExperimentDocument loader
  postprocessing.py             PostprocessingPipeline
src/rupsycho_configurator/      Streamlit configurator app
tests/                          offline tests (+ integration marker)
docs/                           MkDocs sources
examples/                       notebooks, configurations, user-study material
```

## Testing

Tests never need network access by default: experiments run against LangChain's
`FakeListLLM`. Tests that load real Hugging Face models are marked `integration` and are
deselected unless you run `poe test-integration`.

## Commits and releases

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`, `build:`, `chore:` …), enforced by
pre-commit. After CI passes on `main`, the *Release* workflow uses
[python-semantic-release](https://python-semantic-release.readthedocs.io/) to bump the
version, update `CHANGELOG.md`, tag the release and create the GitHub release.

## Documentation

The documentation is built with MkDocs Material and mkdocstrings (Google-style
docstrings) and is deployed to GitHub Pages by the *Docs* workflow on every push to
`main`.
