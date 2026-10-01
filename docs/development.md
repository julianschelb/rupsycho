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

The `dev` extra contains everything needed for the tasks below (pytest, ruff, mypy, MkDocs,
poethepoet, pre-commit, build tools and python-semantic-release). The smaller `test` and
`docs` extras install only the test or documentation tooling.

## Tasks

All tasks are defined with [poethepoet](https://poethepoet.natn.io/) in `pyproject.toml`:

| Command                  | What it does                                            |
| ------------------------ | ------------------------------------------------------- |
| `poe test`               | Offline tests (a fake LLM; no model downloads)          |
| `poe test-cov`           | Tests with coverage                                     |
| `poe test-integration`   | Tests that download and run real models                 |
| `poe lint`               | Ruff lint                                               |
| `poe format`             | Ruff format (rewrites files)                            |
| `poe format-check`       | Ruff format check (does not change files)               |
| `poe typecheck`          | mypy                                                    |
| `poe check`              | lint + format check + typecheck + tests                 |
| `poe docs`               | Serve the documentation locally                         |
| `poe docs-build`         | Build the documentation with `--strict`                 |

## Project layout

```text
src/rupsycho/                   the library
  __init__.py                   experiment_from_file / _from_dict / ... loader functions
  experiment.py                 ExperimentDocument (pydantic + LangChain)
  experiment_collection.py      ExperimentCollection
  mixins/                       run loop, model / persona / prompt management, export
  models/                       pydantic data models of the configuration
  parsers/                      cleaners, validators, judges
  callbacks/                    answer-saving callbacks
  reader.py                     JSON → ExperimentDocument loader
  postprocessing.py             PostprocessingPipeline
  prompts.py                    default prompt templates
  utils/                        small helpers (JSON files, progress bar)
src/rupsycho_configurator/      Streamlit configurator app
tests/                          offline tests (+ integration marker)
docs/                           MkDocs sources (configured in mkdocs.yml)
examples/                       notebooks, configurations, user-study material
.github/workflows/              CI, documentation and release workflows
```

## Testing

Tests never need network access by default: experiments run against LangChain's
`FakeListLLM` (see the fixtures in `tests/conftest.py`). Tests that load real Hugging Face
models are marked `integration` and are deselected unless you run `poe test-integration`
(or `pytest -m integration`).

## Commits and releases

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`, `build:`, `chore:` …), enforced by
pre-commit. After CI passes on `main`, the *Release* workflow uses
[python-semantic-release](https://python-semantic-release.readthedocs.io/) to bump the
version, update `CHANGELOG.md`, tag the release and create the GitHub release. While the
version is below 1.0, breaking changes only bump the minor version.

## Documentation

The documentation is built with MkDocs Material and mkdocstrings and is deployed to GitHub
Pages by the *Docs* workflow on every push to `main`. The workflow runs `mkdocs gh-deploy`,
which publishes to the `gh-pages` branch, so the repository's Pages setting has to serve that
branch. Pull requests only build the site, with `--strict`, so broken references fail the check.

Recent Material for MkDocs releases print a notice about MkDocs 2.0 in red at the start of
every build. It is informational and does not affect the result (set `NO_MKDOCS_2_WARNING=1`
to hide it). The build fails on real problems: unresolved cross-references, links to missing
pages or anchors, and pages that are missing from the `nav` of `mkdocs.yml`.

The API reference is generated from the docstrings, which are expected in
[Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings):
sections such as `Args:`, `Returns:` and `Raises:` with their entries indented below the
heading. Other styles (reST `:param x:` fields, NumPy `Parameters` / `----------`
headings) are not parsed into structured sections and are rendered as plain text.
