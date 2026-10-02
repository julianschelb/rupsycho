# Development Guide

This document covers setup and the day-to-day workflows for working on R.U.Psycho. How to
propose a change is described in
[CONTRIBUTING.md](https://github.com/julianschelb/rupsycho/blob/main/CONTRIBUTING.md); how the
documentation site is built is described in
[DOCUMENTATION.md](https://github.com/julianschelb/rupsycho/blob/main/DOCUMENTATION.md).

## Prerequisites

- Python 3.10 or newer (CI covers 3.10 to 3.14)
- git and pip (or [uv](https://docs.astral.sh/uv/))
- No GPU is needed: the test suite runs on the CPU and downloads no models

## Installation

```bash
git clone https://github.com/julianschelb/rupsycho.git
cd rupsycho
python -m venv .venv && source .venv/bin/activate

# CPU-only PyTorch avoids a multi-GB CUDA download (optional)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Editable install with the tools and every model back-end the tests can use
pip install -e ".[dev,models,configurator]"

# Git hooks: ruff, hygiene checks, mypy and the commit message check
pre-commit install
```

`pip install -e ".[dev]"` is enough for most changes: tests that need a model back-end skip
themselves when it is missing. The extras are defined in `pyproject.toml`:

| Extra          | Installs                                                                                                  | Needed for                                                                      |
| -------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `huggingface`  | torch, transformers, accelerate, sentencepiece, langchain-huggingface                                     | local and remote Hugging Face models, model-based parsers, seeding local models |
| `openai`       | langchain-openai                                                                                          | OpenAI and OpenAI-compatible chat models                                        |
| `ollama`       | langchain-ollama                                                                                          | Ollama                                                                          |
| `google`       | langchain-google-genai                                                                                    | Google Gemini                                                                   |
| `deepseek`     | langchain-deepseek                                                                                        | DeepSeek                                                                        |
| `models`       | every model back-end above                                                                                | all providers at once                                                           |
| `configurator` | streamlit, pypdf, openai, langchain-openai                                                                | the `rup-configurator` app                                                      |
| `notebook`     | ipywidgets                                                                                                | progress bars and widgets in notebooks                                          |
| `quantization` | bitsandbytes                                                                                              | 4-bit and 8-bit model loading                                                   |
| `all`          | models, configurator, notebook, bitsandbytes (not on macOS)                                               | everything a user may need                                                      |
| `test`         | pytest, pytest-cov, hypothesis                                                                            | running the tests                                                               |
| `docs`         | mkdocs (1.x), mkdocs-material, mkdocstrings                                                               | building the documentation                                                      |
| `dev`          | `test` and `docs` plus ruff, mypy, stubs, poethepoet, pre-commit, build, semantic-release, notebook tools | developing the package                                                          |

## Tasks

All recurring tasks are defined with [poethepoet](https://poethepoet.natn.io/) in
`pyproject.toml` (`[tool.poe.tasks]`):

| Command                | Runs                                     | What it does                                          |
| ---------------------- | ---------------------------------------- | ----------------------------------------------------- |
| `poe test`             | `pytest`                                 | Offline test suite (integration tests are deselected) |
| `poe test-cov`         | `pytest --cov --cov-report=term-missing` | Tests with a coverage report                          |
| `poe test-integration` | `pytest -m integration`                  | Tests that download and run real models               |
| `poe lint`             | `ruff check src/ tests/`                 | Lint                                                  |
| `poe format`           | `ruff format src/ tests/`                | Format the code                                       |
| `poe format-check`     | `ruff format --check src/ tests/`        | Check the formatting without changing files           |
| `poe typecheck`        | `mypy src/rupsycho/`                     | Static type check of the package                      |
| `poe docs`             | `mkdocs serve`                           | Serve the documentation locally with live reload      |
| `poe docs-build`       | `mkdocs build --strict`                  | Build the documentation; warnings are errors          |
| `poe docs-deploy`      | `mkdocs gh-deploy --force`               | Publish the documentation to GitHub Pages by hand     |
| `poe bench`            | `python benchmarks/bench_run_loop.py`    | Benchmark the run loop                                |
| `poe check`            | lint, format-check, typecheck, test      | Everything CI checks, in one command                  |

Run the tasks from the activated virtual environment. poe uses Poetry as the task executor when
it finds a `poetry` executable and the project has a `[tool.poetry]` table; if a task then
fails with "poetry: command not found", run it with the plain executor: `poe -e simple check`.

## Testing

```bash
poe test                    # the offline suite
poe test-cov                # with a coverage report
pytest tests/test_seeding.py -k reproducible -x -q   # a single test, stop at the first failure
```

### Offline by design

Tests must be offline, deterministic and fast. Experiments run against LangChain's fake
models (`FakeListLLM` and friends); the HTTP back-ends are exercised against a local fake
server; local Hugging Face models are exercised with the tiny model described below. CI sets
`HF_HUB_OFFLINE=1` for the suite, so a test that tries to reach the Hugging Face Hub fails
there. Such a test belongs to the `integration` marker.

The main shared fixtures live in `tests/conftest.py`:

| Fixture           | Provides                                                                  |
| ----------------- | ------------------------------------------------------------------------- |
| `config_dict`     | The BFI test configuration as a plain dict, without any real model        |
| `fake_experiment` | A fresh BFI experiment wired to a fake LLM that always answers `"3"`      |
| `tiny_hf_dir`     | Directory of the tiny offline Hugging Face model (built once per session) |
| `tiny_hf_config`  | A `local_huggingface` model configuration that points at `tiny_hf_dir`    |

### Markers

| Marker        | Meaning                                                  | Selected by default |
| ------------- | -------------------------------------------------------- | ------------------- |
| `integration` | Downloads and runs real models (slow, needs the network) | no                  |

`addopts` in `pyproject.toml` contains `-m 'not integration'`; `poe test-integration` (or
`pytest -m integration`) overrides it.

### Integration tests

Integration tests download small models from the Hugging Face Hub (a small language model for
an end-to-end run, classification models for the model-based validator and judge) and run them
on the CPU. They need the `huggingface` extra and network access, and the first run downloads
the weights into `~/.cache/huggingface`:

```bash
poe test-integration                              # all of them
pytest -m integration -k judge                    # a subset
```

They are not part of CI. The *Integration* workflow runs them by hand (*Actions, Integration,
Run workflow*, optionally with a `-k` expression) and every Monday, with the Hugging Face cache
restored between runs. Set the optional repository secret `HF_TOKEN` to avoid rate limits.

### Property-based tests

Some tests, for example `tests/test_properties_parsers.py`, are property-based: they use
[Hypothesis](https://hypothesis.readthedocs.io/) and compare the library with plain reference
implementations (`tests/helpers.py`). Locally Hypothesis explores new inputs on every run and
prints how to reproduce a failure. CI runs the suite with a fixed seed
(`PYTEST_ADDOPTS=--hypothesis-seed=0`), so a CI run is reproducible and never red by chance; to
replay it locally use `pytest --hypothesis-seed=0`. The example database `.hypothesis/` is
git-ignored.

### Optional dependencies in tests

The package must work with its mandatory dependencies only, and the `core` job of CI runs the
whole suite in such an environment. A test that needs an extra has to skip itself instead of
failing:

```python
def test_openai_config_builds_a_chat_model():
    pytest.importorskip("langchain_openai")
    ...
```

To reproduce the `core` job locally:

```bash
python -m venv /tmp/core-venv
/tmp/core-venv/bin/pip install -e . pytest hypothesis
/tmp/core-venv/bin/python -m pytest -q -m 'not integration'
```

### The offline tiny model

`tests/tiny_hf.py` builds a randomly initialised one-layer GPT-2 with a character-level
tokenizer and a chat template. It produces gibberish, but it runs through the real
Transformers pipeline, sampling and tokenizer code, so loading, seeding and the run loop of
local models can be tested without a download. The `tiny_hf_dir` fixture builds it once per
session (it skips when torch, transformers or langchain-huggingface are missing) and
`tiny_hf_config` turns it into a model configuration:

```python
import copy

import rupsycho as rup


def test_local_model_runs_end_to_end(config_dict, tiny_hf_config):
    config = copy.deepcopy(config_dict)
    config["models"] = {"tiny": tiny_hf_config}
    experiment = rup.experiment_from_dict(config)

    summary = experiment.run(show_progress=False)

    assert summary.n_failed == 0
```

### Coverage

CI measures the coverage of `src/rupsycho` (branch coverage, see `[tool.coverage]` in
`pyproject.toml`) and fails below a threshold (`--cov-fail-under` in
`.github/workflows/ci.yml`). Look for untested lines locally with `poe test-cov`.

## Linting, formatting and type checking

[Ruff](https://docs.astral.sh/ruff/) lints and formats `src/` and `tests/` (line length 100,
rule sets `E W F I UP B SIM`). [mypy](https://mypy.readthedocs.io/) checks `src/rupsycho/`
with the pydantic plugin; the package ships `py.typed`, so public signatures need type hints.

```bash
poe lint           # ruff check
poe format         # ruff format
poe format-check   # ruff format --check
poe typecheck      # mypy
poe check          # all of the above plus the tests
```

The ruff version is pinned in two places that must agree: `RUFF_VERSION` in
`.github/workflows/ci.yml` and the `ruff-pre-commit` `rev` in `.pre-commit-config.yaml`
(CI fails when they differ). Bump both together.

## Pre-commit hooks

```bash
pre-commit install            # installs the pre-commit and the commit-msg hook
pre-commit run --all-files    # run every hook on the whole repository
```

Run `git commit` from the activated virtual environment, because the mypy hook uses the
`mypy` of that environment.

| Hook                                                    | Checks                                                               |
| ------------------------------------------------------- | -------------------------------------------------------------------- |
| `ruff-check`, `ruff-format`                             | Lint and format `src/` and `tests/`                                  |
| `mypy`                                                  | `mypy src/rupsycho` when package files change                        |
| `trailing-whitespace`                                   | Trailing spaces (two spaces stay a line break in Markdown)           |
| `end-of-file-fixer`, `mixed-line-ending`                | Final newline, LF line endings                                       |
| `check-yaml`, `check-toml`, `check-json`                | Syntax of configuration files                                        |
| `check-merge-conflict`                                  | Leftover merge markers                                               |
| `check-case-conflict`                                   | File names that differ only by case                                  |
| `check-added-large-files`                               | Files above 1 MB                                                     |
| `detect-private-key`                                    | Private keys                                                         |
| `no-api-keys-in-configs`                                | JSON files with a real `api_key` or `huggingfacehub_api_token` value |
| `check-github-workflows`                                | Schema of the workflow files                                         |
| `check-github-workflows-require-timeout`                | Every workflow job sets `timeout-minutes`                            |
| `check-dependabot`                                      | Schema of `dependabot.yml`                                           |
| `check-github-issue-forms`, `check-github-issue-config` | Schemas of the issue forms and of the issue template config          |
| `check-citation-file-format`                            | `CITATION.cff` is valid CFF 1.2.0                                    |
| `conventional-pre-commit`                               | Commit message follows Conventional Commits (commit-msg stage)       |

## Documentation

The documentation is built with MkDocs Material and mkdocstrings from Google-style docstrings:

```bash
poe docs          # live-reload server at http://127.0.0.1:8000
poe docs-build    # strict build into site/
```

The *Docs* workflow builds it on every pull request and deploys it to GitHub Pages on every
push to `main`. See
[DOCUMENTATION.md](https://github.com/julianschelb/rupsycho/blob/main/DOCUMENTATION.md) for the
page layout and for how to document new objects.

## Continuous integration

| Workflow          | Trigger                              | Job and what it checks                                                           |
| ----------------- | ------------------------------------ | -------------------------------------------------------------------------------- |
| `ci.yml`          | push and pull request to `main`      | `lint`: ruff check and format check with the pinned ruff version                 |
|                   |                                      | `typecheck`: mypy on Python 3.12 with the model back-ends installed              |
|                   |                                      | `test`: pytest on Python 3.10 to 3.14 with a coverage gate                       |
|                   |                                      | `core`: the suite with the mandatory dependencies only (Python 3.10)             |
|                   |                                      | `build`: sdist and wheel, `twine check`, install smoke test of the wheel         |
| `docs.yml`        | push, pull request, manual           | `mkdocs build --strict`; deploys to GitHub Pages on pushes to `main`             |
| `integration.yml` | manual, weekly (Monday 04:23 UTC)    | `pytest -m integration` with a cached `~/.cache/huggingface`                     |
| `release.yml`     | after CI succeeded on `main`, manual | semantic-release (version, changelog, tag, GitHub release); optional PyPI upload |

Dependabot proposes updates of the GitHub actions and the Python dependencies once a month.
Every job has a `timeout-minutes`, and the CPU-only PyTorch wheel is installed first so that
CI does not download CUDA libraries.

## Semantic versioning and releases

Versions follow [Semantic Versioning](https://semver.org/). They are not edited by hand:
[python-semantic-release](https://python-semantic-release.readthedocs.io/) derives the next
version and the changelog from [Conventional Commits](https://www.conventionalcommits.org/).

### Commit message format

```text
<type>(<optional scope>): <description>

[optional body]

[optional footer(s)]
```

| Prefix                                                                        | Example                                        | Version bump                    |
| ----------------------------------------------------------------------------- | ---------------------------------------------- | ------------------------------- |
| `fix:`                                                                        | `fix: pass the time argument to all callbacks` | patch (1.0.0 to 1.0.1)          |
| `perf:`                                                                       | `perf: build the prompt inputs once`           | patch                           |
| `feat:`                                                                       | `feat: add an Anthropic model configuration`   | minor (1.0.0 to 1.1.0)          |
| `feat!:`, `fix!:` or a `BREAKING CHANGE:` footer                              | `feat!: rename the seeds parameter`            | major (minor while below 1.0.0) |
| `docs:`, `style:`, `refactor:`, `test:`, `build:`, `ci:`, `chore:`, `revert:` | `docs: explain cumulative mode`                | no release                      |

The `conventional-pre-commit` hook validates commit messages (`pre-commit install` sets it
up). When a pull request is squash-merged, its title becomes the commit message, so the title
has to follow the same format.

### How releases work

1. A pull request is merged into `main`.
2. The *CI* workflow runs on the push to `main`.
3. If it succeeds, the *Release* workflow starts (only for runs of CI that were triggered by a
   push).
4. `semantic-release version` analyses the commits since the last tag. If there are releasable
   commits it bumps the version in `pyproject.toml` and `src/rupsycho/__init__.py`, updates
   `CHANGELOG.md`, commits `chore(release): vX.Y.Z [skip ci]`, tags `vX.Y.Z`, builds the
   distributions and creates a GitHub release; `semantic-release publish` attaches the files to it.
5. If the repository variable `PUBLISH_PYPI` is `true`, the `pypi` job uploads the distributions
   to PyPI with [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC), so no
   PyPI token is stored anywhere.

The release commit is pushed to `main` by `github-actions[bot]`. If branch protection requires
pull requests, allow that actor to bypass the rule, otherwise the push fails.

### Setting up the PyPI upload (once)

1. On [pypi.org](https://pypi.org/manage/account/publishing/), open *Your account, Publishing*
   and add a trusted publisher for the project `rupsycho` (a *pending* publisher if the project
   does not exist yet): owner `julianschelb`, repository `rupsycho`, workflow `release.yml`,
   environment `pypi`.
2. On GitHub, create the environment `pypi` (*Settings, Environments*). Optionally require a
   review before every upload and allow deployments from `main` only; the upload then waits
   until you approve the run under *Actions, Review deployments*. The GitHub release and the tag
   exist already at that point.
3. On GitHub, add the repository variable `PUBLISH_PYPI` with the value `true`
   (*Settings, Secrets and variables, Actions, Variables*). Without it the *Release* workflow
   still tags and publishes the GitHub release, and skips PyPI.

### Publishing an already tagged version

`semantic-release` only publishes versions it creates itself. To publish a version that is
tagged already, for example the initial `v1.0.0` once the trusted publisher is registered,
run the workflow by hand: *Actions, Release, Run workflow*, select the tag (for example
`v1.0.0`) under *Use workflow from*, and tick **publish_current**. The workflow builds the
checked-out commit, verifies that it carries the tag `vX.Y.Z` matching the version in
`pyproject.toml`, and publishes it through the `pypi` environment. The upload needs the setup
above, including the `PUBLISH_PYPI` variable.

### Previewing the next version

```bash
semantic-release version --print    # the next version, without changing anything
```

A warning about a missing token can be ignored for this preview.

## Project layout

```text
src/rupsycho/                   the library
  __init__.py                   curated public API with lazily imported sub-packages
  cli.py, __main__.py           the `rupsycho` command (also `python -m rupsycho`)
  experiment.py                 ExperimentDocument (pydantic + LangChain)
  experiment_collection.py      ExperimentCollection
  mixins/                       run loop, model / persona / prompt management, export
  models/                       pydantic data models: model configurations, prompts, questionnaire
  parsers/                      cleaners, validators, judges (LangChain output parsers)
  callbacks/                    answer-saving callbacks
  postprocessing.py             PostprocessingPipeline
  scoring.py                    item and scale scores from judged answers
  reader.py                     JSON to ExperimentDocument loading
  seeding.py                    seeding of model calls per back-end
  datasets.py, data/            bundled example experiments
  _compat.py                    require(): actionable errors for missing extras
src/rupsycho_configurator/      Streamlit configurator app (rup-configurator)
tests/                          offline tests; tiny_hf.py builds the offline Hugging Face model
benchmarks/                     reproducible benchmarks, see its README (poe bench runs bench_run_loop.py)
examples/                       notebooks, configurations, user-study material
docs/                           MkDocs sources
```

## Quick reference

| Task                        | Command                            |
| --------------------------- | ---------------------------------- |
| Run the tests               | `poe test`                         |
| Run the tests with coverage | `poe test-cov`                     |
| Run the integration tests   | `poe test-integration`             |
| Lint                        | `poe lint`                         |
| Format                      | `poe format`                       |
| Type check                  | `poe typecheck`                    |
| All checks                  | `poe check`                        |
| All pre-commit hooks        | `pre-commit run --all-files`       |
| Serve the docs              | `poe docs`                         |
| Build the docs              | `poe docs-build`                   |
| Preview the next version    | `semantic-release version --print` |
