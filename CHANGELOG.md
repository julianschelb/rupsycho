# CHANGELOG

<!-- version list -->

## Unreleased

Repository restructuring and code clean-up.

### Features

- Documentation site (MkDocs Material with mkdocstrings, deployed to GitHub Pages) replacing the
  Sphinx build; README, CONTRIBUTING, SECURITY and CITATION files added or rewritten.
- CI (lint, format check, type check, tests on Python 3.10-3.13, build), documentation
  deployment and semantic-release workflows, plus pre-commit hooks, issue and pull request
  templates and Dependabot.
- Offline test suite using a fake LLM; tests that download real models are marked `integration`
  and deselected by default.
- `rupsycho.callbacks` now exports `Callback`, `JSONLCallback`, `CSVCallback`, `PrintCallback`
  and `PrintTableCallback`.
- Optional extras: `configurator`, `notebook`, `quantization`, `all` (and `test`, `docs`,
  `dev` for contributors).

### Bug Fixes

- `JSONLCallback`, `PrintCallback` and `PrintTableCallback` did not accept the `time`
  argument that the run loop passes, so their answers were dropped (the error only surfaced as
  a warning).
- Cumulative mode no longer crashes when a model call fails (the previous answer was
  concatenated to the prompt without converting it to a string).
- `ConfigQuestionnaire.from_file` referenced an undefined class; the configurator and its
  helper modules used fragile top-level `utils` imports and now import from the
  `rupsycho_configurator` package.
- `ExperimentParameters.seeds` drew its random default once at import time; it is now
  drawn per experiment.
- Duplicate `NormalPromptTemplateConfig` and `json_loader` definitions removed.
- Exceptions are chained (`raise ... from e`) and warnings carry a `stacklevel`.

### Performance

- Prompt inputs are built once per model instead of once per seed and prompt.
- The CUDA cache is only cleared when CUDA is available.

### Build System

- Packaging moved to PEP 621 metadata (`[project]` in `pyproject.toml`, poetry-core build
  backend); `poetry.lock`, `requirements.txt`, `pytest.ini` and the Makefiles are gone, and
  tool settings live in `pyproject.toml`. Python 3.10-3.13 is declared.
- Dependencies: `streamlit`, `pypdf` and `openai` moved from the required dependencies to the
  `configurator` extra, so `pip install rupsycho` no longer installs the configurator's
  requirements (use `rupsycho[configurator]`). `ipywidgets` and `bitsandbytes` are now optional
  (`notebook`, `quantization`), the unused `nltk`, `protobuf`, `langchain` and
  `langchain-community` are removed, the previously undeclared `numpy`, `scipy`, `tqdm` and
  `langchain-deepseek` are declared, and obsolete pins are relaxed so the package installs on
  current Python versions.

### Refactoring

- Migrated pydantic v1 idioms (`class Config`, `@validator`) to v2.
- Removed generated artefacts (`dist/`, `docs/_build/`), unused placeholder modules
  (`chains`, `models/instruction`, `models/profile`, `utils/plots`), the
  `if __name__ == "__main__"` demo blocks and the verbatim duplicate `rupsycho/parser.py`
  (now a thin alias of `rupsycho.parsers.parser`).
- The run loop was restructured; failed model calls are now reported through `logging`
  instead of `print`.
- Type hints modernised; code is ruff- and mypy-clean, with targeted `# type: ignore` comments
  where precise typing was not practical.
- `user_testing/` moved to `examples/user_testing/` and `CODEOWNERS` to `.github/`.
