# CHANGELOG

<!-- version list -->

## Unreleased

Repository restructuring and code clean-up.

### Features

- Documentation site (MkDocs Material, deployed to GitHub Pages) replacing the Sphinx build.
- CI (lint, type check, tests on Python 3.10-3.13, build), docs deployment and
  semantic-release workflows.
- Offline test suite using a fake LLM; real-model tests are marked `integration`.
- `rupsycho.callbacks` now exports `Callback`, `JSONLCallback`, `CSVCallback`,
  `PrintCallback` and `PrintTableCallback`.
- Optional extras: `configurator`, `notebook`, `quantization`, `all`.

### Bug Fixes

- `JSONLCallback`, `PrintCallback` and `PrintTableCallback` did not accept the `time`
  argument that the run loop passes, so their answers were silently dropped.
- `ConfigQuestionnaire.from_file` referenced an undefined class; the configurator used
  fragile `utils` star imports.
- `ExperimentParameters.seeds` drew its random default once at import time; it is now
  drawn per experiment.
- Duplicate `NormalPromptTemplateConfig` and `json_loader` definitions removed.
- Exceptions are chained (`raise ... from e`) and warnings carry a `stacklevel`.

### Performance

- Prompt inputs are built once per experiment instead of once per seed and call; the chain
  is built once per model and seed.
- CUDA cache is only cleared when CUDA is available.

### Refactoring

- Packaging moved to PEP 621 metadata; unused dependencies (`nltk`, `protobuf`,
  `langchain`, `langchain-community`, `ipywidgets`, `bitsandbytes`) are now optional or
  removed, and obsolete pins relaxed so the package installs on current Python versions.
- Migrated pydantic v1 idioms (`class Config`, `@validator`) to v2.
- Removed generated artefacts (`dist/`, `docs/_build/`, `poetry.lock`, `requirements.txt`),
  placeholder modules and the verbatim duplicate `rupsycho/parser.py` (now a thin alias).
- Type hints modernised; code is ruff- and mypy-clean.
