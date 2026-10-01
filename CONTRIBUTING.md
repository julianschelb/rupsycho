# Contributing

Thanks for your interest in R.U.Psycho!

## Getting started

```bash
git clone https://github.com/julianschelb/rupsycho.git
cd rupsycho
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional, smaller download
pip install -e ".[dev,configurator]"
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

The [development guide](https://julianschelb.github.io/rupsycho/development/) lists all `poe`
tasks and the project layout.

## Workflow

1. Create a branch from `main`.
2. Make your change, with tests for behavioural changes and docs where relevant.
3. Run `poe check` (lint, format check, mypy, offline tests). If you touched the
   documentation or docstrings, also run `poe docs-build`.
4. Open a pull request. CI must pass.

Tests run offline against a fake LLM. Tests that download real models are marked
`integration`; run them with `poe test-integration`.

Docstrings use the [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
(`Args:`, `Returns:`, `Raises:`), because the API reference is generated from them.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/); releases and the
changelog are generated from them:

```text
feat: add Anthropic model configuration
fix: pass the time argument to all callbacks
docs: explain cumulative mode
```

Use `feat!:` or a `BREAKING CHANGE:` footer for incompatible changes.

## Reporting bugs

Please open an issue with the configuration (without API keys), the versions of
`rupsycho` and Python (`pip show rupsycho` and `python --version`), and the full traceback.
