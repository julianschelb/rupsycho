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

## Workflow

1. Create a branch from `main`.
2. Make your change, with tests for behavioural changes and docs where relevant.
3. Run `poe check` (lint, format check, mypy, offline tests).
4. Open a pull request. CI must pass.

Tests run offline against a fake LLM. Tests that download real models are marked
`integration`; run them with `poe test-integration`.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/); releases and the
changelog are generated from them:

```
feat: add Anthropic model configuration
fix: pass the time argument to all callbacks
docs: explain cumulative mode
```

Use `feat!:` or a `BREAKING CHANGE:` footer for incompatible changes.

## Reporting bugs

Please open an issue with the configuration (without API keys), the versions of
`rupsycho` and Python, and the full traceback.
