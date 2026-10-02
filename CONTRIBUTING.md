# Contributing

Thanks for your interest in R.U.Psycho! This project follows a lightweight GitHub flow: a
branch, a pull request, green CI. By taking part you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Bug reports and feature requests** through the
  [issue templates](https://github.com/julianschelb/rupsycho/issues/new/choose). Include the
  configuration (without API keys), the versions of `rupsycho` and Python, and the full
  traceback.
- **Model back-ends, parsers and callbacks** (see [How to add ...](#how-to-add-a-model-back-end)).
- **Documentation**: the pages in `docs/` and the docstrings, see [DOCUMENTATION.md](DOCUMENTATION.md).
- **Tests**, examples and questionnaires.
- **Security issues** are reported privately, see [SECURITY.md](SECURITY.md).

## Development setup

```bash
git clone https://github.com/julianschelb/rupsycho.git
cd rupsycho
python -m venv .venv && source .venv/bin/activate

# CPU-only PyTorch avoids a multi-GB CUDA download (optional)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev,models,configurator]"
pre-commit install        # installs the pre-commit and the commit-msg hook
poe check                 # lint, format check, mypy and the offline tests
```

`pip install -e ".[dev]"` is enough for most changes; tests that need a model back-end skip
themselves when it is not installed. The everyday commands are `poe` tasks:

| Command                | What it does                                               |
| ---------------------- | ---------------------------------------------------------- |
| `poe test`             | Offline test suite (`poe test-cov` adds a coverage report) |
| `poe lint`             | Ruff lint                                                  |
| `poe format`           | Ruff format (`poe format-check` only checks)               |
| `poe typecheck`        | mypy on `src/rupsycho/`                                    |
| `poe docs-build`       | Strict documentation build (`poe docs` serves it locally)  |
| `poe test-integration` | Tests that download real models                            |
| `poe check`            | lint, format check, mypy and tests, like CI                |

[DEVELOPMENT.md](DEVELOPMENT.md) lists all tasks, the extras, the test markers and the CI
workflows.

## Pull requests

1. Create a branch from `main`.
2. Make your change, with tests for behavioural changes and documentation where relevant.
3. Run `poe check`, and `poe docs-build` if you touched documentation or docstrings.
4. Open a pull request against `main`. CI must pass before it is merged.

Checklist (it is also in the pull request template):

- [ ] The PR title and commit messages follow [Conventional Commits](#commit-messages)
- [ ] `poe check` passes
- [ ] Tests are added or updated, run offline, and skip cleanly without optional extras
- [ ] Public objects have Google-style docstrings and the docs are updated
- [ ] No API keys, tokens or personal data in configurations, notebooks or logs

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/) are enforced by a `commit-msg`
hook. Releases and the changelog are generated from them, so they matter:

```text
feat: add an Anthropic model configuration
fix: pass the time argument to all callbacks
docs: explain cumulative mode
```

| Type                                                                  | Effect on the version             |
| --------------------------------------------------------------------- | --------------------------------- |
| `fix`, `perf`                                                         | patch release                     |
| `feat`                                                                | minor release                     |
| `feat!`, `fix!` or a `BREAKING CHANGE:` footer                        | major release (minor below 1.0.0) |
| `docs`, `style`, `refactor`, `test`, `build`, `ci`, `chore`, `revert` | no release                        |

When a pull request is squash-merged, its title becomes the commit message: write it in this
format too. Do not edit `CHANGELOG.md` or the version numbers; the release workflow does it.

## Code style

- Python 3.10 or newer, `from __future__ import annotations`, full type hints. The package
  ships `py.typed` and `poe typecheck` (mypy) must stay clean.
- Google-style docstrings with `Args`, `Returns`, `Raises` and `Example` sections for every
  public object. The API reference is generated from them.
- [Ruff](https://docs.astral.sh/ruff/) formats and lints (line length 100): `poe format` and
  `poe lint`.
- Use `logging` in the library. `print()` is for output that is the feature (CLI output,
  callbacks).
- Import optional dependencies lazily, at the point of use, with `rupsycho._compat.require`, so
  that `import rupsycho` works with the mandatory dependencies only and a missing extra gives an
  actionable error.
- Never put API keys in code, configurations, notebooks or tests.

## Tests

- Tests are **offline, deterministic and fast**: use the fake LLM fixtures (`fake_experiment`,
  `config_dict`) or the tiny offline Hugging Face model (`tiny_hf_config`) from
  `tests/conftest.py`. Do not download models.
- A test that needs a real model download is marked `@pytest.mark.integration`. It is
  deselected by default and runs with `poe test-integration` and in the weekly *Integration*
  workflow.
- A test that needs an optional dependency calls `pytest.importorskip("module")`. The `core` job
  of CI runs the whole suite with the mandatory dependencies only.
- Seed any randomness, avoid sleeps and timing assertions that depend on the machine, and keep the
  suite fast.

## How to add a model back-end

A back-end is a pydantic configuration class that knows how to build one LangChain chat model.
Take `OpenAIModelConfig` in `src/rupsycho/models/model.py` as a template.

1. **Extra.** Declare the provider package as an optional dependency in `pyproject.toml`
   (for example `anthropic = ["langchain-anthropic (>=...)"]`) and add it to the `models` and
   `all` extras.
2. **Configuration class** in `src/rupsycho/models/model.py`, added to `__all__`: a
   `type: str = Field("anthropic", ...)` discriminator, the settings (`name_or_path`,
   `api_key`, `parameters`, ...) and a documented `load_model()`. Import the provider inside
   `load_model()` with `require("langchain_anthropic", "anthropic", feature="Anthropic models")`
   and wrap construction errors in a `ValueError` (`raise ... from e`). Declare API keys and
   tokens with the `Secret` type of that module, so that they are masked in `repr` and in
   exported configurations, read them with `_reveal(...)`, and pass only the values that are
   configured so that the provider's environment variable remains the fallback.
3. **Register it** in `MODEL_CONFIG_CLASSES` in `src/rupsycho/experiment.py` (the key is the
   `type` used in JSON configurations) and add the class to the `models` annotation of
   `ExperimentDocument`.
4. **Seeding.** Models with a `seed` field are seeded automatically; for other back-ends call
   `rupsycho.seeding.register_seeder`, or document that seeds have no effect (a warning is
   emitted once per model type). See `src/rupsycho/seeding.py`.
5. **Tests.** Check that the configuration validates without the extra installed, that
   `load_model()` raises the `ImportError` naming the extra when it is missing, and, behind
   `pytest.importorskip`, that it builds the client. `tests/test_models_config.py`,
   `tests/test_seeding.py` and `tests/test_http_backends.py` show the patterns.
6. **Docs.** Describe the `type` and its settings in `docs/configuration.md` and
   `docs/tutorials/models.md`; the API page picks the class up from its docstring.

## How to add a callback

A callback receives every answer as soon as it is generated (CSV, JSONL, a database, ...).

1. Subclass `rupsycho.callbacks.Callback` and implement `save_answer(self, experiment,
   instruction_item_id, instruction_item, model_id, profile_id, random_seed, time, answer)`.
   It is called once per model call, in the order of the run, from the thread that called
   `run()`. `answer` is `None` when the call failed, and an exception raised by a callback is
   reported as a warning without stopping the experiment.
2. Put the class next to the others in `src/rupsycho/callbacks/` and export it from
   `src/rupsycho/callbacks/__init__.py` (including `__all__`). Keep the constructor cheap and
   write files as UTF-8.
3. Test it with `fake_experiment`: run the experiment with the callback in a `tmp_path` and
   read the output back.
4. Document it with a docstring example and in `docs/tutorials/callbacks.md`.
   `examples/custom_callback.py` is a worked example of a custom callback.

## How to add a parser

Parsers clean, validate or judge the free-text answers of a model. They are LangChain output
parsers (`BaseOutputParser`) and can be chained with `|`.

1. Choose the module by role: `src/rupsycho/parsers/cleaners.py` (text in, cleaner text out),
   `validators.py` (returns a dict with a `validation_status`) or `judges.py` (maps an answer
   to one of the possible answer options).
2. Subclass `BaseOutputParser`, implement `parse(self, text)` and the `_type` property, and
   keep the parser a pure function of its input where possible. Raise
   `OutputParserException` for input that cannot be parsed. Parsers that need a model download
   it lazily, with `require(...)`.
3. Parsers are star-exported by `src/rupsycho/parsers/__init__.py`: give the class a clear
   public name and leave helpers private.
4. Test with plain examples and, where the behaviour is a property of the input
   (idempotence, case or whitespace invariance), with Hypothesis. `tests/test_parser.py` and
   `tests/test_properties_parsers.py` show both styles. Tests that load a model from the Hub
   are `integration` tests.
5. Document the class with a docstring example, add it to `docs/api/parsers.md` if it needs a
   new section, and mention it in `docs/tutorials/postprocessing.md`.

## How releases happen

You never bump versions by hand. After CI succeeds on `main`, the *Release* workflow runs
[python-semantic-release](https://python-semantic-release.readthedocs.io/): it derives the next
version from the Conventional Commits since the last tag, updates `CHANGELOG.md`, tags
`vX.Y.Z`, creates the GitHub release and, once the maintainers enabled it, publishes to PyPI
with Trusted Publishing. The details are in [DEVELOPMENT.md](DEVELOPMENT.md#semantic-versioning-and-releases).
