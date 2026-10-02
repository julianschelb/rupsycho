"""Tests for the ``rupsycho`` command (offline, deterministic, fast).

Most tests run the whole command in-process with a scripted stand-in for the model. The tests
that check seeding and reproducibility use the real (tiny, offline) Hugging Face model of
``tests/tiny_hf.py``.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from langchain_core.language_models.llms import LLM

import rupsycho as rup
from rupsycho import cli
from rupsycho.cli import build_parser, main
from rupsycho.experiment import ExperimentDocument
from rupsycho.models.model import LocalHuggingFaceModelConfig

from .conftest import CONFIG_PATH

REPO_ROOT = Path(__file__).parent.parent
ANSWER = '{answer: "4. Agree a little"}'
ITEMS, PERSONAS = 4, 2  # size of the BFI test configuration
COMMANDS = ("run", "validate", "prompt", "examples", "postprocess", "configurator")


# ============================== helpers and fixtures ==============================


class ScriptedLLM(LLM):
    """Stand-in model: always picks option 4 and fails on prompts that contain ``fail_on``.

    Like API models it has a ``seed`` field, so the run loop can seed it without a warning.
    """

    fail_on: str = ""
    long_message: bool = False
    seed: int | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _call(self, prompt: str, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> str:
        if self.fail_on and self.fail_on in prompt:
            if self.long_message:
                raise RuntimeError("very long explanation " * 100 + "\nsecond line")
            raise RuntimeError(f"cannot answer for {self.fail_on}")
        return ANSWER


def local_model(name: str = "ok") -> dict[str, str]:
    """A model section whose ``name_or_path`` tells the ``loaded_models`` fixture how it behaves."""
    return {"type": "local_huggingface", "name_or_path": name}


@pytest.fixture
def loaded_models(monkeypatch) -> list[str]:
    """Load local Hugging Face models as a ScriptedLLM (no torch needed); lists what was loaded.

    A model whose ``name_or_path`` is ``fail:TEXT`` fails on every prompt containing TEXT
    (``longfail:TEXT`` with a very long message).
    """
    loaded: list[str] = []

    def load_model(self: LocalHuggingFaceModelConfig) -> ScriptedLLM:
        loaded.append(self.name_or_path)
        kind, _, text = self.name_or_path.partition(":")
        failing = kind in ("fail", "longfail")
        return ScriptedLLM(fail_on=text if failing else "", long_message=kind == "longfail")

    monkeypatch.setattr(LocalHuggingFaceModelConfig, "load_model", load_model)
    return loaded


@pytest.fixture
def forbid_model_loading(monkeypatch) -> list[str]:
    """Make loading any local model an error; the list records attempts."""
    attempts: list[str] = []

    def load_model(self: LocalHuggingFaceModelConfig) -> None:
        attempts.append(self.name_or_path)
        raise AssertionError("a model must not be loaded here")

    monkeypatch.setattr(LocalHuggingFaceModelConfig, "load_model", load_model)
    return attempts


DROP = object()  # in make_config(): remove this section


@pytest.fixture
def make_config(tmp_path, config_dict) -> Callable[..., Path]:
    """Write a variation of the BFI test configuration; keyword arguments replace sections."""

    def make(filename: str = "config.json", **sections: Any) -> Path:
        config = copy.deepcopy(config_dict)
        config["parameters"] = {"seeds": ["7"]}
        config["models"] = {"scripted": local_model()}
        for key, value in sections.items():
            if value is DROP:
                config.pop(key, None)
            else:
                config[key] = value
        path = tmp_path / filename
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    return make


@pytest.fixture
def config(make_config, loaded_models) -> Path:
    """A runnable configuration (one scripted model, seed 7) with the scripted model loader."""
    return make_config()


def cli_run(capsys, *args: object) -> tuple[int, str, str]:
    """Run ``rupsycho <args>`` in-process and return ``(exit code, stdout, stderr)``."""
    code = main([str(arg) for arg in args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def read_answers(path: Path) -> list[str]:
    """The answers stored in a results file, in run order."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)["answer"].tolist()
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line)["answer"] for line in path.read_text("utf-8").splitlines()]
    return rup.experiment_from_file(path).get_answers_as_dataframe()["Answer"].tolist()


def answers_of_the_python_api(config_path: Path, seed: str, cumulative: bool = False) -> list[str]:
    """What ``experiment.run()`` answers for ``seed``, to compare the command with."""
    experiment = rup.experiment_from_file(config_path)
    experiment.parameters.seeds = [seed]
    experiment.run(cumulative=cumulative, show_progress=False)
    return experiment.get_answers_as_dataframe()["Answer"].tolist()


# ============================== command line and usage ==============================


class TestUsage:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])
        assert exit_info.value.code == 0
        assert capsys.readouterr().out.strip() == f"rupsycho {rup.__version__}"

    def test_help_lists_commands_and_exit_codes(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            main(["--help"])
        out = capsys.readouterr().out
        assert exit_info.value.code == 0
        assert all(command in out for command in COMMANDS)
        assert "exit codes" in out and "RUPSYCHO_DEBUG" in out

    @pytest.mark.parametrize("command", COMMANDS)
    def test_every_command_has_help(self, command, capsys):
        with pytest.raises(SystemExit) as exit_info:
            main([command, "--help"])
        out = capsys.readouterr().out
        assert exit_info.value.code == 0
        assert out.startswith(f"usage: rupsycho {command}")

    @pytest.mark.parametrize("command", ["run", "validate", "prompt", "examples", "postprocess"])
    def test_help_shows_examples(self, command, capsys):
        with pytest.raises(SystemExit):
            main([command, "--help"])
        assert f"rupsycho {command}" in capsys.readouterr().out.split("examples:")[1]

    @pytest.mark.parametrize(
        "argv",
        [
            [],
            ["unknown"],
            ["run"],
            ["run", "c.json", "--on-error", "explode"],
            ["run", "c.json", "--max-concurrency", "0"],
            ["run", "c.json", "--max-concurrency", "many"],
            ["run", "c.json", "--seeds", "-1"],
            ["run", "c.json", "--seeds", "one"],
            ["run", "c.json", "--seed", "1"],  # abbreviations are not accepted
            ["prompt", "c.json", "--item", "first"],
            ["examples"],
            ["examples", "copy", "bfi"],
            ["postprocess", "c.json", "r.csv"],  # -o is required
            ["validate"],
        ],
    )
    def test_usage_errors_exit_with_2(self, argv, capsys):
        with pytest.raises(SystemExit) as exit_info:
            main(argv)
        assert exit_info.value.code == 2
        assert "error:" in capsys.readouterr().err

    def test_run_defaults(self):
        args = build_parser().parse_args(["run", "c.json"])
        assert (args.output, args.force, args.model, args.seeds) == (None, False, None, None)
        assert (args.cumulative, args.max_concurrency, args.on_error) == (False, 1, "warn")
        assert (args.dry_run, args.quiet, args.handler) == (False, False, cli._cmd_run)

    def test_options_accept_several_values(self):
        args = build_parser().parse_args(
            ["run", "c.json", "--model", "a", "b", "--model", "c", "--seeds", "1", "2"]
        )
        assert args.model == ["a", "b", "c"] and args.seeds == [1, 2]

    def test_docs_cover_every_command_and_option(self):
        docs = (REPO_ROOT / "docs" / "cli.md").read_text(encoding="utf-8")
        parser = build_parser()
        subparsers = next(a for a in parser._actions if hasattr(a, "choices") and a.choices)
        assert set(subparsers.choices) == set(COMMANDS)
        assert all(f"## `{command}`" in docs for command in COMMANDS)

        def walk(command: Any) -> None:
            for action in command._actions:
                for option in action.option_strings:
                    assert option in docs, f"{option} is not documented in docs/cli.md"
                if hasattr(action, "choices") and isinstance(action.choices, dict):
                    for name, sub in action.choices.items():
                        assert name in docs
                        walk(sub)

        walk(parser)


class TestExamplesInTheDocumentation:
    """Every command line shown to users must be accepted by the argument parser."""

    @staticmethod
    def parse(argv: list[str]) -> None:
        try:
            build_parser().parse_args(argv)
        except SystemExit as exit_info:
            assert exit_info.code == 0, f"rejected: rupsycho {shlex.join(argv)}"  # --help

    def test_bash_blocks_of_docs_cli_md(self, capsys):
        docs = (REPO_ROOT / "docs" / "cli.md").read_text(encoding="utf-8")
        lines = [
            line.strip()
            for block in re.findall(r"```bash\n(.*?)```", docs, re.DOTALL)
            for line in block.splitlines()
            if line.strip().startswith("rupsycho ")
        ]
        assert len(lines) >= 20
        for line in lines:
            argv = shlex.split(line, comments=True)[1:]
            self.parse(argv[: argv.index(">")] if ">" in argv else argv)

    @pytest.mark.parametrize("command", [[], *([c] for c in COMMANDS), ["examples", "copy"]])
    def test_examples_of_the_help_texts(self, command, capsys):
        with pytest.raises(SystemExit):
            main([*command, "--help"])
        help_text = capsys.readouterr().out
        lines = [
            re.split(r"\s{2,}", line.strip())[0]  # the explanation follows after two spaces
            for line in help_text.partition("examples:")[2].splitlines()
            if line.strip().startswith("rupsycho ")
        ]
        for line in lines:
            argv = shlex.split(line)[1:]
            self.parse(argv[: argv.index(">")] if ">" in argv else argv)
        assert lines or command in (["configurator"], ["examples", "copy"])


# ============================== run: output formats ==============================


class TestRunWithTheTinyModel:
    """The whole command with a real (tiny, offline) Hugging Face model."""

    @pytest.fixture
    def tiny_config(self, make_config, tiny_hf_config) -> Path:
        return make_config(
            "tiny.json", models={"tiny": tiny_hf_config}, parameters={"seeds": ["1"]}
        )

    def test_csv_output(self, tiny_config, tmp_path, capsys):
        out = tmp_path / "answers.csv"
        code, stdout, stderr = cli_run(capsys, "run", tiny_config, "-o", out)

        assert code == 0 and stdout == ""
        assert f"{ITEMS * PERSONAS} model calls" in stderr and str(out) in stderr
        assert f"{ITEMS * PERSONAS}/{ITEMS * PERSONAS}" in stderr  # the progress bar
        frame = pd.read_csv(out, dtype=str, keep_default_na=False)
        assert len(frame) == ITEMS * PERSONAS
        assert set(frame["model_id"]) == {"tiny"} and set(frame["random_seed"]) == {"1"}
        assert set(frame["profile_id"]) == {"Optimistic Persona", "Conservative Persona"}

    def test_every_format_holds_the_answers_of_the_python_api(self, tiny_config, tmp_path, capsys):
        expected = answers_of_the_python_api(tiny_config, "1")
        assert len(expected) == ITEMS * PERSONAS

        for suffix in ("csv", "jsonl", "json"):
            out = tmp_path / f"answers.{suffix}"
            assert cli_run(capsys, "run", tiny_config, "-o", out, "-q")[0] == 0
            assert read_answers(out) == expected, suffix

    def test_runs_are_reproducible_and_seeds_matter(self, tiny_config, tmp_path, capsys):
        def answers(*extra: object) -> pd.DataFrame:
            out = tmp_path / f"run{len(list(tmp_path.glob('run*.csv')))}.csv"
            assert cli_run(capsys, "run", tiny_config, "-o", out, "-q", *extra)[0] == 0
            return pd.read_csv(out, dtype=str, keep_default_na=False).drop(columns="time")

        first = answers()
        pd.testing.assert_frame_equal(answers(), first)  # same command, same csv (apart from time)
        assert answers("--seeds", 2)["answer"].tolist() != first["answer"].tolist()

    def test_seeds_override_the_configuration(self, tiny_config, tmp_path, capsys):
        out = tmp_path / "answers.csv"
        assert cli_run(capsys, "run", tiny_config, "-o", out, "-q", "--seeds", 1, 2, 1)[0] == 0
        frame = pd.read_csv(out, dtype=str, keep_default_na=False)

        assert len(frame) == 2 * ITEMS * PERSONAS  # the repeated seed is only run once
        assert set(frame["random_seed"]) == {"1", "2"}
        seed_one = frame[frame["random_seed"] == "1"]["answer"].tolist()
        assert seed_one == answers_of_the_python_api(tiny_config, "1")

    def test_cumulative_runs_equal_the_python_api(self, tiny_config, tmp_path, capsys):
        out = tmp_path / "answers.csv"
        assert cli_run(capsys, "run", tiny_config, "-o", out, "--cumulative", "-q")[0] == 0
        assert read_answers(out) == answers_of_the_python_api(tiny_config, "1", cumulative=True)

    def test_local_models_ignore_concurrency_with_a_warning(self, tiny_config, tmp_path, capsys):
        out = tmp_path / "answers.csv"
        code, _, stderr = cli_run(
            capsys, "run", tiny_config, "-o", out, "--max-concurrency", 4, "-q"
        )
        assert code == 0 and len(pd.read_csv(out)) == ITEMS * PERSONAS
        # Said once, in the compact format of the command
        assert stderr.count("rupsycho: warning: Model 'tiny' runs in this process") == 1


class TestRunOutput:
    def test_table_on_stdout_and_summary_on_stderr(self, config, capsys):
        code, stdout, stderr = cli_run(capsys, "run", config)

        assert code == 0
        header, *rows = stdout.splitlines()
        assert header.split()[:2] == ["Instruction", "ID"] and "Answer" in header
        assert len(rows) == ITEMS * PERSONAS
        assert rows[0].split()[:2] == ["0", "I"]  # the item id, then its question: no index column
        assert all(ANSWER in row for row in rows)
        assert f"rupsycho: {ITEMS * PERSONAS} model calls in" in stderr
        assert "8/8" in stderr  # progress bar
        assert "model calls" not in stdout

    def test_quiet_hides_the_progress_bar_and_the_summary(self, config, tmp_path, capsys):
        code, stdout, stderr = cli_run(capsys, "run", config, "-o", tmp_path / "a.csv", "-q")
        assert (code, stdout, stderr) == (0, "", "")

    def test_jsonl_has_one_object_per_answer(self, config, tmp_path, capsys):
        out = tmp_path / "answers.jsonl"
        assert cli_run(capsys, "run", config, "-o", out, "-q")[0] == 0
        rows = [json.loads(line) for line in out.read_text("utf-8").splitlines()]

        assert len(rows) == ITEMS * PERSONAS
        assert {row["answer"] for row in rows} == {ANSWER}
        assert {row["model_id"] for row in rows} == {"scripted"}
        assert [row["instruction_item_id"] for row in rows[:4]] == [0, 0, 1, 1]

    def test_json_export_can_be_loaded_again(self, config, tmp_path, capsys):
        out = tmp_path / "experiment.json"
        assert cli_run(capsys, "run", config, "-o", out, "-q")[0] == 0

        reloaded = rup.experiment_from_file(out)
        frame = reloaded.get_answers_as_dataframe()
        assert len(frame) == ITEMS * PERSONAS and set(frame["Answer"]) == {ANSWER}

    @pytest.mark.parametrize("suffix", [".CSV", ".Jsonl", ".JSON"])
    def test_extensions_are_case_insensitive(self, config, tmp_path, capsys, suffix):
        out = tmp_path / f"answers{suffix}"
        assert cli_run(capsys, "run", config, "-o", out, "-q")[0] == 0
        assert len(read_answers(out)) == ITEMS * PERSONAS

    def test_output_directories_are_created(self, config, tmp_path, capsys):
        out = tmp_path / "results" / "nested" / "answers.csv"
        assert cli_run(capsys, "run", config, "-o", out, "-q")[0] == 0
        assert out.is_file()

    @pytest.mark.parametrize("suffix", [".csv", ".jsonl", ".json"])
    def test_existing_output_is_protected_unless_forced(self, config, tmp_path, capsys, suffix):
        out = tmp_path / f"answers{suffix}"
        out.write_text("precious\n", encoding="utf-8")

        code, stdout, stderr = cli_run(capsys, "run", config, "-o", out, "-q")
        assert code == 1 and stdout == ""
        assert "already exists" in stderr and "--force" in stderr and "Traceback" not in stderr
        assert out.read_text(encoding="utf-8") == "precious\n"

        assert cli_run(capsys, "run", config, "-o", out, "-q", "--force")[0] == 0
        assert len(read_answers(out)) == ITEMS * PERSONAS  # replaced, not appended to

    def test_forcing_twice_does_not_duplicate_rows(self, config, tmp_path, capsys):
        out = tmp_path / "answers.csv"
        for _ in range(2):
            assert cli_run(capsys, "run", config, "-o", out, "-q", "-f")[0] == 0
        assert out.read_text("utf-8").count("experiment_name") == 1
        assert len(read_answers(out)) == ITEMS * PERSONAS

    def test_unknown_output_format_is_a_usage_error(self, config, tmp_path, capsys):
        with pytest.raises(SystemExit) as exit_info:
            main(["run", str(config), "-o", str(tmp_path / "answers.xlsx")])
        assert exit_info.value.code == 2
        assert "unsupported output format '.xlsx'" in capsys.readouterr().err

    def test_answers_are_written_while_the_run_is_going(
        self, make_config, loaded_models, tmp_path, capsys
    ):
        """The run stops at the first failure, but the answer given before it is on disk."""
        config = make_config(models={"flaky": local_model("fail:Grueber")})
        out = tmp_path / "partial.csv"
        code, _, stderr = cli_run(capsys, "run", config, "-o", out, "--on-error", "raise", "-q")

        assert code == 1 and "cannot answer for Grueber" in stderr
        frame = pd.read_csv(out)
        assert frame["profile_id"].tolist() == ["Optimistic Persona"]

    def test_flags_are_passed_to_the_run_loop(self, config, monkeypatch, capsys):
        seen: dict[str, Any] = {}

        def spy(self, callbacks=(), cumulative=False, **options):
            seen.update(callbacks=callbacks, cumulative=cumulative, **options)
            return rup.RunSummary(n_calls=1)

        monkeypatch.setattr(ExperimentDocument, "run", spy)
        argv = ["--cumulative", "--max-concurrency", 3, "--on-error", "ignore", "-q"]
        assert cli_run(capsys, "run", config, *argv)[0] == 0
        assert seen == {
            "callbacks": [],
            "cumulative": True,
            "max_concurrency": 3,
            "on_error": "ignore",
            "show_progress": False,
        }

        assert cli_run(capsys, "run", config)[0] == 0  # the defaults
        assert (seen["cumulative"], seen["max_concurrency"], seen["on_error"]) == (False, 1, "warn")
        assert seen["show_progress"] is True

    def test_json_is_only_written_when_the_run_is_over(self, config, tmp_path, monkeypatch, capsys):
        out = tmp_path / "experiment.json"
        export = ExperimentDocument.export_to_file
        seen: list[bool] = []

        def spy(self, filename, **options):
            seen.append(Path(filename).exists())
            export(self, filename, **options)

        monkeypatch.setattr(ExperimentDocument, "export_to_file", spy)
        assert cli_run(capsys, "run", config, "-o", out, "-q")[0] == 0
        assert seen == [False] and out.is_file()

    def test_json_that_cannot_be_written_is_reported(self, config, tmp_path, monkeypatch, capsys):
        def refuse(self, filename, **options):
            raise PermissionError(13, "Permission denied", str(filename))

        monkeypatch.setattr(ExperimentDocument, "export_to_file", refuse)
        code, _, stderr = cli_run(capsys, "run", config, "-o", tmp_path / "e.json", "-q")
        assert code == 1 and "Permission denied" in stderr and "Traceback" not in stderr


# ============================== run: models, seeds, failures ==============================


class TestRunSelection:
    @pytest.fixture
    def three_models(self, make_config, loaded_models) -> Path:
        return make_config(models={name: local_model(name) for name in ("a", "b", "c")})

    def test_all_models_run_by_default(self, three_models, loaded_models, tmp_path, capsys):
        out = tmp_path / "answers.csv"
        assert cli_run(capsys, "run", three_models, "-o", out, "-q")[0] == 0
        assert sorted(loaded_models) == ["a", "b", "c"]
        assert len(read_answers(out)) == 3 * ITEMS * PERSONAS

    def test_model_filter_runs_and_loads_only_the_selected_models(
        self, three_models, loaded_models, tmp_path, capsys
    ):
        out = tmp_path / "answers.csv"
        assert cli_run(capsys, "run", three_models, "-o", out, "-q", "--model", "b")[0] == 0

        assert loaded_models == ["b"]
        assert set(pd.read_csv(out)["model_id"]) == {"b"}

    @pytest.mark.parametrize("args", [["--model", "a", "c"], ["--model", "a", "--model", "c"]])
    def test_several_models(self, three_models, loaded_models, tmp_path, capsys, args):
        out = tmp_path / "answers.csv"
        assert cli_run(capsys, "run", three_models, "-o", out, "-q", *args)[0] == 0
        assert sorted(set(pd.read_csv(out)["model_id"])) == ["a", "c"]

    def test_unknown_model_ids_are_reported_with_the_available_ones(
        self, three_models, loaded_models, capsys
    ):
        code, stdout, stderr = cli_run(capsys, "run", three_models, "--model", "a", "zzz", "yyy")
        assert code == 1 and stdout == "" and loaded_models == []
        assert "unknown model id(s): zzz, yyy (available: a, b, c)" in stderr

    def test_seeds_override_the_configuration(self, config, tmp_path, capsys):
        out = tmp_path / "answers.csv"
        assert cli_run(capsys, "run", config, "-o", out, "-q", "--seeds", 5, 6)[0] == 0
        assert sorted(set(pd.read_csv(out)["random_seed"])) == [5, 6]

    def test_configured_seeds_are_used_without_the_option(self, config, tmp_path, capsys):
        out = tmp_path / "answers.csv"
        code, _, stderr = cli_run(capsys, "run", config, "-o", out)
        assert set(pd.read_csv(out)["random_seed"]) == {7}
        assert "rupsycho: note:" not in stderr

    def test_missing_seeds_are_announced_because_the_run_is_not_reproducible(
        self, make_config, loaded_models, tmp_path, capsys
    ):
        config = make_config(parameters={})
        code, _, stderr = cli_run(capsys, "run", config, "-o", tmp_path / "a.csv")
        assert code == 0
        assert "rupsycho: note: the configuration defines no seeds; using the random seed" in stderr
        assert "--seeds" in stderr

        _, _, stderr = cli_run(capsys, "run", config, "-o", tmp_path / "b.csv", "-q")
        assert stderr == ""  # quiet mode skips advice

        _, _, stderr = cli_run(capsys, "run", config, "-o", tmp_path / "c.csv", "--seeds", 3)
        assert "rupsycho: note:" not in stderr  # the seeds are given now

    def test_lazy_load_models_false_is_not_honoured_so_that_models_load_one_by_one(
        self, make_config, loaded_models, capsys
    ):
        config = make_config(
            models={name: local_model(name) for name in ("a", "b")},
            parameters={"seeds": ["7"], "lazy_load_models": False},
        )
        assert cli_run(capsys, "run", config, "--model", "a", "-q")[0] == 0
        assert loaded_models == ["a"]


class TestRunFailures:
    @pytest.fixture
    def flaky(self, make_config, loaded_models) -> Path:
        """Every call for the conservative persona fails."""
        return make_config(models={"flaky": local_model("fail:Grueber")})

    def test_failed_calls_give_exit_code_3_and_are_reported(self, flaky, tmp_path, capsys, caplog):
        out = tmp_path / "answers.csv"
        code, _, stderr = cli_run(capsys, "run", flaky, "-o", out)

        assert code == 3
        line = next(line for line in stderr.splitlines() if "model calls in" in line)
        assert line.startswith("rupsycho: 8 model calls in") and "4 failed" in line
        assert "first error: RuntimeError: cannot answer for Grueber" in line and str(out) in line
        assert "missing from the results" not in stderr  # the library warning is replaced by this
        assert "Model call failed" in caplog.text  # logged for every failed call
        answers = pd.read_csv(out)["answer"]
        assert answers.notna().sum() == ITEMS  # only the optimistic persona answered

    def test_a_long_error_is_shortened_to_one_line(self, make_config, loaded_models, capsys):
        config = make_config(models={"chatty": local_model("longfail:Grueber")})
        code, _, stderr = cli_run(capsys, "run", config, "-q")

        line = next(line for line in stderr.splitlines() if "first error:" in line)
        assert code == 3 and len(line) < 450 and "[...]" in line
        assert "very long explanation" in line and "second line" not in stderr

    def test_the_failure_is_reported_even_when_quiet(self, flaky, capsys):
        code, stdout, stderr = cli_run(capsys, "run", flaky, "-q")
        assert code == 3 and "4 failed" in stderr
        assert len(stdout.splitlines()) == 1 + ITEMS  # the table lists the answers that exist

    def test_ignore_is_silent_but_still_exit_code_3(self, flaky, capsys, caplog):
        code, _, stderr = cli_run(capsys, "run", flaky, "-q", "--on-error", "ignore")
        assert code == 3 and "4 failed" in stderr
        assert "Model call failed" not in caplog.text

    def test_raise_stops_with_exit_code_1_and_a_hint(self, flaky, capsys):
        code, _, stderr = cli_run(capsys, "run", flaky, "-q", "--on-error", "raise")
        assert code == 1
        assert "rupsycho: error: RuntimeError: cannot answer for Grueber" in stderr
        assert "RUPSYCHO_DEBUG" in stderr and "Traceback" not in stderr

    def test_debug_mode_shows_the_traceback(self, flaky, monkeypatch):
        monkeypatch.setenv("RUPSYCHO_DEBUG", "1")
        with pytest.raises(RuntimeError, match="cannot answer for Grueber"):
            main(["run", str(flaky), "-q", "--on-error", "raise"])

    def test_interruption_exits_with_130(self, config, monkeypatch, capsys):
        def interrupt(self, *args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(ExperimentDocument, "run", interrupt)
        code, _, stderr = cli_run(capsys, "run", config)
        assert code == 130 and "interrupted" in stderr

    def test_missing_extras_are_reported_without_a_traceback(self, config, monkeypatch, capsys):
        def missing(self: LocalHuggingFaceModelConfig) -> None:
            raise ImportError(
                "Local models need 'transformers'. Install: pip install 'rupsycho[huggingface]'"
            )

        monkeypatch.setattr(LocalHuggingFaceModelConfig, "load_model", missing)
        code, _, stderr = cli_run(capsys, "run", config, "-q")
        assert code == 1 and "pip install 'rupsycho[huggingface]'" in stderr
        assert "ImportError" not in stderr and "RUPSYCHO_DEBUG" not in stderr

    def test_unexpected_errors_name_their_type(self, config, monkeypatch, capsys):
        def broken(self, *args, **kwargs):
            raise KeyError("boom")

        monkeypatch.setattr(ExperimentDocument, "run", broken)
        code, _, stderr = cli_run(capsys, "run", config)
        assert code == 1 and "rupsycho: error: KeyError: 'boom'" in stderr
        assert "RUPSYCHO_DEBUG=1" in stderr

    def test_cumulative_needs_a_chat_prompt(self, make_config, loaded_models, capsys):
        plain = {"type": "normal", "template": "{persona_description} {question} {answer_options}"}
        config = make_config(prompt_template=plain)
        assert cli_run(capsys, "run", config, "-q")[0] == 0

        code, _, stderr = cli_run(capsys, "run", config, "-q", "--cumulative")
        assert code == 1 and "Cumulative mode needs a chat prompt template" in stderr

    def test_output_that_is_a_directory_is_an_error(self, config, tmp_path, capsys):
        folder = tmp_path / "folder.csv"
        folder.mkdir()
        code, _, stderr = cli_run(capsys, "run", config, "-o", folder, "-q", "--force")
        assert code == 1 and "Traceback" not in stderr

    @pytest.mark.parametrize("command", [["run"], ["run", "--dry-run"], ["prompt"], ["validate"]])
    def test_a_missing_configuration_is_an_error(self, command, tmp_path, capsys):
        missing = tmp_path / "missing.json"
        code, stdout, stderr = cli_run(capsys, *command, missing)
        assert code == 1
        assert f"{missing}: no such file" in stdout + stderr and "Traceback" not in stderr


# ============================== run --dry-run ==============================


class TestDryRun:
    def test_reports_the_work_and_the_first_prompt_without_loading_a_model(
        self, make_config, forbid_model_loading, capsys
    ):
        config = make_config(models={"small": local_model("s"), "large": local_model("l")})
        code, stdout, stderr = cli_run(capsys, "run", config, "--dry-run")

        assert code == 0 and forbid_model_loading == [] and stderr == ""
        assert "Dry run: no model was loaded" in stdout
        assert "models:     small, large" in stdout and "seeds:      7" in stdout
        assert (
            f"calls:      {ITEMS} items x {PERSONAS} personas x 2 models x 1 seeds = 16 calls"
            in stdout
        )
        assert "output:     stdout (table)" in stdout
        assert "persona 0 'Optimistic Persona'" in stdout
        assert "Ms Muller is 18 years old" in stdout and "I see myself as someone who..." in stdout
        assert "1. Disagree strongly" in stdout  # the questionnaire's default answer options

    def test_options_change_the_plan(self, make_config, forbid_model_loading, tmp_path, capsys):
        config = make_config(models={"small": local_model("s"), "large": local_model("l")})
        out = tmp_path / "answers.csv"
        argv = ["run", config, "--dry-run", "--model", "large", "--seeds", 1, 2, 3, "-o", out]
        code, stdout, _ = cli_run(capsys, *argv)

        assert code == 0
        assert "models:     large" in stdout and "seeds:      1, 2, 3" in stdout
        assert f"{ITEMS} items x {PERSONAS} personas x 1 models x 3 seeds = 24 calls" in stdout
        assert f"output:     {out}" in stdout
        assert not out.exists()  # nothing is written

    def test_an_existing_output_is_reported_as_a_dry_run_would_fail_too(
        self, make_config, forbid_model_loading, tmp_path, capsys
    ):
        out = tmp_path / "answers.csv"
        out.write_text("precious\n", encoding="utf-8")
        code, stdout, stderr = cli_run(capsys, "run", make_config(), "--dry-run", "-o", out)
        assert code == 1 and "already exists" in stderr and stdout == ""
        assert out.read_text(encoding="utf-8") == "precious\n"

    def test_no_seeds_mean_the_default_seed(self, make_config, forbid_model_loading, capsys):
        config = make_config(parameters={"seeds": []})
        code, stdout, stderr = cli_run(capsys, "run", config, "--dry-run")
        assert code == 0 and "seeds:      42" in stdout and "x 1 seeds =" in stdout
        assert "note" not in stderr  # a run without seeds is deterministic: the default is 42

    def test_lazy_load_models_false_does_not_load_models_either(
        self, make_config, forbid_model_loading, capsys
    ):
        config = make_config(parameters={"seeds": ["7"], "lazy_load_models": False})
        assert cli_run(capsys, "run", config, "--dry-run")[0] == 0
        assert forbid_model_loading == []

    def test_a_prompt_that_cannot_be_assembled_stops_the_run_before_any_model_is_loaded(
        self, make_config, config_dict, forbid_model_loading, capsys
    ):
        personas = copy.deepcopy(config_dict["demographic_profiles"])
        personas["Conservative Persona"]["template"] = "{title} {nmae}"
        config = make_config(demographic_profiles=personas)

        code, _, stderr = cli_run(capsys, "run", config, "-q")
        assert code == 1 and "persona 1 'Conservative Persona'" in stderr
        assert forbid_model_loading == []

    def test_a_dry_run_reports_invalid_configurations_too(self, make_config, capsys):
        code, _, stderr = cli_run(capsys, "run", make_config(models={}), "--dry-run")
        assert code == 1 and "no models" in stderr


# ============================== validate ==============================


def invalid(label: str, mutate: Callable[[dict], Any], message: str) -> Any:
    return pytest.param(mutate, message, id=label)


class TestValidate:
    def test_valid_configuration(self, config, forbid_model_loading, capsys):
        code, stdout, stderr = cli_run(capsys, "validate", config)
        name = "Generative Models for Big Five Inventory"
        assert code == 0 and stderr == "" and forbid_model_loading == []
        expected = f"OK  {config}: {name} - 4 items x 2 personas x 1 models x 1 seeds = 8 calls\n"
        assert stdout == expected

    def test_an_experiment_without_a_name(self, make_config, forbid_model_loading, capsys):
        config = make_config(name=DROP)
        code, stdout, _ = cli_run(capsys, "validate", config)
        assert code == 0 and stdout.startswith(f"OK  {config}: unnamed - 4 items")

    def test_the_repository_test_configuration_is_valid_and_loads_no_model(
        self, forbid_model_loading, capsys
    ):
        code, stdout, _ = cli_run(capsys, "validate", CONFIG_PATH)
        assert code == 0 and forbid_model_loading == []
        assert stdout.startswith(f"OK  {CONFIG_PATH}:") and "= 8 calls" in stdout

    def test_every_file_gets_a_line_and_one_invalid_file_fails_the_command(
        self, make_config, forbid_model_loading, tmp_path, capsys
    ):
        good = make_config("good.json")
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        missing = tmp_path / "missing.json"

        code, stdout, _ = cli_run(capsys, "validate", good, bad, missing, good)
        lines = stdout.splitlines()
        assert code == 1 and len(lines) == 4
        assert [line.split()[0] for line in lines] == ["OK", "ERROR", "ERROR", "OK"]
        assert lines[1].startswith(f"ERROR {bad}: invalid JSON (")
        assert lines[2] == f"ERROR {missing}: no such file"

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            invalid(
                "questionnaire-is-not-an-object",
                lambda c: c.update(questionnaire="nope"),
                "'questionnaire' must be an object (a dictionary)",
            ),
            invalid(
                "questionnaire-name-missing",
                lambda c: c["questionnaire"].pop("name"),
                "Questionnaire.name: Field required",
            ),
            invalid(
                "several-problems-are-counted",
                lambda c: [
                    c["questionnaire"].pop("name"),
                    c["questionnaire"].pop("general_instruction"),
                ],
                "Questionnaire.name: Field required (and 1 more error)",
            ),
            invalid(
                "the-count-of-problems-is-pluralised",
                lambda c: c.update(questionnaire={"instruction_items": "x"}),
                "Questionnaire.name: Field required (and 2 more errors)",
            ),
            invalid(
                "unknown-model-type",
                lambda c: c.update(models={"x": {"type": "nope"}}),
                "Unknown or missing model type for key: x",
            ),
            invalid(
                "model-without-a-name",
                lambda c: c.update(models={"x": {"type": "local_huggingface"}}),
                "LocalHuggingFaceModelConfig.name_or_path: Field required",
            ),
            invalid("no-models", lambda c: c.update(models={}), "no models"),
            invalid(
                "no-personas",
                lambda c: c.update(demographic_profiles={}),
                "no demographic_profiles",
            ),
            invalid(
                "no-items",
                lambda c: c["questionnaire"].update(instruction_items=[]),
                "the questionnaire has no instruction_items",
            ),
            invalid("no-questionnaire", lambda c: c.pop("questionnaire"), "no questionnaire"),
            invalid(
                "misspelled-section",
                lambda c: c.update(modles=c.pop("models")),
                "unknown top-level key(s): 'modles' (did you mean 'models'?)",
            ),
            invalid(
                "unclosed-prompt-template",
                lambda c: c.update(
                    prompt_template={
                        "type": "chat",
                        "messages": [{"role": "user", "content": "{x"}],
                    }
                ),
                "Invalid prompt template",
            ),
            invalid(
                "unknown-prompt-type",
                lambda c: c.update(prompt_template={"type": "wat"}),
                "Unknown or missing prompt type.",
            ),
            invalid(
                "seeds-must-be-a-list",
                lambda c: c.update(parameters={"seeds": {"first": 7}}),
                "ExperimentParameters.seeds: Value error, seeds must be a list of integers",
            ),
            invalid(
                "seeds-must-be-integers",
                lambda c: c.update(parameters={"seeds": ["7", "abc"]}),
                "ExperimentParameters.seeds: Value error, seeds must be integers, got 'abc'",
            ),
            invalid(
                "persona-template-with-an-unknown-attribute",
                lambda c: c["demographic_profiles"]["Conservative Persona"].update(
                    template="{title} {nmae} is {age}"
                ),
                "cannot assemble the prompt of item 0, persona 1 'Conservative Persona': "
                "ValueError(\"The persona template uses the attribute 'nmae'",
            ),
            invalid(
                "item-without-answer-options-and-no-defaults",
                lambda c: (
                    c["questionnaire"].pop("default_answer_options"),
                    c["questionnaire"]["instruction_items"][0].update(
                        answer_options={"a": {"text": "yes"}}
                    ),
                ),
                "cannot assemble the prompt of item 1, persona 0 'Optimistic Persona': "
                "AttributeError",
            ),
            invalid(
                "prompt-template-with-an-unknown-variable",
                lambda c: c.update(
                    prompt_template={
                        "type": "chat",
                        "messages": [{"role": "user", "content": "{question} {nonsense}"}],
                    }
                ),
                "cannot assemble the prompt of item 0, persona 0 'Optimistic Persona'",
            ),
            invalid(
                "parameters-must-be-an-object",
                lambda c: c.update(parameters="fast"),
                "'parameters' must be an object (a dictionary)",
            ),
            invalid(
                "personas-must-be-an-object",
                lambda c: c.update(demographic_profiles=["a"]),
                "'demographic_profiles' must be an object (a dictionary)",
            ),
        ],
    )
    def test_invalid_configurations_get_a_one_line_reason(
        self, mutate, message, config_dict, tmp_path, capsys
    ):
        broken = copy.deepcopy(config_dict)
        broken["models"] = {"scripted": local_model()}
        mutate(broken)
        path = tmp_path / "broken.json"
        path.write_text(json.dumps(broken), encoding="utf-8")

        code, stdout, stderr = cli_run(capsys, "validate", path)
        assert code == 1
        assert stdout.startswith(f"ERROR {path}: ") and len(stdout.splitlines()) == 1
        assert message in stdout
        assert "Traceback" not in stdout + stderr

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            (b"", "invalid JSON"),
            (b'{"name": "x", ', "invalid JSON (Expecting property name"),
            (b"[1, 2]", "the top level must be a JSON object"),
            (b"\xff\xfe\x00", "invalid JSON ('utf-8' codec"),
        ],
    )
    def test_files_that_are_not_json_objects(self, content, message, tmp_path, capsys):
        path = tmp_path / "broken.json"
        path.write_bytes(content)
        code, stdout, _ = cli_run(capsys, "validate", path)
        assert code == 1 and f"ERROR {path}: {message}" in stdout

    def test_only_the_first_line_of_an_error_is_shown(self, config, monkeypatch, capsys):
        def refuse(_config):
            raise ValueError("first line\nsecond line\nthird line")

        monkeypatch.setattr(rup, "experiment_from_dict", refuse)
        code, stdout, _ = cli_run(capsys, "validate", config)
        assert code == 1 and stdout == f"ERROR {config}: first line\n"

    def test_a_directory_is_not_a_configuration(self, tmp_path, capsys):
        code, stdout, _ = cli_run(capsys, "validate", tmp_path)
        assert code == 1 and f"ERROR {tmp_path}: no such file" in stdout

    def test_unreadable_files_are_reported(self, config, monkeypatch, capsys):
        def deny(self, *args, **kwargs):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(Path, "read_text", deny)
        code, stdout, _ = cli_run(capsys, "validate", config)
        assert code == 1 and "cannot read the file (Permission denied)" in stdout


# ============================== prompt ==============================


class TestPrompt:
    def test_default_prompt(self, config, capsys):
        code, stdout, stderr = cli_run(capsys, "prompt", config)

        assert code == 0
        assert "Act like you are Ms Muller is 18 years old" in stdout
        assert "Question: I see myself as someone who..." in stdout
        assert "Answer Options: 1. Disagree strongly, 2. Disagree a little" in stdout
        assert stdout.rstrip().endswith("Answer:")
        assert (
            stderr.strip() == "rupsycho: assembled prompt of item 0, persona 0 'Optimistic Persona'"
        )

    @pytest.mark.parametrize("persona", ["1", "Conservative Persona"])
    def test_item_and_persona_by_index_or_id(self, config, capsys, persona):
        code, stdout, stderr = cli_run(capsys, "prompt", config, "--item", 3, "--persona", persona)
        assert code == 0 and "Mr Grueber is 65 years old" in stdout
        assert "Question: Is depressed, blue" in stdout and "item 3, persona 1" in stderr

    def test_matches_the_python_api(self, config, capsys):
        experiment = rup.experiment_from_file(config)
        experiment.print_assembled_prompt(item_idx=2, persona_idx=1)
        printed = capsys.readouterr().out

        _, stdout, _ = cli_run(capsys, "prompt", config, "--item", 2, "--persona", 1)
        assert stdout.strip() in printed

    def test_answer_options_of_an_item_take_precedence_over_the_defaults(
        self, make_config, config_dict, capsys
    ):
        questionnaire = copy.deepcopy(config_dict["questionnaire"])
        own = {"a": {"text": "yes", "weight": 1}, "b": {"text": "no", "weight": 0}}
        questionnaire["instruction_items"][1]["answer_options"] = {
            "options": own,
            "delimiter": " | ",
        }
        path = make_config(questionnaire=questionnaire)

        _, first, _ = cli_run(capsys, "prompt", path, "--item", 0)
        _, second, _ = cli_run(capsys, "prompt", path, "--item", 1)
        assert "Answer Options: 1. Disagree strongly" in first
        assert "Answer Options: yes | no" in second

    def test_models_are_not_needed(self, make_config, forbid_model_loading, capsys):
        code, stdout, _ = cli_run(capsys, "prompt", make_config(models={}))
        assert code == 0 and "Ms Muller" in stdout

    @pytest.mark.parametrize(
        ("option", "value", "message"),
        [
            ("--item", 4, "--item 4 is out of range: 4 items (0-3)"),
            ("--item", -1, "--item -1 is out of range: 4 items (0-3)"),
            ("--persona", 2, "--persona '2' is out of range: 2 personas (0-1)"),
            ("--persona", -1, "--persona '-1' is out of range"),
            ("--persona", "Nobody", "--persona 'Nobody' is out of range"),
        ],
    )
    def test_out_of_range_indexes_are_reported_cleanly(
        self, config, capsys, option, value, message
    ):
        code, stdout, stderr = cli_run(capsys, "prompt", config, option, value)
        assert code == 1 and stdout == ""
        assert f"rupsycho: error: {message}" in stderr and "Traceback" not in stderr
        if option == "--persona":
            assert "Optimistic Persona, Conservative Persona" in stderr  # the valid ids

    def test_a_template_with_unknown_variables_is_reported(self, make_config, capsys):
        broken = {
            "type": "chat",
            "messages": [{"role": "user", "content": "{question} {nonsense}"}],
        }
        code, _, stderr = cli_run(capsys, "prompt", make_config(prompt_template=broken))
        assert code == 1 and "cannot assemble the prompt of item 0, persona 0" in stderr


# ============================== examples ==============================


class TestExamples:
    def test_list(self, capsys):
        code, stdout, stderr = cli_run(capsys, "examples", "list")
        assert code == 0 and stderr == ""
        assert stdout.split() == rup.list_examples() and "bfi" in stdout.split()

    def test_show_prints_the_example_as_json(self, capsys):
        code, stdout, _ = cli_run(capsys, "examples", "show", "bfi")
        assert code == 0 and json.loads(stdout) == rup.load_example_config("bfi")

    def test_copy_to_a_file(self, tmp_path, capsys):
        dest = tmp_path / "mine" / "experiment.json"
        code, stdout, stderr = cli_run(capsys, "examples", "copy", "bfi", dest)

        assert code == 0 and stdout == "" and f"wrote {dest}" in stderr
        assert json.loads(dest.read_text("utf-8")) == rup.load_example_config("bfi")
        assert dest.read_text("utf-8") == cli_run(capsys, "examples", "show", "bfi")[1]

    def test_copy_to_a_directory_keeps_the_example_name(self, tmp_path, capsys):
        assert cli_run(capsys, "examples", "copy", "bfi", tmp_path)[0] == 0
        assert (tmp_path / "bfi.json").is_file()

    def test_copy_refuses_to_overwrite_unless_forced(self, tmp_path, capsys):
        dest = tmp_path / "experiment.json"
        dest.write_text("precious", encoding="utf-8")

        code, _, stderr = cli_run(capsys, "examples", "copy", "bfi", dest)
        assert code == 1 and "already exists" in stderr and "--force" in stderr
        assert dest.read_text(encoding="utf-8") == "precious"

        assert cli_run(capsys, "examples", "copy", "bfi", dest, "--force")[0] == 0
        assert json.loads(dest.read_text("utf-8"))["name"]

    @pytest.mark.parametrize("action", ["show", "copy"])
    def test_unknown_examples(self, action, tmp_path, capsys):
        extra = [tmp_path / "x.json"] if action == "copy" else []
        code, stdout, stderr = cli_run(capsys, "examples", action, "nope", *extra)
        assert code == 1 and stdout == ""
        assert "rupsycho: error: Unknown example 'nope'. Available examples: bfi" in stderr

    def test_a_copied_example_is_a_valid_experiment(self, tmp_path, forbid_model_loading, capsys):
        dest = tmp_path / "bfi.json"
        cli_run(capsys, "examples", "copy", "bfi", dest)
        code, stdout, _ = cli_run(capsys, "validate", dest)
        assert code == 0 and stdout.startswith(f"OK  {dest}:") and forbid_model_loading == []


# ============================== postprocess ==============================


class TestPostprocess:
    @pytest.fixture
    def results(self, config, tmp_path, capsys) -> Path:
        """The answers of a (scripted) run, as written by ``run -o``."""
        out = tmp_path / "results.csv"
        assert cli_run(capsys, "run", config, "-o", out, "-q")[0] == 0
        return out

    def test_scores_the_results_of_a_run(self, config, results, tmp_path, capsys):
        out = tmp_path / "scored.csv"
        code, stdout, stderr = cli_run(capsys, "postprocess", config, results, "-o", out)

        assert code == 0 and stdout == ""
        assert f"rupsycho: wrote {out} ({ITEMS * PERSONAS} rows)" in stderr
        scored = pd.read_csv(out, dtype=str, keep_default_na=False)
        raw = pd.read_csv(results, dtype=str, keep_default_na=False)
        assert len(scored) == len(raw) == ITEMS * PERSONAS
        assert {"cleaned_answer", "validation_status", "valid", "decision"} <= set(scored.columns)
        # The questionnaire only has default answer options, which the judge falls back to
        assert set(scored["decision"]) == {"4. Agree a little"}
        assert set(scored["valid"]) == {"True"}
        assert scored["answer"].tolist() == raw["answer"].tolist()

    def test_regex_cleaner_extracts_the_answer(self, config, results, tmp_path, capsys):
        out = tmp_path / "scored.csv"
        pattern = r'answer:\s*"([^"]*)"'
        argv = ["postprocess", config, results, "-o", out, "--pattern", pattern, "-q"]
        assert cli_run(capsys, *argv) == (0, "", "")

        scored = pd.read_csv(out, dtype=str, keep_default_na=False)
        assert set(scored["cleaned_answer"]) == {"4. Agree a little"}
        assert set(scored["decision"]) == {"4. Agree a little"}

    def test_explicit_regex_cleaner(self, config, results, tmp_path, capsys):
        out = tmp_path / "scored.csv"
        argv = ["postprocess", config, results, "-o", out, "--cleaner", "regex", "--pattern", "(4)"]
        assert cli_run(capsys, *argv)[0] == 0
        assert set(pd.read_csv(out, dtype=str)["cleaned_answer"]) == {"4"}

    def test_several_files_and_globs(self, config, results, tmp_path, capsys):
        second = tmp_path / "second.csv"
        second.write_text(results.read_text("utf-8"), encoding="utf-8")
        out = tmp_path / "scored" / "all.csv"

        code, _, _ = cli_run(
            capsys, "postprocess", config, str(tmp_path / "*.csv"), "-o", out, "-q"
        )
        assert code == 0 and len(pd.read_csv(out)) == 2 * ITEMS * PERSONAS  # a new directory too

    def test_recursive_globs_match_like_the_pipeline_does(self, config, results, tmp_path, capsys):
        runs = tmp_path / "runs"
        runs.mkdir()
        (runs / "first.csv").write_text(results.read_text("utf-8"), encoding="utf-8")
        out = tmp_path / "scored.csv"

        # '**' also matches no directory at all: runs/first.csv
        code, _, _ = cli_run(
            capsys, "postprocess", config, str(runs / "**" / "*.csv"), "-o", out, "-q"
        )
        assert code == 0 and len(pd.read_csv(out)) == ITEMS * PERSONAS

    def test_quiet_hides_progress_and_summary(self, config, results, tmp_path, capsys):
        out = tmp_path / "scored.csv"
        assert cli_run(capsys, "postprocess", config, results, "-o", out, "-q") == (0, "", "")
        _, _, stderr = cli_run(capsys, "postprocess", config, results, "-o", out, "-f")
        assert "Cleaning" in stderr and f"wrote {out}" in stderr

    def test_existing_output_is_protected_unless_forced(self, config, results, tmp_path, capsys):
        out = tmp_path / "scored.csv"
        out.write_text("precious", encoding="utf-8")

        code, _, stderr = cli_run(capsys, "postprocess", config, results, "-o", out)
        assert code == 1 and "already exists" in stderr and out.read_text("utf-8") == "precious"
        assert cli_run(capsys, "postprocess", config, results, "-o", out, "--force", "-q")[0] == 0
        assert "decision" in out.read_text("utf-8")

    def test_no_model_is_ever_loaded(self, make_config, results, tmp_path, monkeypatch, capsys):
        eager = make_config("eager.json", parameters={"seeds": ["7"], "lazy_load_models": False})
        attempts: list[str] = []

        def load_model(self: LocalHuggingFaceModelConfig) -> None:
            attempts.append(self.name_or_path)
            raise AssertionError("a model must not be loaded to score answers")

        monkeypatch.setattr(LocalHuggingFaceModelConfig, "load_model", load_model)
        out = tmp_path / "scored.csv"
        code, _, stderr = cli_run(capsys, "postprocess", eager, results, "-o", out, "-q")
        assert attempts == [], stderr
        assert code == 0, stderr

    def test_the_results_are_never_overwritten(self, config, results, capsys):
        before = results.read_text("utf-8")
        code, _, stderr = cli_run(capsys, "postprocess", config, results, "-o", results, "-f")
        assert code == 1 and "is also a results file" in stderr
        assert results.read_text("utf-8") == before

    def test_results_that_do_not_exist(self, config, tmp_path, capsys):
        out = tmp_path / "o.csv"
        argv = ["postprocess", config, tmp_path / "a.csv", str(tmp_path / "b*.csv"), "-o", out]
        code, _, stderr = cli_run(capsys, *argv)
        assert code == 1 and "no results file matches:" in stderr and "b*.csv" in stderr

    def test_files_that_are_not_results(self, config, tmp_path, capsys):
        stray = tmp_path / "stray.csv"
        stray.write_text("a,b\n1,2\n", encoding="utf-8")
        code, _, stderr = cli_run(
            capsys, "postprocess", config, stray, "-o", tmp_path / "o.csv", "-q"
        )
        assert code == 1 and "lacks the column(s)" in stderr and "Traceback" not in stderr

    def test_an_invalid_configuration_is_reported(self, make_config, results, tmp_path, capsys):
        broken = make_config("broken.json", questionnaire=DROP)
        code, _, stderr = cli_run(capsys, "postprocess", broken, results, "-o", tmp_path / "o.csv")
        assert code == 1 and "no questionnaire" in stderr

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            (["--cleaner", "regex"], "--cleaner regex needs --pattern"),
            (
                ["--cleaner", "basic", "--pattern", "(x)"],
                "--pattern only applies to --cleaner regex",
            ),
            (["--pattern", "("], "invalid --pattern: missing )"),
            (["--pattern", "x"], "--pattern needs a capturing group"),
            (["-o", "scored.txt"], "unsupported output format '.txt'; use .csv"),
        ],
    )
    def test_invalid_options_are_usage_errors(
        self, config, results, tmp_path, capsys, options, message
    ):
        argv = ["postprocess", str(config), str(results), "-o", str(tmp_path / "o.csv"), *options]
        with pytest.raises(SystemExit) as exit_info:
            main(argv)
        assert exit_info.value.code == 2
        assert message in capsys.readouterr().err


# ============================== configurator ==============================


class TestConfigurator:
    def test_names_the_missing_extra(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "_has_module", lambda name: name != "streamlit")
        code, stdout, stderr = cli_run(capsys, "configurator")
        assert code == 1 and stdout == ""
        assert "pip install 'rupsycho[configurator]'" in stderr and "Streamlit" in stderr

    @pytest.mark.parametrize(("returned", "expected"), [(None, 0), (0, 0), (5, 5)])
    def test_delegates_to_the_launcher(self, monkeypatch, capsys, returned, expected):
        calls: list[list[str]] = []
        monkeypatch.setattr(cli, "_has_module", lambda name: True)
        monkeypatch.setattr(
            "rupsycho_configurator.launcher.main", lambda: calls.append(sys.argv[:]) or returned
        )
        before = sys.argv[:]

        assert cli_run(capsys, "configurator")[0] == expected
        assert calls == [["rup-configurator"]]  # not the arguments of this command
        assert sys.argv == before

    def test_arguments_after_a_double_dash_are_passed_to_streamlit(self, monkeypatch, capsys):
        calls: list[list[str]] = []
        monkeypatch.setattr(cli, "_has_module", lambda name: True)
        monkeypatch.setattr(
            "rupsycho_configurator.launcher.main", lambda: calls.append(sys.argv[:]) and 0
        )
        assert cli_run(capsys, "configurator", "--", "--server.port", "8502")[0] == 0
        assert calls == [["rup-configurator", "--server.port", "8502"]]

    def test_the_launcher_is_not_run_without_streamlit(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "_has_module", lambda name: False)
        monkeypatch.setattr(
            "rupsycho_configurator.launcher.main", lambda: pytest.fail("must not be called")
        )
        assert cli_run(capsys, "configurator")[0] == 1

    def test_module_detection(self):
        assert cli._has_module("json") and not cli._has_module("no_such_module_anywhere")


# ============================== entry points ==============================


@pytest.fixture(scope="module")
def processes() -> dict[str, subprocess.CompletedProcess]:
    """Run the entry points in real processes (concurrently: importing rupsycho takes seconds)."""
    scripts = Path(sys.executable).parent
    console_script = str(scripts / ("rupsycho.exe" if os.name == "nt" else "rupsycho"))
    commands = {
        "module-help": [sys.executable, "-m", "rupsycho", "--help"],
        "module-version": [sys.executable, "-m", "rupsycho", "--version"],
        "module-failure": [sys.executable, "-m", "rupsycho", "validate", "missing.json"],
        "module-success": [sys.executable, "-m", "rupsycho", "validate", str(CONFIG_PATH)],
        "script-version": [console_script, "--version"],
    }
    spawned = {
        name: subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=REPO_ROOT
        )
        for name, command in commands.items()
        if name != "script-version" or Path(console_script).exists()
    }
    results = {}
    for name, process in spawned.items():
        out, err = process.communicate(timeout=300)
        results[name] = subprocess.CompletedProcess(commands[name], process.returncode, out, err)
    return results


class TestEntryPoints:
    def test_python_dash_m_rupsycho(self, processes):
        result = processes["module-help"]
        assert result.returncode == 0 and result.stdout.startswith("usage: rupsycho ")
        assert all(command in result.stdout for command in COMMANDS)

    def test_version_flag(self, processes):
        assert processes["module-version"].stdout.strip() == f"rupsycho {rup.__version__}"

    def test_exit_codes_reach_the_shell(self, processes):
        assert processes["module-success"].returncode == 0
        assert processes["module-success"].stdout.startswith("OK  ")
        assert processes["module-failure"].returncode == 1
        assert processes["module-failure"].stdout == "ERROR missing.json: no such file\n"

    def test_console_script(self, processes):
        if "script-version" not in processes:
            pytest.skip("the rupsycho console script is not installed next to this interpreter")
        assert processes["script-version"].stdout.strip() == f"rupsycho {rup.__version__}"
