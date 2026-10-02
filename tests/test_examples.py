"""Tests of everything in ``examples/``: configurations, scripts, notebooks and sample results.

The examples are documentation that users copy, so they are held to the standard of the code:

* every configuration validates and has the size the documentation claims,
* every script compiles, the offline ones run (in a subprocess, like a user would start them)
  and the one that downloads a model is run against the tiny offline model of ``tiny_hf.py``,
* every notebook is valid, has no error output and uses only names that still exist; the
  quickstart is re-executed, and the notebooks that need a model run against a stand-in,
* the sample results are what the current callbacks and pipeline write, and
* the documentation pages list the files that exist and contain code that runs.

Nothing here needs the network. Everything that writes does so below ``tmp_path``; a module-wide
guard checks that ``examples/`` itself stays untouched. The slow parts (scripts in subprocesses,
notebooks in kernels) run in parallel threads, once per module.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import importlib.util
import json
import os
import py_compile
import re
import shlex
import shutil
import subprocess
import sys
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest
from langchain_core.language_models.fake import FakeListLLM

import rupsycho as rup
from rupsycho.models.model import LocalHuggingFaceModelConfig

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
DATA = EXAMPLES / "data"
SAMPLES = DATA / "output"
STUDY = EXAMPLES / "user_testing"
DOCS_PAGE = REPO / "docs" / "examples.md"
README = EXAMPLES / "README.md"

SCRIPTS = sorted(EXAMPLES.glob("*.py"))
NOTEBOOKS = sorted(EXAMPLES.rglob("*.ipynb"))
OWN_NOTEBOOKS = [path for path in NOTEBOOKS if path.parent == EXAMPLES]

# Scripts for maintainers; the documentation page does not need to list them
MAINTENANCE_SCRIPTS = {"make_sample_outputs.py", "execute_notebooks.py"}

# What to do when the callbacks, the pipeline or the API change what the examples show
REGENERATE = (
    "regenerate the stored examples with: python examples/make_sample_outputs.py && "
    "python examples/execute_notebooks.py --write"
)


# ============================== helpers ==============================


@contextlib.contextmanager
def _no_bytecode() -> Iterator[None]:
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous


def load_script(name: str) -> ModuleType:
    """Import ``examples/<name>`` as a module, without leaving bytecode in the repository."""
    path = EXAMPLES / name
    module_name = f"rupsycho_example_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses look their module up here
    with _no_bytecode():
        spec.loader.exec_module(module)
    return module


def run_script(
    name: str, *args: str, cwd: Path, timeout: float = 60
) -> subprocess.CompletedProcess[str]:
    """Run ``python examples/<name> <args>`` in ``cwd``, offline, and capture its output."""
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "HF_HUB_OFFLINE": "1",  # a script that tries to download fails instead of hanging
        "TRANSFORMERS_OFFLINE": "1",
    }
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )


def run_in_parallel(jobs: dict[str, Callable[[], Any]], workers: int = 3) -> dict[str, Any]:
    """Run the jobs in threads. A job that raises contributes its exception as the result."""

    def guarded(job: Callable[[], Any]) -> Any:
        try:
            return job()
        except Exception as error:
            return error

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {name: pool.submit(guarded, job) for name, job in jobs.items()}
        return {name: future.result() for name, future in futures.items()}


def result_of(outcome: Any) -> Any:
    """Return the result of a job of ``run_in_parallel`` or raise the exception it ended with."""
    if isinstance(outcome, BaseException):
        raise outcome
    return outcome


def snapshot(root: Path) -> dict[str, tuple[int, int]]:
    """Path -> (size, modification time) of every file below ``root``."""
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(autouse=True, scope="module")
def examples_folder_stays_untouched() -> Iterator[None]:
    """Fail the module if any test wrote into (or deleted from) ``examples/``."""
    before = snapshot(EXAMPLES)
    yield
    assert snapshot(EXAMPLES) == before


@pytest.fixture
def scratch_examples(tmp_path: Path) -> Path:
    """A private copy of ``examples/`` in which notebooks and scripts may write files."""
    root = tmp_path / "examples"
    shutil.copytree(EXAMPLES, root, ignore=shutil.ignore_patterns("__pycache__"))
    return root


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


# ============================== (a) configurations ==============================

# file -> (questions, personas, models, seeds); the number of model calls is their product
EXPECTED_SIZES = {
    "bdi_qwen72.json": (21, 250, 1, 1),
    "bfi_demo_config.json": (5, 2, 1, 1),
    "bfi_small_and_mid.json": (44, 250, 5, 1),
    "example_config_for_demographics_judge.json": (2, 1, 1, 3),
    "gsdb_new_qwen.json": (23, 250, 1, 1),
    "rfq_json_format_small.json": (11, 250, 1, 1),
    "rfq_model_friendly_small.json": (11, 250, 1, 1),
    "rfq_natural_language_small.json": (11, 250, 1, 1),
    "trolley_qwen72.json": (3, 250, 1, 1),
    "example-questionnaire.json": (2, 2, 1, 1),  # user study
}
CONFIGS = sorted(DATA.glob("*.json")) + sorted((STUDY / "data").glob("*.json"))


def test_every_configuration_is_covered_by_the_expected_sizes():
    """A new or removed file must be reflected in the table above (and in the documentation)."""
    assert {path.name for path in CONFIGS} == set(EXPECTED_SIZES)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.name)
def test_configuration_validates_and_has_the_documented_size(path: Path):
    experiment = rup.experiment_from_dict(json.loads(path.read_text(encoding="utf-8")))

    questions, personas, models, seeds = EXPECTED_SIZES[path.name]
    assert experiment.questionnaire.get_number_of_questions() == questions
    assert len(experiment.demographic_profiles) == personas
    assert len(experiment.models) == models
    assert len(experiment.parameters.seeds) == seeds
    # Validating must never load (download) a model
    assert experiment.parameters.lazy_load_models
    assert all(
        experiment.runnable_models[key] is experiment.models[key] for key in experiment.models
    )


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.name)
def test_prompt_of_the_first_and_last_cell_of_the_grid_assembles(path: Path, capsys):
    experiment = rup.experiment_from_file(path)
    questions = experiment.questionnaire.get_number_of_questions()
    personas = len(experiment.demographic_profiles)

    for item, persona in {(0, 0), (questions - 1, personas - 1)}:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a prompt that cannot be assembled only warns
            experiment.print_assembled_prompt(item_idx=item, persona_idx=persona)
        printed = capsys.readouterr().out
        assert experiment.questionnaire.instruction_items[item].question in printed


# ============================== (b) the bundled twin ==============================


def test_bundled_example_is_the_twin_of_the_demo_configuration():
    """``bfi_demo_config.json`` is the readable copy of the example that ships in the wheel.

    The notebooks, the documentation and the packaged ``load_example_experiment("bfi")`` all
    talk about the same experiment, so name, questionnaire, personas, prompt and seeds must be
    identical. Two parts may differ without making them different experiments: the free-text
    ``description`` (the packaged one speaks about being bundled) and the ``models`` section (the
    packaged example is free to point to another default model than the file in the repository).
    """
    bundled = rup.load_example_config("bfi")
    demo = json.loads((DATA / "bfi_demo_config.json").read_text(encoding="utf-8"))

    free = {"description", "models"}
    assert {k: v for k, v in bundled.items() if k not in free} == {
        k: v for k, v in demo.items() if k not in free
    }
    assert bundled["models"] and demo["models"]
    rup.experiment_from_dict(bundled)  # both variants are valid experiments


def test_prose_names_the_model_of_the_demo_configuration():
    """Notebooks and docs tell users which model they will download; keep that in sync."""
    demo = json.loads((DATA / "bfi_demo_config.json").read_text(encoding="utf-8"))
    (section,) = demo["models"].values()
    model = section["name_or_path"]
    prose = {
        "docs/examples.md": DOCS_PAGE.read_text(encoding="utf-8"),
        "examples/run_experiment.py": (EXAMPLES / "run_experiment.py").read_text(encoding="utf-8"),
        "examples/Basics.ipynb": "\n".join(markdown_cells(EXAMPLES / "Basics.ipynb")),
        "examples/Callbacks.ipynb": "\n".join(markdown_cells(EXAMPLES / "Callbacks.ipynb")),
    }
    for name, text in prose.items():
        assert model in text, f"{name} should name the model {model} of the demo configuration"
        assert "SmolLM" not in text, f"{name} still mentions the former demo model"


# ============================== (c) scripts ==============================


def test_scripts_exist():
    assert {path.name for path in SCRIPTS} >= {
        "run_experiment.py",
        "reproducibility_check.py",
        "custom_callback.py",
        "api_models.py",
    }


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_script_compiles_and_has_a_module_docstring(script: Path, tmp_path: Path):
    py_compile.compile(str(script), cfile=str(tmp_path / "script.pyc"), doraise=True)
    docstring = ast.get_docstring(ast.parse(script.read_text(encoding="utf-8")))
    assert docstring and len(docstring) > 200, f"{script.name} needs a descriptive docstring"


# script -> (arguments, text its output must contain). "{tmp}" is replaced by a scratch folder.
# run_experiment.py without --dry-run is the one way to start a script that needs the network.
OFFLINE_SCRIPTS: dict[str, tuple[list[str], list[str]]] = {
    "api_models.py": ([], ["nothing is sent over the network", "All configuration snippets"]),
    "custom_callback.py": (
        [],
        ["model calls: 10, failed: 2", "after a second run  : 10 (replaced)", "NULL answer"],
    ),
    "reproducibility_check.py": ([], ["All checks passed."]),
    "make_sample_outputs.py": (["--output-dir", "{tmp}/out"], ["bfi_demo_config_output.csv"]),
    "run_experiment.py": (["--dry-run"], ["Dry run: no model was loaded", "= 10"]),
    "execute_notebooks.py": (["--help"], ["skip-execution"]),
}


def test_every_script_is_classified():
    assert {path.name for path in SCRIPTS} == set(OFFLINE_SCRIPTS)


@pytest.fixture(scope="module")
def script_runs(tmp_path_factory) -> dict[str, tuple[Any, Path]]:
    """Every offline script, started the way a user would, all at once."""
    jobs, folders = {}, {}
    for name, (args, _) in OFFLINE_SCRIPTS.items():
        folders[name] = tmp_path_factory.mktemp(Path(name).stem)
        arguments = [argument.format(tmp=folders[name]) for argument in args]
        jobs[name] = partial(run_script, name, *arguments, cwd=folders[name], timeout=60)
    outcomes = run_in_parallel(jobs)
    return {name: (outcomes[name], folders[name]) for name in jobs}


@pytest.mark.parametrize("name", sorted(OFFLINE_SCRIPTS))
def test_offline_script_runs(name: str, script_runs):
    if name == "execute_notebooks.py":
        pytest.importorskip("nbclient")  # the script cannot even start without the dev tools
        pytest.importorskip("nbformat")
    outcome, folder = script_runs[name]
    result = result_of(outcome)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    for text in OFFLINE_SCRIPTS[name][1]:
        assert text in result.stdout, f"{text!r} missing from the output of {name}"
    # Nothing but the explicitly requested output directory is written
    assert not [path for path in folder.iterdir() if path.suffix in {".csv", ".jsonl", ".db"}]


def test_run_experiment_streams_the_answers_of_a_local_model(
    tiny_hf_dir: Path, tmp_path: Path, monkeypatch, capsys
):
    """The script that downloads a model, pointed at the tiny local model instead."""
    args = ["--model", str(tiny_hf_dir), "--max-new-tokens", "6"]

    result = run_script("run_experiment.py", *args, cwd=tmp_path, timeout=60)

    assert result.returncode == 0, result.stdout + result.stderr
    answers = _csv(tmp_path / "bfi_answers.csv")
    assert len(answers) == 10  # 5 questions x 2 personas x 1 seed
    assert set(answers["model_id"]) == {tiny_hf_dir.name}
    assert "10 model calls" in result.stdout

    # A second run would append to the file: the script refuses (in-process, no second start)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        load_script("run_experiment.py").main(args)
    assert exit_info.value.code == 2
    assert "already exists" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "no such file"),
        ("{", "invalid configuration"),  # not JSON
        ('{"questionnaire": 3}', "invalid configuration"),  # violates the schema
        ('{"models": 3}', "invalid configuration"),  # wrong type of a whole section
    ],
    ids=["missing", "not-json", "schema", "section-type"],
)
def test_run_experiment_reports_bad_input_as_a_usage_error(
    content: str | None, message: str, tmp_path: Path, capsys
):
    script = load_script("run_experiment.py")
    config = tmp_path / "config.json"
    if content is not None:
        config.write_text(content, encoding="utf-8")

    with pytest.raises(SystemExit) as exit_info:
        script.main([str(config), "--dry-run"])

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_run_experiment_applies_its_overrides_without_loading_a_model():
    script = load_script("run_experiment.py")

    experiment = script.build_experiment(
        None, model="some/model", device="cpu", max_new_tokens=5, seeds=["1", "2"]
    )

    section = experiment.models["some/model"]
    assert section.type == "local_huggingface" and section.name_or_path == "some/model"
    assert section.parameters["max_new_tokens"] == 5
    assert experiment.parameters.seeds == ["1", "2"]
    assert experiment.runnable_models["some/model"] is section  # still only configured


def test_api_models_snippets_are_valid_configurations():
    module = load_script("api_models.py")
    assert set(module.PROVIDERS) == {"openai", "ollama", "google", "deepseek"}
    for provider in module.PROVIDERS.values():
        experiment = rup.experiment_from_dict(module.configuration_with(provider))
        assert list(experiment.models) == [provider.identifier]
        assert experiment.models[provider.identifier].type == provider.model["type"]
        assert provider.identifier in provider.code  # the code style uses the same identifier


def test_api_models_run_explains_what_is_missing(monkeypatch, capsys):
    module = load_script("api_models.py")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert module.main(["--run", "openai"]) == 2  # 2: cannot run
    assert "OPENAI_API_KEY" in capsys.readouterr().out


def test_api_models_run_uses_the_configured_model(monkeypatch, capsys):
    """``--run`` end to end with the model loader replaced (no key, no network)."""
    module = load_script("api_models.py")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")
    monkeypatch.setattr(module, "installed", lambda provider: True)
    loaded = []

    def load_model(config):
        loaded.append(config)
        return FakeListLLM(responses=["3. Neither agree nor disagree"])

    monkeypatch.setattr("rupsycho.models.model.DeepSeekModelConfig.load_model", load_model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        status = module.main(["--run", "deepseek", "--max-concurrency", "2"])

    assert status == 0 and len(loaded) == 1
    assert loaded[0].name_or_path == "deepseek-chat"
    assert capsys.readouterr().out.count("3. Neither agree nor disagree") == 10


def test_custom_callback_stores_failed_calls_as_null_and_replaces_rows(tmp_path: Path):
    module = load_script("custom_callback.py")
    database = tmp_path / "answers.db"
    experiment = rup.load_example_experiment("bfi", models={})
    experiment.add_model(module.ScriptedLLM(fail_on="is depressed, blue"), identifier="scripted")

    with module.SQLiteCallback(database) as sink:
        for _ in range(2):  # the second run replaces the rows of the first
            summary = experiment.run(callbacks=[sink], on_error="ignore", show_progress=False)
        rows = sink.connection.execute(
            "SELECT item_id, persona_id, answer FROM answers ORDER BY item_id, persona_id"
        ).fetchall()

    assert summary.n_failed == 2
    assert len(rows) == 10
    assert [row[2] is None for row in rows] == [False] * 6 + [True] * 2 + [False] * 2
    assert all(row[0] == 3 for row in rows if row[2] is None)


# ============================== sample results ==============================


def test_sample_results_are_what_the_current_callbacks_write(tmp_path: Path):
    """Regenerate the samples and compare; only the measured ``time`` may differ."""
    script = load_script("make_sample_outputs.py")
    csv_path, jsonl_path = script.generate(tmp_path)

    shipped, fresh = _csv(SAMPLES / csv_path.name), _csv(csv_path)
    assert list(shipped.columns) == list(fresh.columns), REGENERATE
    assert shipped.drop(columns="time").equals(fresh.drop(columns="time")), REGENERATE

    def rows(path: Path) -> list[dict[str, Any]]:
        parsed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return [{k: v for k, v in row.items() if k != "time"} for row in parsed]

    assert rows(SAMPLES / jsonl_path.name) == rows(jsonl_path), REGENERATE
    assert all("answers" not in row["instruction_item"] for row in rows(jsonl_path))


def test_processed_sample_is_what_the_current_pipeline_writes(tmp_path: Path):
    from rupsycho.parsers.cleaners import RegexExtractorCleaner
    from rupsycho.parsers.judges import MultipleChoiceJudge
    from rupsycho.parsers.validators import ValidatorParser
    from rupsycho.postprocessing import PostprocessingPipeline

    pipeline = PostprocessingPipeline(
        DATA / "bfi_demo_config.json",
        str(SAMPLES / "bfi_demo_config_output.csv"),
        cleaner=RegexExtractorCleaner(pattern=r'answer\s*:\s*"([^"]+)"'),
        validator=ValidatorParser(),
        judge=MultipleChoiceJudge(possible_answers=[]),
        output_path=tmp_path / "processed.csv",
        show_progress=False,
    )
    pipeline.run()

    shipped = _csv(SAMPLES / "bfi_demo_config_output_processed.csv")
    fresh = _csv(tmp_path / "processed.csv")
    assert list(shipped.columns) == list(fresh.columns), REGENERATE
    assert shipped.equals(fresh), REGENERATE
    # The sample is meant to show every outcome of the three stages
    assert set(shipped["valid"]) == {"True", "False"}
    assert {"not present", "inconclusive"} <= set(shipped["decision"])


# ============================== (d) notebooks: validity ==============================


@pytest.fixture(scope="module")
def nbformat():
    return pytest.importorskip("nbformat")


def error_outputs(notebook: Any) -> list[str]:
    problems = []
    for index, cell in enumerate(notebook.cells):
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                problems.append(f"cell {index}: {output.ename}: {output.evalue}")
            elif output.output_type == "stream" and "Traceback (most recent" in output.text:
                problems.append(f"cell {index}: traceback in {output.name}")
    return problems


def stdout_of(notebook: Any) -> str:
    """All printed text and plain-text results of a notebook."""
    texts = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream" and output.name == "stdout":
                texts.append(output.text)
            elif output.output_type in {"execute_result", "display_data"}:
                texts.append(output.data.get("text/plain", ""))
    return "\n".join(texts)


def printed_by_cell(notebook: Any) -> dict[str, str]:
    """Cell id -> text printed to stdout (what ``print`` wrote, nothing library dependent).

    Lines that report a measured speed ("... faster: True") are left out: they depend on how busy
    the machine is, unlike everything else the notebooks print.
    """
    return {
        cell.id: "".join(
            line
            for output in cell.outputs
            if output.output_type == "stream" and output.name == "stdout"
            for line in output.text.splitlines(keepends=True)
            if "faster" not in line
        )
        for cell in notebook.cells
        if cell.cell_type == "code"
    }


def assert_printed_text_is_current(path: Path, executed: Any, nbformat: Any) -> None:
    """The stored ``print`` output of every cell equals what the cell prints today."""
    stored = printed_by_cell(nbformat.read(path, as_version=4))
    fresh = printed_by_cell(executed)
    stale = [cell_id for cell_id, text in stored.items() if text and fresh.get(cell_id) != text]
    assert stale == [], f"{path.name}: stored output out of date in cells {stale}; {REGENERATE}"


def test_notebooks_exist():
    names = {path.name for path in OWN_NOTEBOOKS}
    assert names == {
        "quickstart.ipynb",
        "Basics.ipynb",
        "Callbacks.ipynb",
        "Parsers.ipynb",
        "Postprocessing.ipynb",
    }


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: str(path.relative_to(EXAMPLES)))
def test_notebook_is_valid_and_has_no_error_output(path: Path, nbformat):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert error_outputs(notebook) == []


@pytest.mark.parametrize("path", OWN_NOTEBOOKS, ids=lambda path: path.name)
def test_stored_notebook_is_clean_and_deterministic(path: Path, nbformat):
    notebook = nbformat.read(path, as_version=4)
    raw = path.read_text(encoding="utf-8")

    assert notebook.metadata.kernelspec.name == "python3"
    assert not re.search(r"/Users/|/home/|/private/|/tmp/", raw), "absolute path"
    assert "application/vnd.jupyter.widget" not in raw, "widget state is not reproducible"
    assert len(raw) < 100_000, "keep the stored outputs small"
    for cell in notebook.cells:
        assert "execution" not in cell.metadata, "execution timings change on every run"
        if cell.cell_type != "code":
            continue
        if "skip-execution" in cell.metadata.get("tags", []):
            # Cells that need a model download carry no (stale or broken) output
            assert cell.outputs == [] and cell.execution_count is None
        elif cell.source.strip():
            assert cell.execution_count is not None, f"cell {cell.id} was never executed"


def test_cells_that_load_models_are_marked_as_not_executed(nbformat):
    """The model cells of the notebooks that need a download are tagged, not silently stale."""
    for name, expected in {"Basics": 3, "Callbacks": 1, "Parsers": 2, "quickstart": 0}.items():
        notebook = nbformat.read(EXAMPLES / f"{name}.ipynb", as_version=4)
        tagged = [c for c in notebook.cells if "skip-execution" in c.metadata.get("tags", [])]
        assert len(tagged) == expected, name


# ============================== (d) notebooks and scripts: current API ==============================

BANNED = {
    r"\bexample_experiment_bfi\b": "deprecated, use load_example_experiment('bfi', models={})",
    r"\brup(sycho)?\.parser\b": "deprecated alias of rupsycho.parsers.parser",
    r"\btqdm\.pandas\b": "global tqdm hack, use show_progress",
}
PATH_LITERAL = re.compile(r"^(?:\./)?(?:examples/)?data/[\w./-]+\.(?:json|csv|jsonl)$")


def code_of(path: Path) -> list[str]:
    """The Python sources of a script or the code cells of a notebook (magics removed)."""
    if path.suffix == ".py":
        return [path.read_text(encoding="utf-8")]
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    sources = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    return [re.sub(r"(?m)^\s*[%!].*$", "pass", source) for source in sources]


def api_problems(source: str) -> list[str]:
    """Names imported from or accessed on ``rupsycho`` that do not exist (any more)."""
    tree = ast.parse(source)
    bound: dict[str, Any] = {}
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "rupsycho":
                    try:
                        module = importlib.import_module(alias.name)
                    except ImportError as error:
                        problems.append(f"import {alias.name}: {error}")
                        continue
                    bound[alias.asname or "rupsycho"] = module if alias.asname else rup
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "rupsycho":
            try:
                module = importlib.import_module(node.module or "")
            except ImportError as error:
                problems.append(f"from {node.module} import ...: {error}")
                continue
            for alias in node.names:
                try:
                    value = getattr(module, alias.name)
                except AttributeError:
                    try:
                        value = importlib.import_module(f"{node.module}.{alias.name}")
                    except ImportError:
                        problems.append(f"from {node.module} import {alias.name}: no such name")
                        continue
                bound[alias.asname or alias.name] = value

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain, current = [], node
        while isinstance(current, ast.Attribute):
            chain.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name) or current.id not in bound:
            continue
        value, path = bound[current.id], [current.id]
        for attribute in reversed(chain):
            path.append(attribute)
            try:
                value = getattr(value, attribute)
            except AttributeError:
                problems.append(f"{'.'.join(path)} does not exist")
                break
    return problems


def path_problems(source: str, folder: Path) -> list[str]:
    """String literals that point to files below ``data/`` which do not exist.

    Paths are looked up relative to ``folder`` (where the notebook lives), to ``examples/`` and
    to the repository root, because notebooks are run from the first two and docs from the last.
    """
    problems = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if PATH_LITERAL.match(node.value):
                relative = node.value.removeprefix("./")
                if not any((root / relative).exists() for root in (folder, EXAMPLES, REPO)):
                    problems.append(f"{node.value} does not exist")
    return problems


@pytest.mark.parametrize(
    "path", [*NOTEBOOKS, *SCRIPTS], ids=lambda path: str(path.relative_to(EXAMPLES))
)
def test_examples_use_only_current_api_and_existing_files(path: Path):
    sources = code_of(path)
    problems = [
        problem
        for source in sources
        for problem in api_problems(source) + path_problems(source, path.parent)
    ]
    for pattern, reason in BANNED.items():
        problems += [f"{pattern}: {reason}" for source in sources if re.search(pattern, source)]
    assert problems == []


def test_the_api_checker_catches_what_it_is_meant_to_catch():
    """Guard the guard: removed names, missing attributes and missing files are reported."""
    source = "\n".join(
        [
            "import rupsycho as rup",
            "from rupsycho import seeding, no_such_name",
            "from rupsycho.parsers.cleaners import BasicCleaner, NoSuchCleaner",
            "rup.example_removed()",
            "seeding.seed_model(None, 1); seeding.nope",
            "path = 'data/nope.json'",
        ]
    )
    found = api_problems(source) + path_problems(source, EXAMPLES)
    assert any("no_such_name" in p for p in found)
    assert any("NoSuchCleaner" in p for p in found)
    assert any("rup.example_removed" in p for p in found)
    assert any("seeding.nope" in p for p in found)
    assert any("data/nope.json" in p for p in found)
    assert not any("seed_model" in p or "BasicCleaner" in p for p in found)


# ============================== (e) notebooks: execution ==============================

QUICKSTART_CLAIMS = [
    "model calls: 10",
    "failed     : 0",
    "5 of 10 calls failed",
    "identical answers, same order    : True",
    "same seeds      -> identical answers : True",
    "different seeds -> different answers : True",
    "supports seeding : False",
    "identical answers: False",
    "not present",
]

# A first cell for the notebooks that load a real model: make the loader return a stand-in
# (and the console progress bar, because the notebook variant needs ipywidgets)
STAND_IN = """
from langchain_core.language_models.fake import FakeListLLM
from tqdm import tqdm

import rupsycho.mixins.experiment_processing as processing
import rupsycho.models.model as models

models.LocalHuggingFaceModelConfig.load_model = lambda self: FakeListLLM(
    responses=['{answer: "4. Agree a little"}']
)
processing.tqdm = tqdm
"""


@pytest.fixture(scope="module")
def runner(nbformat):
    pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    return load_script("execute_notebooks.py")


@dataclass
class Execution:
    """A notebook that was executed in a private copy of ``examples/``."""

    notebook: Any
    root: Path


@pytest.fixture(scope="module")
def executions(runner, tmp_path_factory) -> dict[str, Any]:
    """Execute five notebooks at once, each in its own copy of ``examples/``.

    The two notebooks that load a real model run completely (``skip_tag=None``) against a
    stand-in model; the other three skip their tagged cells like the stored versions do.
    """
    kernel_tmp = tmp_path_factory.mktemp("kernel-tmp")

    def execute(name: str, **options: Any) -> Execution:
        root = tmp_path_factory.mktemp(name) / "examples"
        shutil.copytree(EXAMPLES, root, ignore=shutil.ignore_patterns("__pycache__"))
        notebook = runner.execute(root / f"{name}.ipynb", timeout=180, **options)
        return Execution(notebook, root)

    stand_in = {"skip_tag": None, "setup_code": STAND_IN}
    jobs = {
        "quickstart": partial(execute, "quickstart"),
        "Parsers": partial(execute, "Parsers"),
        "Postprocessing": partial(execute, "Postprocessing"),
        "Basics": partial(execute, "Basics", **stand_in),
        "Callbacks": partial(execute, "Callbacks", **stand_in),
    }
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TMPDIR", str(kernel_tmp))  # inherited by the kernels
        outcomes = run_in_parallel(jobs)
    outcomes["kernel_tmp"] = kernel_tmp
    return outcomes


def test_quickstart_claims_hold_in_the_stored_notebook(nbformat):
    text = stdout_of(nbformat.read(EXAMPLES / "quickstart.ipynb", as_version=4))
    for claim in QUICKSTART_CLAIMS:
        assert claim in text, claim


def test_quickstart_re_executes_offline(executions, nbformat):
    """The quickstart runs without downloads and errors, removes its scratch folder, and the
    stored outputs are what the code prints today."""
    run = result_of(executions["quickstart"])

    assert error_outputs(run.notebook) == []
    text = stdout_of(run.notebook)
    for claim in QUICKSTART_CLAIMS:
        assert claim in text, claim
    assert list(executions["kernel_tmp"].glob("rupsycho-quickstart-*")) == []
    assert_printed_text_is_current(EXAMPLES / "quickstart.ipynb", run.notebook, nbformat)


def test_parsers_and_postprocessing_re_execute_offline(executions, nbformat):
    for name in ("Parsers", "Postprocessing"):
        run = result_of(executions[name])
        assert error_outputs(run.notebook) == [], name
        assert_printed_text_is_current(EXAMPLES / f"{name}.ipynb", run.notebook, nbformat)
    # The notebook rewrote its result file in the copy; it equals the shipped one
    run = result_of(executions["Postprocessing"])
    copied = run.root / "data" / "output" / "bfi_demo_config_output_processed.csv"
    assert copied.read_bytes() == (SAMPLES / copied.name).read_bytes(), REGENERATE


def test_basics_notebook_runs_against_a_stand_in_model(executions, nbformat):
    run = result_of(executions["Basics"])

    assert error_outputs(run.notebook) == []
    assert_printed_text_is_current(EXAMPLES / "Basics.ipynb", run.notebook, nbformat)
    answers = pd.read_csv(run.root / "output" / "basics_answers.csv", dtype=str)
    assert len(answers) == 10 and set(answers["Answer"]) == {'{answer: "4. Agree a little"}'}
    exported = rup.experiment_from_file(run.root / "output" / "basics_experiment.json")
    assert exported.questionnaire.instruction_items[0].answers


def test_callbacks_notebook_runs_against_a_stand_in_model(executions, nbformat):
    run = result_of(executions["Callbacks"])

    assert error_outputs(run.notebook) == []
    assert_printed_text_is_current(EXAMPLES / "Callbacks.ipynb", run.notebook, nbformat)
    output = run.root / "output"
    assert len(_csv(output / "callbacks_answers.csv")) == 10
    assert len((output / "callbacks_answers.jsonl").read_text(encoding="utf-8").splitlines()) == 10
    text = "".join(o.get("text", "") for c in run.notebook.cells for o in c.get("outputs", []))
    assert "Instruction ID" in text and "10 model calls" in text  # the table and the summary


def test_executor_skips_tagged_cells_and_runs_the_setup_cell(runner, tmp_path: Path, nbformat):
    notebook = nbformat.v4.new_notebook()
    skipped = nbformat.v4.new_code_cell("raise RuntimeError('would download a model')")
    skipped.metadata["tags"] = ["skip-execution"]
    notebook.cells = [
        nbformat.v4.new_code_cell("print('a', end='')"),
        skipped,
        nbformat.v4.new_code_cell("print('b', value)"),
    ]
    path = tmp_path / "tiny.ipynb"
    nbformat.write(notebook, path)

    executed = runner.execute(path, setup_code="value = 41 + 1")

    assert len(executed.cells) == 3  # the setup cell is not kept
    assert executed.cells[1].outputs == [] and executed.cells[1].execution_count is None
    assert stdout_of(executed).replace("\n", "") == "ab 42"
    assert "execution" not in executed.cells[0].metadata
    assert executed.metadata.kernelspec.name == "python3"


def test_executor_reports_every_failing_notebook_and_changes_nothing(
    runner, tmp_path: Path, nbformat, monkeypatch, capsys
):
    folder = tmp_path / "folder"
    folder.mkdir()
    for name, source in (("good.ipynb", "print('fine')"), ("bad.ipynb", "1 / 0")):
        notebook = nbformat.v4.new_notebook()
        notebook.cells = [nbformat.v4.new_code_cell(source)]
        nbformat.write(notebook, folder / name)
    before = snapshot(folder)
    monkeypatch.setattr(runner, "EXAMPLES_DIR", folder)

    status = runner.main([])

    printed = capsys.readouterr().out
    assert status == 1
    assert "ok      good.ipynb" in printed
    assert "FAILED  bad.ipynb: ZeroDivisionError: division by zero" in printed
    assert snapshot(folder) == before  # without --write the notebooks stay as they were


def test_executor_merges_stream_chunks_but_not_different_streams(runner, nbformat):
    outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="a"),
        nbformat.v4.new_output("stream", name="stdout", text="b\n"),
        nbformat.v4.new_output("stream", name="stderr", text="warning\n"),
        nbformat.v4.new_output("stream", name="stdout", text="c\n"),
    ]
    merged = runner.coalesce_streams(outputs)
    assert [(o.name, o.text) for o in merged] == [
        ("stdout", "ab\n"),
        ("stderr", "warning\n"),
        ("stdout", "c\n"),
    ]


# ============================== documentation ==============================


def markdown_cells(path: Path) -> list[str]:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"]


def shell_commands(markdown: str, program: str) -> list[str]:
    """The commands of the fenced ``bash`` blocks that start with ``program``."""
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", markdown, flags=re.S):
        for line in block.replace("\\\n", " ").splitlines():
            if line.strip().startswith(program):
                commands.append(line.strip())
    return commands


def test_cli_commands_of_the_notebooks_are_accepted_by_the_parser():
    cli = pytest.importorskip("rupsycho.cli")
    parser = cli.build_parser()
    commands = []
    for notebook in ("quickstart.ipynb", "Postprocessing.ipynb"):
        for cell in markdown_cells(EXAMPLES / notebook):
            commands += shell_commands(cell, "rupsycho ")

    assert len(commands) >= 6
    for command in commands:
        words = shlex.split(command, comments=True)
        parser.parse_args(words[1:])  # exits with an error message if an option is wrong


def test_postprocess_command_of_the_notebook_reproduces_the_sample(
    tmp_path: Path, scratch_examples: Path, monkeypatch
):
    """``rupsycho postprocess`` with the arguments shown in Postprocessing.ipynb."""
    cli = pytest.importorskip("rupsycho.cli")
    (command,) = [
        c
        for cell in markdown_cells(EXAMPLES / "Postprocessing.ipynb")
        for c in shell_commands(cell, "rupsycho postprocess")
    ]
    words = shlex.split(command, comments=True)[1:]
    words[words.index("-o") + 1] = str(tmp_path / "processed.csv")  # write into the scratch folder
    monkeypatch.chdir(scratch_examples)  # the command uses paths relative to examples/

    assert cli.main([*words, "-q"]) == 0

    produced = _csv(tmp_path / "processed.csv")
    shipped = _csv(SAMPLES / "bfi_demo_config_output_processed.csv")
    assert list(produced["decision"]) == list(shipped["decision"])
    assert list(produced["cleaned_answer"]) == list(shipped["cleaned_answer"])


def documented_sizes(page: Path) -> dict[str, tuple[int, ...]]:
    """``file -> (questions, personas, models, model calls)`` from the tables of a page."""
    sizes: dict[str, tuple[int, ...]] = {}
    for line in page.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        files = re.findall(r"`([\w.\-]+\.json)`", cells[0]) if line.startswith("|") else []
        if files:
            numbers = tuple(int(re.sub(r"\D", "", cell)) for cell in cells[-4:])
            sizes.update(dict.fromkeys(files, numbers))
    return sizes


PAGES = pytest.mark.parametrize(
    "page", [DOCS_PAGE, README], ids=["docs/examples.md", "examples/README.md"]
)


@PAGES
def test_documented_sizes_of_the_configurations_are_right(page: Path):
    expected = {
        name: (questions, personas, models, questions * personas * models * seeds)
        for name, (questions, personas, models, seeds) in EXPECTED_SIZES.items()
        if name != "example-questionnaire.json"  # the user study is not part of the tables
    }
    assert documented_sizes(page) == expected


@PAGES
def test_documentation_lists_every_notebook_and_script(page: Path):
    text = page.read_text(encoding="utf-8")
    required = [p.name for p in OWN_NOTEBOOKS] + [
        p.name for p in SCRIPTS if page == README or p.name not in MAINTENANCE_SCRIPTS
    ]
    assert [name for name in required if name not in text] == []


@pytest.mark.parametrize("path", OWN_NOTEBOOKS, ids=lambda path: path.name)
def test_links_between_the_notebooks_and_files_exist(path: Path):
    for cell in markdown_cells(path):
        for target in re.findall(r"\]\((?!https?:|#)([^)#]+)\)", cell):
            assert (path.parent / target).exists(), f"{path.name} links to missing {target}"


def test_links_of_the_documentation_point_to_existing_files():
    docs = DOCS_PAGE.read_text(encoding="utf-8")
    for kind, target in re.findall(
        r"github\.com/julianschelb/rupsycho/(blob|tree)/main/([^)\s]+)", docs
    ):
        assert (REPO / target).exists(), f"{kind}/{target}"
    for target in re.findall(r"\]\((?!https?:|#)([^)#]+)\)", docs):
        assert (DOCS_PAGE.parent / target).exists(), target

    readme = README.read_text(encoding="utf-8")
    for target in re.findall(r"\]\((?!https?:|#)([^)#]+)\)", readme):
        assert (EXAMPLES / target).exists(), target


def test_minimal_end_to_end_script_of_the_documentation_runs(tmp_path, monkeypatch):
    """Run the code block of docs/examples.md with the real model replaced by a stand-in."""
    page = DOCS_PAGE.read_text(encoding="utf-8")
    block = re.search(r"## Minimal end-to-end script\s+```python\n(.*?)```", page, re.S)
    assert block is not None
    config = repr(str(DATA / "bfi_demo_config.json"))
    code = block.group(1).replace('"examples/data/bfi_demo_config.json"', config)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        LocalHuggingFaceModelConfig,
        "load_model",
        lambda self: FakeListLLM(responses=['{answer: "4. Agree a little"}', "3"]),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(code, "docs/examples.md", "exec"), {"__name__": "docs_example"})

    results = _csv(tmp_path / "results.csv")
    processed = _csv(tmp_path / "processed.csv")
    assert len(results) == len(processed) == 10
    assert {"cleaned_answer", "valid", "decision"} <= set(processed.columns)
