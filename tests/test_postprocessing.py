"""End-to-end tests for ``rupsycho.postprocessing.PostprocessingPipeline``.

Every test produces its input the way a user does: a fake-LLM experiment is run with a real
``CSVCallback`` and the resulting CSV file is post-processed with real cleaners, validators and
judges. Nothing is downloaded.

The pipeline adds the columns ``cleaned_answer``, ``validation_status`` (the validator's dictionary,
stored by ``to_csv`` as its repr), ``valid`` (a boolean shortcut for it) and ``decision``. Missing
answers (the blank cell of a failed model call) give ``None`` in all of them. Problems are logged
(logger ``rupsycho.postprocessing``) instead of printed.

Tests marked ``xfail(strict=True)`` describe the *correct* behaviour of a known defect: they fail
today and turn into a loud XPASS as soon as the defect is fixed, at which point the marker has to
be removed.
"""

import ast
import copy
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake import FakeListLLM

import rupsycho as rup
import rupsycho.postprocessing as postprocessing_module
from rupsycho.callbacks import CSVCallback
from rupsycho.experiment import ExperimentDocument
from rupsycho.parsers.cleaners import BasicCleaner, PromptRemovalCleaner, RegexExtractorCleaner
from rupsycho.parsers.judges import DemographicsJudge, MultipleChoiceJudge
from rupsycho.parsers.validators import ValidatorParser
from rupsycho.postprocessing import REQUIRED_COLUMNS, PostprocessingPipeline

from .conftest import CONFIG_PATH

LOGGER = "rupsycho.postprocessing"
BFI_OPTIONS = [
    "1. Disagree strongly",
    "2. Disagree a little",
    "3. Neither agree nor disagree",
    "4. Agree a little",
    "5. Agree strongly",
]
RAW_COLUMNS = [
    "experiment_name",
    "instruction_item_id",
    "instruction_item",
    "model_id",
    "profile_id",
    "random_seed",
    "time",
    "answer",
]
ADDED_COLUMNS = ["cleaned_answer", "validation_status", "valid", "decision"]
ANSWER_PATTERN = r'"answer":\s*"?([^"]*?)"?\s*}'
FAILURE = "<<model error>>"

# The BFI test experiment has 4 items and 2 profiles: the run loop asks item 0 for both profiles,
# then item 1, ... so the i-th response belongs to item i // 2.
RESPONSES = [
    "I choose 3",
    "Sorry, as an AI I cannot answer",
    "5",
    '{"answer": "4. Agree a little"}',
    "As an AI language model, I do not have opinions.",
    "I strongly agree \U0001f60a\n5. Agree strongly",
    "1 or 2",
    "Answer: 1",
]
BASIC_CLEANED = [
    "I choose 3",
    "Sorry, as an AI I cannot answer",
    "5",
    '{"answer": "4. Agree a little"}',
    "As an AI language model, I do not have opinions.",
    "I strongly agree 5. Agree strongly",
    "1 or 2",
    "Answer: 1",
]
STATUSES = ["valid", "invalid", "valid", "valid", "invalid", "valid", "valid", "valid"]
DECISIONS = [
    BFI_OPTIONS[2],
    "not present",
    BFI_OPTIONS[4],
    BFI_OPTIONS[3],
    "not present",
    BFI_OPTIONS[4],
    "inconclusive",
    BFI_OPTIONS[0],
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class ScriptedLLM(FakeListLLM):
    """A ``FakeListLLM`` whose call fails for the response ``FAILURE`` (a broken model call)."""

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        response = super()._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
        if response == FAILURE:
            raise RuntimeError("simulated model failure")
        return response


class Spy:
    """Duck-typed cleaner/validator/judge that records its arguments."""

    def __init__(self, result="ok"):
        self.result = result
        self.calls = []

    def parse(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class FailsFor:
    """Wraps a cleaner/validator/judge that cannot handle some inputs (it raises for them)."""

    def __init__(self, inner, bad_inputs):
        self.inner = inner
        self.bad_inputs = set(bad_inputs)

    def parse(self, *args, **kwargs):
        value = kwargs["text"] if "text" in kwargs else args[0]
        if value in self.bad_inputs:
            raise RuntimeError(f"cannot handle {value!r}")
        return self.inner.parse(*args, **kwargs)


def write_config(path, config):
    path.write_text(json.dumps(config), encoding="utf-8")
    return str(path)


def make_experiment(config):
    return rup.experiment_from_dict(copy.deepcopy(config))


def run_to_csv(experiment, responses, csv_path, model_id="fake"):
    """Run ``experiment`` with a scripted model and let a real CSVCallback write the results."""
    experiment.clear_models()
    experiment.add_model(ScriptedLLM(responses=list(responses)), identifier=model_id)
    experiment.run(callbacks=[CSVCallback(str(csv_path))], on_error="ignore", show_progress=False)
    return str(csv_path)


def build_pipeline(
    config, patterns, output, *, cleaner=None, validator=None, judge=None, **options
):
    options.setdefault("show_progress", False)  # the progress bars have tests of their own
    return PostprocessingPipeline(
        config,
        patterns,
        cleaner or BasicCleaner(),
        validator or ValidatorParser(),
        judge or MultipleChoiceJudge(possible_answers=BFI_OPTIONS),
        output_path=str(output),
        **options,
    )


def read_literal(path):
    """Read a CSV without any type inference or NA handling: every cell is a plain string."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def status_of(cell):
    """The pipeline stores the validator's result dict (as its repr) in ``validation_status``."""
    return ast.literal_eval(cell)["validation_status"] if cell.startswith("{") else cell


def run_python(code, *args):
    """Run ``code`` in a fresh interpreter (import side effects are what is tested)."""
    return subprocess.run(
        [sys.executable, "-c", code, *map(str, args)], capture_output=True, text=True, timeout=120
    )


@pytest.fixture
def config_path(tmp_path, config_dict):
    """The BFI configuration (default answer options only) written to disk, without models."""
    return write_config(tmp_path / "config.json", config_dict)


@pytest.fixture(scope="module")
def results_csv(tmp_path_factory, config_dict):
    """Results of one fake-LLM run of the BFI experiment (shared, so tests must not modify it)."""
    path = tmp_path_factory.mktemp("results") / "results.csv"
    return run_to_csv(make_experiment(config_dict), RESPONSES, path)


def per_item_options_config(config_dict):
    """BFI configuration where every item has its own answer options and there are no defaults."""
    config = copy.deepcopy(config_dict)
    del config["questionnaire"]["default_answer_options"]

    def options(*texts):
        return {str(i): {"text": text, "weight": i} for i, text in enumerate(texts, start=1)}

    own_options = [
        options("1. Never", "2. Sometimes", "3. Always"),
        options("1. No", "2. Yes"),
        options("A. Left", "B. Right"),
        options("1. Disagree", "2. Agree", "3. Neutral"),
    ]
    for item, answer_options in zip(config["questionnaire"]["instruction_items"], own_options):
        item["answer_options"] = answer_options
    return config


# ===========================================================================
# the happy path
# ===========================================================================


@pytest.mark.parametrize(
    ("cleaner", "cleaned"),
    [
        pytest.param(BasicCleaner(), BASIC_CLEANED, id="basic-cleaner"),
        pytest.param(
            RegexExtractorCleaner(pattern=ANSWER_PATTERN),
            [RESPONSES[0], RESPONSES[1], RESPONSES[2], "4. Agree a little", *RESPONSES[4:]],
            id="regex-extractor-cleaner",
        ),
        pytest.param(
            RegexExtractorCleaner(pattern=ANSWER_PATTERN) | BasicCleaner(),
            [*BASIC_CLEANED[:3], "4. Agree a little", *BASIC_CLEANED[4:]],
            id="chain-of-cleaners",
        ),
    ],
)
def test_pipeline_end_to_end(config_path, results_csv, tmp_path, cleaner, cleaned):
    output = tmp_path / "processed.csv"

    returned = build_pipeline(config_path, [results_csv], output, cleaner=cleaner).run()

    result = read_literal(output)
    assert list(result.columns) == RAW_COLUMNS + ADDED_COLUMNS
    assert len(result) == 8
    # the original columns survive untouched and in the original order
    pd.testing.assert_frame_equal(result[RAW_COLUMNS], read_literal(results_csv))
    assert result["answer"].tolist() == RESPONSES
    assert result["instruction_item_id"].tolist() == ["0", "0", "1", "1", "2", "2", "3", "3"]
    assert result["cleaned_answer"].tolist() == cleaned
    assert [status_of(cell) for cell in result["validation_status"]] == STATUSES
    assert result["valid"].tolist() == [str(status == "valid") for status in STATUSES]
    assert result["decision"].tolist() == DECISIONS
    assert list(returned.columns) == RAW_COLUMNS + ADDED_COLUMNS


def test_pipeline_run_returns_the_processed_frame(config_path, results_csv, tmp_path):
    output = tmp_path / "processed.csv"

    frame = build_pipeline(config_path, [results_csv], output).run()

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 8
    assert frame["cleaned_answer"].tolist() == BASIC_CLEANED
    assert frame["decision"].tolist() == DECISIONS
    # the validator's dictionary is kept as it is (the CSV file can only hold its repr)
    assert all(isinstance(verdict, dict) for verdict in frame["validation_status"])
    assert [verdict["validation_status"] for verdict in frame["validation_status"]] == STATUSES
    # ``valid`` is a real boolean column
    assert frame["valid"].dtype == bool
    assert frame["valid"].tolist() == [status == "valid" for status in STATUSES]
    # and what was returned is what was written
    written = read_literal(output)
    assert written["cleaned_answer"].tolist() == frame["cleaned_answer"].tolist()
    assert written["decision"].tolist() == frame["decision"].tolist()


def test_pipeline_keeps_the_validator_details(config_path, results_csv, tmp_path):
    output = tmp_path / "processed.csv"
    frame = build_pipeline(config_path, [results_csv], output).run()

    details = [verdict["details"] for verdict in frame["validation_status"]]
    assert details[0] == {"apologies": False, "being_ai": False, "refusal": False}
    assert details[1] == {"apologies": True, "being_ai": False, "refusal": False}
    assert details[4] == {"apologies": False, "being_ai": True, "refusal": True}
    # in the file the dictionary is its repr and can be read back with ast.literal_eval
    cells = read_literal(output)["validation_status"].tolist()
    assert [ast.literal_eval(cell)["details"] for cell in cells] == details
    assert [ast.literal_eval(cell)["text"] for cell in cells] == BASIC_CLEANED


@pytest.mark.parametrize(
    ("verdict", "valid"),
    [
        pytest.param({"validation_status": "valid"}, True, id="valid"),
        pytest.param({"validation_status": "invalid"}, False, id="invalid"),
        pytest.param({"validation_status": "valid", "details": {}}, True, id="with-details"),
        pytest.param({"validation_status": "maybe"}, False, id="unknown-status"),
        pytest.param({}, False, id="no-status"),
        pytest.param(None, False, id="none"),
    ],
)
def test_pipeline_valid_column_is_true_only_for_a_valid_verdict(
    config_path, results_csv, tmp_path, verdict, valid
):
    frame = build_pipeline(
        config_path, [results_csv], tmp_path / "out.csv", validator=Spy(verdict)
    ).run()

    assert frame["valid"].tolist() == [valid] * 8
    assert frame["valid"].dtype == bool
    assert read_literal(tmp_path / "out.csv")["valid"].tolist() == [str(valid)] * 8


def test_pipeline_logs_instead_of_printing(config_path, results_csv, tmp_path, caplog, capsys):
    output = tmp_path / "processed.csv"

    with caplog.at_level(logging.INFO, logger=LOGGER):
        build_pipeline(config_path, [results_csv], output).run()

    assert capsys.readouterr().out == ""
    records = [record for record in caplog.records if record.name == LOGGER]
    messages = [record.getMessage() for record in records]
    assert records
    assert all(record.levelno == logging.INFO for record in records)
    assert any(results_csv in message for message in messages)  # where the results come from
    assert any(str(output) in message for message in messages)  # where the processed ones went


def test_pipeline_shows_progress_bars_by_default(config_path, results_csv, tmp_path, capsys):
    build_pipeline(config_path, [results_csv], tmp_path / "out.csv", show_progress=True).run()

    captured = capsys.readouterr()
    assert "8/8" in captured.err  # a bar that counted the eight rows
    assert captured.out == ""  # the bars go to stderr


def test_pipeline_show_progress_false_is_silent(config_path, results_csv, tmp_path, capsys):
    pipeline = build_pipeline(config_path, [results_csv], tmp_path / "out.csv", show_progress=False)

    pipeline.run()

    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")
    assert len(read_literal(tmp_path / "out.csv")) == 8


def test_pipeline_is_reachable_from_the_top_level_package():
    assert rup.postprocessing.PostprocessingPipeline is PostprocessingPipeline
    assert rup.postprocessing is postprocessing_module


def test_pipeline_defaults_to_progress_bars_and_to_raising(config_path):
    defaults = PostprocessingPipeline(
        config_path, [], BasicCleaner(), ValidatorParser(), MultipleChoiceJudge(BFI_OPTIONS)
    )

    assert defaults.show_progress is True
    assert defaults.errors == "raise"


def test_importing_the_module_does_not_register_tqdm_with_pandas():
    # tqdm.pandas() patches pandas globally; it is only called when a pipeline runs with bars
    code = (
        "import pandas as pd\n"
        "import rupsycho.postprocessing\n"
        "print(hasattr(pd.Series, 'progress_apply'), hasattr(pd.DataFrame, 'progress_apply'))"
    )
    result = run_python(code)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False False"


def test_pipeline_default_output_file(config_path, results_csv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    PostprocessingPipeline(
        config_path,
        [results_csv],
        BasicCleaner(),
        ValidatorParser(),
        MultipleChoiceJudge(BFI_OPTIONS),
        show_progress=False,
    ).run()

    assert len(read_literal(tmp_path / "processed_results.csv")) == 8


def test_pipeline_without_a_run_does_not_write_anything(config_path, results_csv, tmp_path):
    output = tmp_path / "processed.csv"

    pipeline = build_pipeline(config_path, [results_csv], output)

    assert not output.exists()
    assert pipeline.output_path == str(output)
    assert pipeline.results_file_patterns == [results_csv]
    assert pipeline.config_file_path == config_path


def test_pipeline_can_run_repeatedly(config_path, results_csv, tmp_path):
    pipeline = build_pipeline(config_path, [results_csv], tmp_path / "out.csv")

    first = pipeline.run()
    second = pipeline.run()

    pd.testing.assert_frame_equal(first, second)
    assert len(read_literal(tmp_path / "out.csv")) == 8


def test_pipeline_can_process_its_own_output_again(config_path, results_csv, tmp_path):
    first_output, second_output = tmp_path / "first.csv", tmp_path / "second.csv"
    build_pipeline(config_path, [results_csv], first_output).run()

    build_pipeline(config_path, [str(first_output)], second_output).run()

    first, second = read_literal(first_output), read_literal(second_output)
    assert list(second.columns) == RAW_COLUMNS + ADDED_COLUMNS  # no column is added twice
    pd.testing.assert_frame_equal(first, second)


def test_pipeline_numeric_only_answers_are_treated_as_text(config_path, tmp_path, fake_experiment):
    # every answer is "3": pandas reads the column as integers, the pipeline must still see text
    csv_path = run_to_csv(fake_experiment, ["3"], tmp_path / "numbers.csv")
    cleaner, judge = Spy("cleaned"), Spy("decided")

    build_pipeline(
        config_path, [csv_path], tmp_path / "out.csv", cleaner=cleaner, judge=judge
    ).run()

    assert [args for args, _ in cleaner.calls] == [("3",)] * 8
    assert all(isinstance(args[0], str) for args, _ in cleaner.calls)
    assert [kwargs["text"] for _, kwargs in judge.calls] == ["cleaned"] * 8


def test_pipeline_calls_each_stage_with_the_output_of_the_previous_one(
    config_path, results_csv, tmp_path
):
    cleaner, validator, judge = Spy("cleaned"), Spy({"validation_status": "valid"}), Spy("decided")

    build_pipeline(
        config_path,
        [results_csv],
        tmp_path / "out.csv",
        cleaner=cleaner,
        validator=validator,
        judge=judge,
    ).run()

    assert [args[0] for args, _ in cleaner.calls] == RESPONSES
    assert [args[0] for args, _ in validator.calls] == ["cleaned"] * 8
    assert all(kwargs["text"] == "cleaned" for _, kwargs in judge.calls)
    result = read_literal(tmp_path / "out.csv")
    assert set(result["cleaned_answer"]) == {"cleaned"}
    assert set(result["decision"]) == {"decided"}


def test_pipeline_applies_langchain_chains_with_invoke(config_path, results_csv, tmp_path):
    # a chain (``a | b``) has no ``parse`` method of its own: the pipeline calls ``invoke``
    chain = BasicCleaner() | RegexExtractorCleaner(pattern=r"(\d)")
    assert not hasattr(chain, "parse")

    frame = build_pipeline(config_path, [results_csv], tmp_path / "out.csv", cleaner=chain).run()

    assert frame["cleaned_answer"].tolist()[:3] == ["3", "Sorry, as an AI I cannot answer", "5"]


# ===========================================================================
# the experiment the results belong to
# ===========================================================================


def test_pipeline_loads_the_experiment_but_drops_its_models():
    # the shipped configuration lists a Hugging Face model: it must be neither loaded nor kept
    pipeline = PostprocessingPipeline(
        str(CONFIG_PATH),
        [],
        BasicCleaner(),
        ValidatorParser(),
        MultipleChoiceJudge(BFI_OPTIONS),
    )

    assert pipeline.experiment.models == {}
    assert pipeline.experiment.runnable_models == {}
    assert len(pipeline.experiment.questionnaire.instruction_items) == 4


def test_pipeline_never_loads_the_models_of_the_configuration(tmp_path):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["parameters"]["lazy_load_models"] = False  # an experiment would load the model now
    config["models"] = {
        "broken": {"type": "local_huggingface", "name_or_path": str(tmp_path / "does-not-exist")}
    }

    pipeline = PostprocessingPipeline(
        config, [], BasicCleaner(), ValidatorParser(), MultipleChoiceJudge(BFI_OPTIONS)
    )

    assert pipeline.experiment.models == {}
    assert pipeline.experiment.runnable_models == {}
    # the caller's dictionary is left alone
    assert list(config["models"]) == ["broken"]
    assert config["parameters"]["lazy_load_models"] is False


@pytest.mark.parametrize("form", ["path-string", "path-object", "dictionary", "experiment"])
def test_pipeline_accepts_the_experiment_as_path_dictionary_or_object(
    form, tmp_path, config_dict, results_csv
):
    config_file = write_config(tmp_path / "config.json", config_dict)
    config = {
        "path-string": config_file,
        "path-object": Path(config_file),
        "dictionary": copy.deepcopy(config_dict),
        "experiment": make_experiment(config_dict),
    }[form]

    pipeline = build_pipeline(config, [results_csv], tmp_path / "out.csv")
    frame = pipeline.run()

    assert isinstance(pipeline.experiment, ExperimentDocument)
    assert len(pipeline.experiment.questionnaire.instruction_items) == 4
    assert frame["decision"].tolist() == DECISIONS


def test_pipeline_dictionary_configuration_without_models_or_parameters(
    tmp_path, config_dict, results_csv
):
    config = {
        key: value for key, value in config_dict.items() if key not in ("models", "parameters")
    }

    frame = build_pipeline(config, [results_csv], tmp_path / "out.csv").run()

    assert frame["decision"].tolist() == DECISIONS


# ===========================================================================
# answer options
# ===========================================================================


def test_pipeline_uses_the_answer_options_of_each_instruction_item(tmp_path, config_dict):
    config = per_item_options_config(config_dict)
    # item 0: Never/Sometimes/Always, item 1: No/Yes, item 2: Left/Right, item 3: Disagree/Agree/Neutral
    answers = ["2", "Always", "2", "no", "B", "left", "3", "Agree"]
    csv_path = run_to_csv(make_experiment(config), answers, tmp_path / "results.csv")
    config_file = write_config(tmp_path / "config.json", config)
    # the judge's own options are never consulted when the item brings its own
    judge = MultipleChoiceJudge(possible_answers=["unused"])

    build_pipeline(config_file, [csv_path], tmp_path / "out.csv", judge=judge).run()

    assert read_literal(tmp_path / "out.csv")["decision"].tolist() == [
        "2. Sometimes",
        "3. Always",
        "2. Yes",
        "1. No",
        "B. Right",
        "A. Left",
        "3. Neutral",
        "2. Agree",
    ]


def test_pipeline_looks_options_up_by_instruction_item_id_not_by_position(tmp_path, config_dict):
    config = per_item_options_config(config_dict)
    csv_path = run_to_csv(make_experiment(config), ["2"] * 8, tmp_path / "results.csv")
    shuffled = pd.read_csv(csv_path, dtype=str, keep_default_na=False).sample(
        frac=1, random_state=3
    )
    shuffled.to_csv(csv_path, index=False)
    judge = MultipleChoiceJudge(possible_answers=["unused"])

    build_pipeline(
        write_config(tmp_path / "config.json", config),
        [csv_path],
        tmp_path / "out.csv",
        judge=judge,
    ).run()

    result = read_literal(tmp_path / "out.csv")
    assert result["instruction_item_id"].tolist() == shuffled["instruction_item_id"].tolist()
    expected = {"0": "2. Sometimes", "1": "2. Yes", "2": "not present", "3": "2. Agree"}
    assert dict(zip(result["instruction_item_id"], result["decision"])) == expected


def test_pipeline_passes_the_options_of_the_item_to_the_judge(tmp_path, config_dict):
    config = per_item_options_config(config_dict)
    csv_path = run_to_csv(make_experiment(config), ["1"] * 8, tmp_path / "results.csv")
    judge = Spy()

    build_pipeline(
        write_config(tmp_path / "config.json", config),
        [csv_path],
        tmp_path / "out.csv",
        judge=judge,
    ).run()

    options_per_item = [
        ["1. Never", "2. Sometimes", "3. Always"],
        ["1. No", "2. Yes"],
        ["A. Left", "B. Right"],
        ["1. Disagree", "2. Agree", "3. Neutral"],
    ]
    assert [kwargs["possible_answers"] for _, kwargs in judge.calls] == [
        options for options in options_per_item for _ in range(2)
    ]


def test_pipeline_falls_back_to_the_default_answer_options(config_path, results_csv, tmp_path):
    judge = Spy()

    build_pipeline(config_path, [results_csv], tmp_path / "out.csv", judge=judge).run()

    # the BFI items have no options of their own: the prompt used the questionnaire defaults
    assert [kwargs["possible_answers"] for _, kwargs in judge.calls] == [BFI_OPTIONS] * 8


def test_pipeline_items_with_their_own_options_do_not_use_the_defaults(tmp_path, config_dict):
    config = copy.deepcopy(config_dict)
    own = {"1": {"text": "A. Left", "weight": 1}, "2": {"text": "B. Right", "weight": 2}}
    config["questionnaire"]["instruction_items"][1]["answer_options"] = own
    config["questionnaire"]["instruction_items"][3]["answer_options"] = own
    csv_path = run_to_csv(make_experiment(config), ["3"] * 8, tmp_path / "results.csv")
    judge = Spy()

    build_pipeline(
        write_config(tmp_path / "config.json", config),
        [csv_path],
        tmp_path / "out.csv",
        judge=judge,
    ).run()

    own_options = ["A. Left", "B. Right"]
    per_item = [BFI_OPTIONS, own_options, BFI_OPTIONS, own_options]
    assert [kwargs["possible_answers"] for _, kwargs in judge.calls] == [
        options for options in per_item for _ in range(2)
    ]


def test_pipeline_without_any_answer_options_leaves_the_choice_to_the_judge(
    tmp_path, config_dict, results_csv
):
    config = copy.deepcopy(config_dict)
    del config["questionnaire"]["default_answer_options"]  # neither items nor defaults have any
    csv_path = results_csv  # (an experiment cannot be run without options, the results exist)
    spy = Spy()
    config_file = write_config(tmp_path / "config.json", config)

    build_pipeline(config_file, [csv_path], tmp_path / "spy.csv", judge=spy).run()
    frame = build_pipeline(config_file, [csv_path], tmp_path / "judge.csv").run()

    assert [kwargs["possible_answers"] for _, kwargs in spy.calls] == [[]] * 8
    # an empty list means "use your own": MultipleChoiceJudge falls back to its options
    assert frame["decision"].tolist() == DECISIONS


def test_pipeline_with_the_demographics_judge(tmp_path, config_dict):
    config = copy.deepcopy(config_dict)
    del config["questionnaire"]["default_answer_options"]
    config["questionnaire"]["instruction_items"] = [
        {
            "question": "What gender do you identify with?",
            "answer_options": {"1": {"text": "gender"}},
        },
        {"question": "What is your age in years?", "answer_options": {"1": {"text": "age"}}},
    ]
    answers = ["I am a woman", "male", "I'm twenty-five", "I am 42 years old"]
    csv_path = run_to_csv(make_experiment(config), answers, tmp_path / "results.csv")

    build_pipeline(
        write_config(tmp_path / "config.json", config),
        [csv_path],
        tmp_path / "out.csv",
        judge=DemographicsJudge(),
    ).run()

    assert read_literal(tmp_path / "out.csv")["decision"].tolist() == ["female", "male", "25", "42"]


# ===========================================================================
# several result files
# ===========================================================================


def test_pipeline_combines_all_files_matching_the_patterns(config_path, tmp_path, config_dict):
    for model_id, answers in [("model-a", ["1"] * 8), ("model-b", ["5"] * 8)]:
        experiment = make_experiment(config_dict)
        run_to_csv(experiment, answers, tmp_path / f"res_{model_id}.csv", model_id=model_id)
    patterns = [str(tmp_path / "res_*.csv"), str(tmp_path / "nothing_*.csv")]

    build_pipeline(config_path, patterns, tmp_path / "out.csv").run()

    result = read_literal(tmp_path / "out.csv")
    assert len(result) == 16
    decisions = result.groupby("model_id")["decision"].agg(set).to_dict()
    assert decisions == {"model-a": {BFI_OPTIONS[0]}, "model-b": {BFI_OPTIONS[4]}}
    assert result["model_id"].value_counts().to_dict() == {"model-a": 8, "model-b": 8}


def test_pipeline_row_order_does_not_depend_on_the_filesystem(
    config_path, tmp_path, config_dict, monkeypatch
):
    for model_id in ("model-a", "model-b"):
        experiment = make_experiment(config_dict)
        run_to_csv(experiment, ["3"] * 8, tmp_path / f"res_{model_id}.csv", model_id=model_id)
    real_glob = postprocessing_module.glob.glob
    reversed_glob = SimpleNamespace(
        glob=lambda pattern, **kwargs: sorted(real_glob(pattern, **kwargs), reverse=True)
    )
    monkeypatch.setattr(postprocessing_module, "glob", reversed_glob)

    build_pipeline(config_path, [str(tmp_path / "res_*.csv")], tmp_path / "out.csv").run()

    models = read_literal(tmp_path / "out.csv")["model_id"].tolist()
    assert models == ["model-a"] * 8 + ["model-b"] * 8


def test_pipeline_processes_the_patterns_in_the_order_they_are_given(
    config_path, tmp_path, config_dict
):
    for model_id in ("model-a", "model-b"):
        experiment = make_experiment(config_dict)
        run_to_csv(experiment, ["3"] * 8, tmp_path / f"res_{model_id}.csv", model_id=model_id)
    patterns = [str(tmp_path / "res_model-b.csv"), str(tmp_path / "res_model-a.csv")]

    build_pipeline(config_path, patterns, tmp_path / "out.csv").run()

    models = read_literal(tmp_path / "out.csv")["model_id"].tolist()
    assert models == ["model-b"] * 8 + ["model-a"] * 8


def test_pipeline_patterns_can_reach_into_sub_directories(config_path, tmp_path, config_dict):
    (tmp_path / "runs" / "a").mkdir(parents=True)
    (tmp_path / "runs" / "b" / "c").mkdir(parents=True)
    run_to_csv(make_experiment(config_dict), ["3"] * 8, tmp_path / "runs/a/res.csv", "model-a")
    run_to_csv(make_experiment(config_dict), ["3"] * 8, tmp_path / "runs/b/c/res.csv", "model-b")

    build_pipeline(
        config_path, str(tmp_path / "runs" / "**" / "res.csv"), tmp_path / "out.csv"
    ).run()

    models = read_literal(tmp_path / "out.csv")["model_id"].tolist()
    assert models == ["model-a"] * 8 + ["model-b"] * 8


def test_pipeline_accepts_a_tuple_of_patterns(config_path, results_csv, tmp_path):
    build_pipeline(config_path, (results_csv,), tmp_path / "out.csv").run()

    assert len(read_literal(tmp_path / "out.csv")) == 8


def test_pipeline_accepts_the_literal_path_of_a_file_with_glob_characters_in_its_name(
    config_path, results_csv, tmp_path
):
    literal = tmp_path / "results[1].csv"
    literal.write_bytes(Path(results_csv).read_bytes())

    build_pipeline(config_path, [str(literal)], tmp_path / "out.csv").run()

    assert len(read_literal(tmp_path / "out.csv")) == 8


def test_pipeline_loads_a_file_once_even_if_several_patterns_match_it(
    config_path, results_csv, tmp_path
):
    patterns = [results_csv, str(Path(results_csv).parent / "*.csv")]

    build_pipeline(config_path, patterns, tmp_path / "out.csv").run()

    assert len(read_literal(tmp_path / "out.csv")) == 8


@pytest.mark.parametrize("as_list", [False, True], ids=["single", "in-a-list"])
def test_pipeline_accepts_pathlib_paths_as_result_patterns(
    config_path, results_csv, tmp_path, as_list
):
    pattern = Path(results_csv)

    build_pipeline(config_path, [pattern] if as_list else pattern, tmp_path / "out.csv").run()

    assert len(read_literal(tmp_path / "out.csv")) == 8


# ===========================================================================
# difficult answers
# ===========================================================================


def test_pipeline_treats_a_blank_answer_as_missing(config_path, tmp_path, fake_experiment):
    answers = ["I choose 3", "", "5", "Answer: 4", "I choose 1", "2", "Agree strongly", "3"]
    csv_path = run_to_csv(fake_experiment, answers, tmp_path / "results.csv")
    assert read_literal(csv_path).loc[1, "answer"] == ""  # the callback stored the gap

    frame = build_pipeline(config_path, [csv_path], tmp_path / "out.csv").run()

    # missing in, missing out: no stage is asked about the gap
    assert pd.isna(frame.loc[1, "answer"])
    assert frame.loc[1, "cleaned_answer"] is None
    assert frame.loc[1, "validation_status"] is None
    assert frame.loc[1, "decision"] is None
    assert not frame.loc[1, "valid"]
    result = read_literal(tmp_path / "out.csv")
    assert len(result) == 8
    assert result.loc[
        1, ["answer", "cleaned_answer", "validation_status", "decision"]
    ].tolist() == [
        "",
        "",
        "",
        "",
    ]
    assert result.loc[1, "valid"] == "False"
    # the neighbours are not affected
    assert result["decision"].drop(1).tolist() == [
        BFI_OPTIONS[2],
        BFI_OPTIONS[4],
        BFI_OPTIONS[3],
        BFI_OPTIONS[0],
        BFI_OPTIONS[1],
        BFI_OPTIONS[4],
        BFI_OPTIONS[2],
    ]


def test_pipeline_does_not_call_any_stage_for_a_missing_answer(
    config_path, tmp_path, fake_experiment
):
    answers = ["3", "", "5", "4", "3", "2", "1", "5"]
    csv_path = run_to_csv(fake_experiment, answers, tmp_path / "results.csv")
    cleaner, validator, judge = Spy("cleaned"), Spy({"validation_status": "valid"}), Spy("decided")

    build_pipeline(
        config_path,
        [csv_path],
        tmp_path / "out.csv",
        cleaner=cleaner,
        validator=validator,
        judge=judge,
    ).run()

    assert (len(cleaner.calls), len(validator.calls), len(judge.calls)) == (7, 7, 7)


def test_pipeline_keeps_answers_that_pandas_would_read_as_missing(
    config_path, tmp_path, fake_experiment
):
    odd = ["None", "NA", "N/A", "null", "NaN"]
    answers = [*odd, "I choose 3", "5", "Agree strongly"]
    csv_path = run_to_csv(fake_experiment, answers, tmp_path / "results.csv")

    build_pipeline(config_path, [csv_path], tmp_path / "out.csv").run()

    result = read_literal(tmp_path / "out.csv")
    assert result["answer"].tolist() == answers
    assert result["cleaned_answer"].tolist() == answers
    assert result["valid"].tolist() == ["True"] * 8
    assert result["decision"].tolist() == [
        *["not present"] * 5,
        BFI_OPTIONS[2],
        BFI_OPTIONS[4],
        BFI_OPTIONS[4],
    ]


def test_pipeline_survives_a_failed_generation(config_path, tmp_path, fake_experiment):
    answers = ["I choose 3", FAILURE, "5", "4", "3", "2", "1", "5"]
    csv_path = run_to_csv(fake_experiment, answers, tmp_path / "results.csv")
    assert read_literal(csv_path).loc[1, "answer"] == ""  # the callback stored the gap

    frame = build_pipeline(config_path, [csv_path], tmp_path / "out.csv").run()

    result = read_literal(tmp_path / "out.csv")
    assert len(result) == 8
    assert result.loc[0, "decision"] == BFI_OPTIONS[2]
    # the failed call has no answer, hence nothing to clean, validate or judge
    assert result.loc[1, "decision"] == ""
    assert result.loc[1, "valid"] == "False"
    assert frame.loc[1, "decision"] is None
    assert result["decision"].drop(1).tolist() == [
        BFI_OPTIONS[2],
        BFI_OPTIONS[4],
        BFI_OPTIONS[3],
        BFI_OPTIONS[2],
        BFI_OPTIONS[1],
        BFI_OPTIONS[0],
        BFI_OPTIONS[4],
    ]


def test_pipeline_numeric_answers_survive_a_blank_row(config_path, tmp_path, fake_experiment):
    answers = ["3", "", "5", "4", "3", "2", "1", "5"]
    csv_path = run_to_csv(fake_experiment, answers, tmp_path / "results.csv")

    build_pipeline(config_path, [csv_path], tmp_path / "out.csv").run()

    result = read_literal(tmp_path / "out.csv")
    assert result["answer"].tolist() == ["3", "", "5", "4", "3", "2", "1", "5"]
    assert result["cleaned_answer"].tolist() == ["3", "", "5", "4", "3", "2", "1", "5"]
    assert result["decision"].tolist()[0] == BFI_OPTIONS[2]


def test_pipeline_flags_refusals_written_with_typographic_apostrophes(
    config_path, tmp_path, fake_experiment
):
    answers = ["I’m sorry, but I can’t help with that.", "I can’t answer that."]
    csv_path = run_to_csv(fake_experiment, answers * 4, tmp_path / "results.csv")

    frame = build_pipeline(config_path, [csv_path], tmp_path / "out.csv").run()

    statuses = [status_of(c) for c in read_literal(tmp_path / "out.csv")["validation_status"]]
    assert statuses == ["invalid"] * 8
    assert not frame["valid"].any()
    # the cleaner maps the typographic apostrophe instead of dropping it
    assert frame["cleaned_answer"][0] == "I'm sorry, but I can't help with that."


def test_pipeline_with_the_prompt_removal_cleaner(config_path, tmp_path, fake_experiment):
    answers = [f"Answer: {digit}" for digit in "35413251"]
    csv_path = run_to_csv(fake_experiment, answers, tmp_path / "results.csv")
    cleaner = PromptRemovalCleaner(prompt="Answer:")

    build_pipeline(config_path, [csv_path], tmp_path / "out.csv", cleaner=cleaner).run()

    result = read_literal(tmp_path / "out.csv")
    assert result["cleaned_answer"].tolist() == list("35413251")
    assert result["decision"].tolist() == [BFI_OPTIONS[int(d) - 1] for d in "35413251"]


def test_pipeline_handles_a_results_file_without_rows(config_path, tmp_path):
    csv_path = tmp_path / "empty.csv"
    CSVCallback(str(csv_path))  # a fresh callback writes just the header

    frame = build_pipeline(config_path, [str(csv_path)], tmp_path / "out.csv").run()

    result = read_literal(tmp_path / "out.csv")
    assert list(result.columns) == RAW_COLUMNS + ADDED_COLUMNS
    assert len(result) == 0
    assert len(frame) == 0


def test_pipeline_ignores_a_results_file_without_rows_next_to_a_full_one(
    config_path, results_csv, tmp_path
):
    empty = tmp_path / "empty.csv"
    CSVCallback(str(empty))

    build_pipeline(config_path, [results_csv, str(empty)], tmp_path / "out.csv").run()

    assert read_literal(tmp_path / "out.csv")["decision"].tolist() == DECISIONS


# ===========================================================================
# errors="raise" / errors="coerce"
# ===========================================================================

# the row the failing stages choke on: "Sorry, as an AI I cannot answer" (response 1)
BAD_ROW = 1


def failing_stages(stage):
    """Cleaner, validator and judge, one of which cannot handle the text of ``BAD_ROW``."""
    cleaner, validator = BasicCleaner(), ValidatorParser()
    judge = MultipleChoiceJudge(possible_answers=BFI_OPTIONS)
    if stage == "cleaner":
        cleaner = FailsFor(cleaner, [RESPONSES[BAD_ROW]])
    elif stage == "validator":
        validator = FailsFor(validator, [BASIC_CLEANED[BAD_ROW]])
    else:
        judge = FailsFor(judge, [BASIC_CLEANED[BAD_ROW]])
    return {"cleaner": cleaner, "validator": validator, "judge": judge}


@pytest.mark.parametrize("stage", ["cleaner", "validator", "judge"])
def test_pipeline_raises_by_default_and_names_the_original_error(
    config_path, results_csv, tmp_path, stage
):
    pipeline = build_pipeline(
        config_path, [results_csv], tmp_path / "out.csv", **failing_stages(stage)
    )

    with pytest.raises(RuntimeError, match="cannot handle"):
        pipeline.run()

    assert not (tmp_path / "out.csv").exists()


@pytest.mark.parametrize(
    ("stage", "label", "expected"),
    [
        pytest.param(
            "cleaner",
            "Cleaning",
            # nothing to validate or judge without a cleaned answer
            {"cleaned_answer": None, "validation_status": None, "valid": False, "decision": None},
            id="cleaner",
        ),
        pytest.param(
            "validator",
            "Validating",
            {
                "cleaned_answer": BASIC_CLEANED[BAD_ROW],
                "validation_status": None,
                "valid": False,
                "decision": DECISIONS[BAD_ROW],
            },
            id="validator",
        ),
        pytest.param(
            "judge",
            "Judging",
            {
                "cleaned_answer": BASIC_CLEANED[BAD_ROW],
                "validation_status": "invalid",
                "valid": False,
                "decision": None,
            },
            id="judge",
        ),
    ],
)
def test_pipeline_coerce_leaves_the_cells_of_a_failing_row_empty(
    config_path, results_csv, tmp_path, caplog, stage, label, expected
):
    pipeline = build_pipeline(
        config_path, [results_csv], tmp_path / "out.csv", errors="coerce", **failing_stages(stage)
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        frame = pipeline.run()

    row = frame.loc[BAD_ROW]
    verdict = row["validation_status"]
    assert row["cleaned_answer"] == expected["cleaned_answer"]
    assert (None if verdict is None else verdict["validation_status"]) == expected[
        "validation_status"
    ]
    assert bool(row["valid"]) is expected["valid"]
    assert row["decision"] == expected["decision"]
    # every other row is processed as usual
    good = frame.drop(BAD_ROW)
    assert good["decision"].tolist() == [d for i, d in enumerate(DECISIONS) if i != BAD_ROW]
    assert len(frame) == 8
    # the problem is logged once, naming the stage, the offending text and the reason
    warnings_logged = [
        r for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING
    ]
    assert len(warnings_logged) == 1
    message = warnings_logged[0].getMessage()
    assert label.lower() in message.lower()
    assert "failed" in message
    assert "cannot handle" in message
    assert len(read_literal(tmp_path / "out.csv")) == 8


def test_pipeline_coerce_with_a_misconfigured_parser(config_path, results_csv, tmp_path, caplog):
    # a pattern without a capture group fails (OutputParserException) for every answer with a digit
    cleaner = RegexExtractorCleaner(pattern=r"\d")
    pipeline = build_pipeline(
        config_path, [results_csv], tmp_path / "out.csv", cleaner=cleaner, errors="coerce"
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        frame = pipeline.run()

    with_digit = [any(char.isdigit() for char in answer) for answer in RESPONSES]
    assert frame["cleaned_answer"].isna().tolist() == with_digit
    assert frame["validation_status"].isna().tolist() == with_digit
    assert frame["decision"].isna().tolist() == with_digit
    # nothing is valid: the failed rows have no verdict and the other two are refusals
    assert not frame["valid"].any()
    logged = [
        record.getMessage()
        for record in caplog.records
        if record.name == LOGGER and record.levelno >= logging.WARNING
    ]
    assert len(logged) == sum(with_digit)
    assert all("cleaning" in message.lower() for message in logged)
    assert all("RegexExtractorCleaner encountered an error" in message for message in logged)


def test_pipeline_raise_reports_a_misconfigured_parser(config_path, results_csv, tmp_path):
    cleaner = RegexExtractorCleaner(pattern=r"\d")
    pipeline = build_pipeline(config_path, [results_csv], tmp_path / "out.csv", cleaner=cleaner)

    with pytest.raises(OutputParserException, match="RegexExtractorCleaner"):
        pipeline.run()


@pytest.mark.parametrize("errors", ["ignore", "RAISE", "", None, True])
def test_pipeline_rejects_an_unknown_error_policy(config_path, errors):
    with pytest.raises(ValueError, match="errors must be 'raise' or 'coerce'"):
        PostprocessingPipeline(
            config_path,
            [],
            BasicCleaner(),
            ValidatorParser(),
            MultipleChoiceJudge(BFI_OPTIONS),
            errors=errors,
        )


def test_pipeline_options_are_keyword_only(config_path):
    with pytest.raises(TypeError):
        PostprocessingPipeline(
            config_path,
            [],
            BasicCleaner(),
            ValidatorParser(),
            MultipleChoiceJudge(BFI_OPTIONS),
            "out.csv",
            "coerce",
        )


# ===========================================================================
# configuration and input errors
# ===========================================================================


def test_pipeline_unknown_config_file_names_the_path(tmp_path):
    missing = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="missing.json"):
        build_pipeline(str(missing), [], tmp_path / "out.csv")


def test_pipeline_config_file_with_malformed_json(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError):  # json.JSONDecodeError is a ValueError
        build_pipeline(str(broken), [], tmp_path / "out.csv")


def test_pipeline_accepts_a_config_without_parameters(results_csv, tmp_path, config_dict):
    # the documented defaults (random seed, lazy model loading) make "parameters" optional
    config = {key: value for key, value in config_dict.items() if key != "parameters"}
    assert config["models"] == {}
    config_file = write_config(tmp_path / "no_parameters.json", config)

    build_pipeline(config_file, [results_csv], tmp_path / "out.csv").run()

    assert read_literal(tmp_path / "out.csv")["decision"].tolist() == DECISIONS


def test_pipeline_accepts_a_single_pattern_given_as_a_string(config_path, results_csv, tmp_path):
    pipeline = build_pipeline(config_path, results_csv, tmp_path / "out.csv")
    pipeline.run()

    assert pipeline.results_file_patterns == [results_csv]
    assert len(read_literal(tmp_path / "out.csv")) == 8


def test_pipeline_invalid_config_file_raises_a_descriptive_error(tmp_path):
    invalid = write_config(tmp_path / "invalid.json", {"questionnaire": "not a questionnaire"})

    with pytest.raises(ValueError, match="questionnaire") as excinfo:
        build_pipeline(invalid, [], tmp_path / "out.csv")

    assert not isinstance(excinfo.value, AttributeError)


def test_pipeline_config_without_a_questionnaire_cannot_run(results_csv, tmp_path, config_dict):
    config = {key: value for key, value in config_dict.items() if key != "questionnaire"}
    pipeline = build_pipeline(config, [results_csv], tmp_path / "out.csv")

    with pytest.raises(ValueError, match="no questionnaire items"):
        pipeline.run()

    assert not (tmp_path / "out.csv").exists()


def test_pipeline_without_matching_result_files_cannot_run(config_path, tmp_path):
    pipeline = build_pipeline(config_path, [str(tmp_path / "nothing_*.csv")], tmp_path / "out.csv")

    with pytest.raises(FileNotFoundError):
        pipeline.run()

    assert not (tmp_path / "out.csv").exists()


def test_pipeline_without_matching_result_files_names_the_patterns(config_path, tmp_path):
    pattern = str(tmp_path / "nothing_*.csv")
    pipeline = build_pipeline(config_path, [pattern], tmp_path / "out.csv")

    with pytest.raises(FileNotFoundError) as excinfo:
        pipeline.run()

    assert pattern in str(excinfo.value)


def test_pipeline_without_any_pattern_cannot_run(config_path, tmp_path):
    with pytest.raises(FileNotFoundError, match="No result files match"):
        build_pipeline(config_path, [], tmp_path / "out.csv").run()


@pytest.mark.parametrize("missing", REQUIRED_COLUMNS)
def test_pipeline_result_file_without_a_required_column(config_path, tmp_path, missing):
    header = [column for column in RAW_COLUMNS if column != missing]
    broken = tmp_path / "broken.csv"
    broken.write_text(
        ",".join(header) + "\n" + ",".join("x" for _ in header) + "\n", encoding="utf-8"
    )
    pipeline = build_pipeline(config_path, [str(broken)], tmp_path / "out.csv")

    with pytest.raises(ValueError, match=f"lacks the column.*{missing}") as excinfo:
        pipeline.run()

    assert str(broken) in str(excinfo.value)
    assert not (tmp_path / "out.csv").exists()


def test_pipeline_required_columns_are_written_by_the_csv_callback(tmp_path):
    CSVCallback(str(tmp_path / "header.csv"))

    header = (tmp_path / "header.csv").read_text(encoding="utf-8").strip().split(",")

    assert header == RAW_COLUMNS
    assert REQUIRED_COLUMNS == ("instruction_item_id", "answer")
    assert set(REQUIRED_COLUMNS) <= set(header)


def test_pipeline_checks_every_result_file(config_path, results_csv, tmp_path):
    broken = tmp_path / "res_broken.csv"
    broken.write_text("experiment_name,instruction_item_id\nE,0\n", encoding="utf-8")
    pipeline = build_pipeline(config_path, [results_csv, str(broken)], tmp_path / "out.csv")

    with pytest.raises(ValueError, match="res_broken.csv"):
        pipeline.run()


def test_pipeline_item_id_beyond_the_questionnaire(config_path, results_csv, tmp_path):
    frame = read_literal(results_csv)
    frame.loc[0, "instruction_item_id"] = "99"
    frame.to_csv(tmp_path / "shifted.csv", index=False)
    pipeline = build_pipeline(config_path, [str(tmp_path / "shifted.csv")], tmp_path / "out.csv")

    with pytest.raises((IndexError, KeyError, ValueError)):
        pipeline.run()

    assert not (tmp_path / "out.csv").exists()


def test_pipeline_judge_errors_are_not_swallowed(config_path, results_csv, tmp_path):
    class Exploding:
        def parse(self, text, possible_answers):
            raise RuntimeError("judge exploded")

    pipeline = build_pipeline(config_path, [results_csv], tmp_path / "out.csv", judge=Exploding())

    with pytest.raises(RuntimeError, match="judge exploded"):
        pipeline.run()


# ===========================================================================
# the base installation
# ===========================================================================

BASE_INSTALLATION = """
import sys

# the huggingface extra is not installed: importing these raises ImportError
for name in ("torch", "transformers", "scipy", "num2words"):
    sys.modules[name] = None

from rupsycho.parsers.cleaners import BasicCleaner
from rupsycho.parsers.judges import MultipleChoiceJudge
from rupsycho.parsers.validators import ValidatorParser
from rupsycho.postprocessing import PostprocessingPipeline

config, results, output = sys.argv[1:4]
options = sys.argv[4:]
frame = PostprocessingPipeline(
    config,
    results,
    BasicCleaner(),
    ValidatorParser(),
    MultipleChoiceJudge(options),
    output_path=output,
    show_progress=False,
).run()
print("|".join(frame["decision"]))
"""


def test_pipeline_runs_without_torch_and_transformers(config_path, results_csv, tmp_path):
    result = run_python(
        BASE_INSTALLATION, config_path, results_csv, tmp_path / "out.csv", *BFI_OPTIONS
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "|".join(DECISIONS)
