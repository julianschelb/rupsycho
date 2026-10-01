"""End-to-end tests for ``rupsycho.postprocessing.PostprocessingPipeline``.

Every test produces its input the way a user does: a fake-LLM experiment is run with a real
``CSVCallback`` and the resulting CSV file is post-processed with real cleaners, validators and
judges. Nothing is downloaded.

Tests marked ``xfail(strict=True)`` describe the *correct* behaviour of a known defect: they fail
today and turn into a loud XPASS as soon as the defect is fixed, at which point the marker has to
be removed.
"""

import ast
import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from langchain_core.language_models.fake import FakeListLLM

import rupsycho as rup
import rupsycho.postprocessing as postprocessing_module
from rupsycho.callbacks import CSVCallback
from rupsycho.parsers.cleaners import BasicCleaner, PromptRemovalCleaner, RegexExtractorCleaner
from rupsycho.parsers.judges import DemographicsJudge, MultipleChoiceJudge
from rupsycho.parsers.validators import ValidatorParser
from rupsycho.postprocessing import PostprocessingPipeline

from .conftest import CONFIG_PATH
from .test_parser_utils import fake_num2words  # noqa: F401  (fixture, used via usefixtures)

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
ADDED_COLUMNS = ["cleaned_answer", "validation_status", "decision"]
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


def write_config(path, config):
    path.write_text(json.dumps(config), encoding="utf-8")
    return str(path)


def make_experiment(config):
    return rup.experiment_from_dict(copy.deepcopy(config))


def run_to_csv(experiment, responses, csv_path, model_id="fake"):
    """Run ``experiment`` with a scripted model and let a real CSVCallback write the results."""
    experiment.clear_models()
    experiment.add_model(ScriptedLLM(responses=list(responses)), identifier=model_id)
    experiment.run(callbacks=[CSVCallback(str(csv_path))])
    return str(csv_path)


def build_pipeline(config, patterns, output, *, cleaner=None, validator=None, judge=None):
    return PostprocessingPipeline(
        config,
        patterns,
        cleaner or BasicCleaner(),
        validator or ValidatorParser(),
        judge or MultipleChoiceJudge(possible_answers=BFI_OPTIONS),
        output_path=str(output),
    )


def read_literal(path):
    """Read a CSV without any type inference or NA handling: every cell is a plain string."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def status_of(cell):
    """The pipeline stores the validator's result dict (as its repr) in ``validation_status``."""
    return ast.literal_eval(cell)["validation_status"] if cell.startswith("{") else cell


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
    ],
)
def test_pipeline_end_to_end(config_path, results_csv, tmp_path, cleaner, cleaned):
    output = tmp_path / "processed.csv"

    build_pipeline(config_path, [results_csv], output, cleaner=cleaner).run()

    result = read_literal(output)
    assert list(result.columns) == RAW_COLUMNS + ADDED_COLUMNS
    assert len(result) == 8
    # the original columns survive untouched and in the original order
    pd.testing.assert_frame_equal(result[RAW_COLUMNS], read_literal(results_csv))
    assert result["answer"].tolist() == RESPONSES
    assert result["instruction_item_id"].tolist() == ["0", "0", "1", "1", "2", "2", "3", "3"]
    assert result["cleaned_answer"].tolist() == cleaned
    assert [status_of(cell) for cell in result["validation_status"]] == STATUSES
    assert result["decision"].tolist() == DECISIONS


def test_pipeline_keeps_the_validator_details(config_path, results_csv, tmp_path):
    output = tmp_path / "processed.csv"
    build_pipeline(config_path, [results_csv], output).run()

    cells = read_literal(output)["validation_status"].tolist()

    details = [ast.literal_eval(cell)["details"] for cell in cells]
    assert details[0] == {"apologies": False, "being_ai": False, "refusal": False}
    assert details[1] == {"apologies": True, "being_ai": False, "refusal": False}
    assert details[4] == {"apologies": False, "being_ai": True, "refusal": True}


def test_pipeline_prints_progress_messages(config_path, results_csv, tmp_path, capsys):
    output = tmp_path / "processed.csv"

    build_pipeline(config_path, [results_csv], output).run()

    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "Loading data...",
        "Processing data...",
        f"Processed results saved to {output}",
    ]


def test_pipeline_default_output_file(config_path, results_csv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    PostprocessingPipeline(
        config_path,
        [results_csv],
        BasicCleaner(),
        ValidatorParser(),
        MultipleChoiceJudge(BFI_OPTIONS),
    ).run()

    assert len(read_literal(tmp_path / "processed_results.csv")) == 8


def test_pipeline_without_a_run_does_not_write_anything(config_path, results_csv, tmp_path):
    output = tmp_path / "processed.csv"

    pipeline = build_pipeline(config_path, [results_csv], output)

    assert not output.exists()
    assert pipeline.output_path == str(output)
    assert pipeline.results_file_patterns == [results_csv]
    assert pipeline.config_file_path == config_path


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


@pytest.mark.xfail(
    strict=True,
    reason="PostprocessingPipeline ignores questionnaire.default_answer_options (open TODO in the code)",
)
def test_pipeline_falls_back_to_the_default_answer_options(config_path, results_csv, tmp_path):
    judge = Spy()

    build_pipeline(config_path, [results_csv], tmp_path / "out.csv", judge=judge).run()

    # the BFI items have no options of their own: the prompt used the questionnaire defaults
    assert [kwargs["possible_answers"] for _, kwargs in judge.calls] == [BFI_OPTIONS] * 8


@pytest.mark.usefixtures("fake_num2words")
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


@pytest.mark.xfail(
    strict=True,
    reason="rows follow glob.glob order, which is arbitrary, instead of sorted file names",
)
def test_pipeline_row_order_does_not_depend_on_the_filesystem(
    config_path, tmp_path, config_dict, monkeypatch
):
    for model_id in ("model-a", "model-b"):
        experiment = make_experiment(config_dict)
        run_to_csv(experiment, ["3"] * 8, tmp_path / f"res_{model_id}.csv", model_id=model_id)
    real_glob = postprocessing_module.glob.glob
    reversed_glob = SimpleNamespace(glob=lambda pattern: sorted(real_glob(pattern), reverse=True))
    monkeypatch.setattr(postprocessing_module, "glob", reversed_glob)

    build_pipeline(config_path, [str(tmp_path / "res_*.csv")], tmp_path / "out.csv").run()

    models = read_literal(tmp_path / "out.csv")["model_id"].tolist()
    assert models == ["model-a"] * 8 + ["model-b"] * 8


# ===========================================================================
# difficult answers
# ===========================================================================


@pytest.mark.xfail(
    strict=True,
    reason="pd.read_csv turns blank/'None'/'NA'/'null' answers into NaN and BasicCleaner crashes on them",
)
@pytest.mark.parametrize("odd", ["", "None", "NA", "N/A", "null", "NaN"], ids=repr)
def test_pipeline_survives_blank_and_na_like_answers(config_path, tmp_path, fake_experiment, odd):
    answers = ["I choose 3", odd, "5", "Answer: 4", "I choose 1", "2", "Agree strongly", "3"]
    csv_path = run_to_csv(fake_experiment, answers, tmp_path / "results.csv")

    build_pipeline(config_path, [csv_path], tmp_path / "out.csv").run()

    result = read_literal(tmp_path / "out.csv")
    assert len(result) == 8
    assert result.loc[1, "cleaned_answer"] == odd
    assert result.loc[1, "decision"] == "not present"
    assert result["decision"].drop(1).tolist() == [
        BFI_OPTIONS[2],
        BFI_OPTIONS[4],
        BFI_OPTIONS[3],
        BFI_OPTIONS[0],
        BFI_OPTIONS[1],
        BFI_OPTIONS[4],
        BFI_OPTIONS[2],
    ]


@pytest.mark.xfail(
    strict=True,
    reason="a failed model call is saved as a blank answer, which crashes the pipeline (NaN)",
)
def test_pipeline_survives_a_failed_generation(config_path, tmp_path, fake_experiment):
    answers = ["I choose 3", FAILURE, "5", "4", "3", "2", "1", "5"]
    csv_path = run_to_csv(fake_experiment, answers, tmp_path / "results.csv")
    assert read_literal(csv_path).loc[1, "answer"] == ""  # the callback stored the gap

    build_pipeline(config_path, [csv_path], tmp_path / "out.csv").run()

    result = read_literal(tmp_path / "out.csv")
    assert len(result) == 8
    assert result.loc[1, "decision"] == "not present"
    assert result.loc[0, "decision"] == BFI_OPTIONS[2]


@pytest.mark.xfail(
    strict=True,
    reason="with a blank answer pandas reads the numeric answers as floats: '3' becomes '3.0'",
)
def test_pipeline_numeric_answers_survive_a_blank_row(config_path, tmp_path, fake_experiment):
    answers = ["3", "", "5", "4", "3", "2", "1", "5"]
    csv_path = run_to_csv(fake_experiment, answers, tmp_path / "results.csv")

    build_pipeline(config_path, [csv_path], tmp_path / "out.csv").run()

    result = read_literal(tmp_path / "out.csv")
    assert result["cleaned_answer"].tolist() == ["3", "", "5", "4", "3", "2", "1", "5"]
    assert result["decision"].tolist()[0] == BFI_OPTIONS[2]


@pytest.mark.xfail(
    strict=True,
    reason="BasicCleaner drops the typographic apostrophe: 'I’m sorry' becomes 'Im sorry'",
)
def test_pipeline_flags_refusals_written_with_typographic_apostrophes(
    config_path, tmp_path, fake_experiment
):
    answers = ["I’m sorry, but I can’t help with that.", "I can’t answer that."]
    csv_path = run_to_csv(fake_experiment, answers * 4, tmp_path / "results.csv")

    build_pipeline(config_path, [csv_path], tmp_path / "out.csv").run()

    statuses = [status_of(c) for c in read_literal(tmp_path / "out.csv")["validation_status"]]
    assert statuses == ["invalid"] * 8


@pytest.mark.xfail(
    strict=True,
    reason="PromptRemovalCleaner.parse returns a dict, which the validator cannot process",
)
def test_pipeline_with_the_prompt_removal_cleaner(config_path, tmp_path, fake_experiment):
    answers = [f"Answer: {digit}" for digit in "35413251"]
    csv_path = run_to_csv(fake_experiment, answers, tmp_path / "results.csv")
    cleaner = PromptRemovalCleaner(prompt="Answer:")

    build_pipeline(config_path, [csv_path], tmp_path / "out.csv", cleaner=cleaner).run()

    result = read_literal(tmp_path / "out.csv")
    assert result["cleaned_answer"].tolist() == list("35413251")
    assert result["decision"].tolist() == [BFI_OPTIONS[int(d) - 1] for d in "35413251"]


@pytest.mark.xfail(
    strict=True,
    reason="a results file without rows crashes: DataFrame.apply(axis=1) returns a frame when empty",
)
def test_pipeline_handles_a_results_file_without_rows(config_path, tmp_path):
    csv_path = tmp_path / "empty.csv"
    CSVCallback(str(csv_path))  # a fresh callback writes just the header

    build_pipeline(config_path, [str(csv_path)], tmp_path / "out.csv").run()

    result = read_literal(tmp_path / "out.csv")
    assert list(result.columns) == RAW_COLUMNS + ADDED_COLUMNS
    assert len(result) == 0


def test_pipeline_ignores_a_results_file_without_rows_next_to_a_full_one(
    config_path, results_csv, tmp_path
):
    empty = tmp_path / "empty.csv"
    CSVCallback(str(empty))

    build_pipeline(config_path, [results_csv, str(empty)], tmp_path / "out.csv").run()

    assert read_literal(tmp_path / "out.csv")["decision"].tolist() == DECISIONS


# ===========================================================================
# configuration and input errors
# ===========================================================================


def test_pipeline_unknown_config_file_names_the_path(tmp_path):
    missing = tmp_path / "missing.json"

    with pytest.raises((RuntimeError, FileNotFoundError, ValueError), match="missing.json"):
        build_pipeline(str(missing), [], tmp_path / "out.csv")


@pytest.mark.xfail(
    strict=True,
    reason="ExperimentDocument.__init__ needs a 'parameters' key: data.get('parameters').lazy_load_models",
)
def test_pipeline_accepts_a_config_without_parameters(results_csv, tmp_path, config_dict):
    # the documented defaults (random seed, lazy model loading) make "parameters" optional
    config = {key: value for key, value in config_dict.items() if key != "parameters"}
    assert config["models"] == {}
    config_file = write_config(tmp_path / "no_parameters.json", config)

    build_pipeline(config_file, [results_csv], tmp_path / "out.csv").run()

    assert read_literal(tmp_path / "out.csv")["decision"].tolist() == DECISIONS


@pytest.mark.xfail(
    strict=True,
    reason="a pattern given as a plain string is iterated character by character (first glob: '/')",
)
def test_pipeline_accepts_a_single_pattern_given_as_a_string(config_path, results_csv, tmp_path):
    build_pipeline(config_path, results_csv, tmp_path / "out.csv").run()

    assert len(read_literal(tmp_path / "out.csv")) == 8


@pytest.mark.xfail(
    strict=True,
    reason="a config that fails validation makes experiment_from_file return None: AttributeError",
)
def test_pipeline_invalid_config_file_raises_a_descriptive_error(tmp_path):
    invalid = write_config(tmp_path / "invalid.json", {"questionnaire": "not a questionnaire"})

    with pytest.raises(Exception) as excinfo:
        build_pipeline(invalid, [], tmp_path / "out.csv")

    assert not isinstance(excinfo.value, AttributeError)
    assert "invalid.json" in str(excinfo.value)


def test_pipeline_without_matching_result_files_cannot_run(config_path, tmp_path):
    pipeline = build_pipeline(config_path, [str(tmp_path / "nothing_*.csv")], tmp_path / "out.csv")

    with pytest.raises((ValueError, FileNotFoundError)):
        pipeline.run()

    assert not (tmp_path / "out.csv").exists()


@pytest.mark.xfail(
    strict=True,
    reason="without result files the user only sees pandas' 'No objects to concatenate'",
)
def test_pipeline_without_matching_result_files_names_the_patterns(config_path, tmp_path):
    pattern = str(tmp_path / "nothing_*.csv")
    pipeline = build_pipeline(config_path, [pattern], tmp_path / "out.csv")

    with pytest.raises(Exception) as excinfo:
        pipeline.run()

    assert pattern in str(excinfo.value)


def test_pipeline_result_file_without_an_answer_column(config_path, tmp_path):
    broken = tmp_path / "broken.csv"
    broken.write_text("experiment_name,instruction_item_id\nE,0\n", encoding="utf-8")
    pipeline = build_pipeline(config_path, [str(broken)], tmp_path / "out.csv")

    with pytest.raises((KeyError, ValueError), match="answer"):
        pipeline.run()


def test_pipeline_item_id_beyond_the_questionnaire(config_path, results_csv, tmp_path):
    frame = read_literal(results_csv)
    frame.loc[0, "instruction_item_id"] = "99"
    frame.to_csv(tmp_path / "shifted.csv", index=False)
    pipeline = build_pipeline(config_path, [str(tmp_path / "shifted.csv")], tmp_path / "out.csv")

    with pytest.raises((IndexError, KeyError, ValueError)):
        pipeline.run()


def test_pipeline_judge_errors_are_not_swallowed(config_path, results_csv, tmp_path):
    class Exploding:
        def parse(self, text, possible_answers):
            raise RuntimeError("judge exploded")

    pipeline = build_pipeline(config_path, [results_csv], tmp_path / "out.csv", judge=Exploding())

    with pytest.raises(RuntimeError, match="judge exploded"):
        pipeline.run()
