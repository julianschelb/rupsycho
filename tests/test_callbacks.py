"""Unit tests for ``rupsycho.callbacks``: the callbacks that persist or print every answer.

Most tests call ``save_answer`` directly with a real ``InstructionItem`` and a stand-in experiment
(the callbacks only read ``experiment.name``). A few tests run the BFI ``fake_experiment`` from
``conftest.py`` to check the callbacks against the real run loop, including model calls that fail
(the callbacks then receive ``answer=None``) and calls that run concurrently.
"""

import builtins
import copy
import csv
import inspect
import json
import threading
from time import sleep
from types import SimpleNamespace

import pandas as pd
import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.language_models.llms import LLM

import rupsycho as rup
from rupsycho.callbacks import (
    Callback,
    CSVCallback,
    JSONLCallback,
    PrintCallback,
    PrintTableCallback,
)
from rupsycho.models.questionnaire import InstructionItem

EXPERIMENT = SimpleNamespace(name="Demo experiment")

CSV_HEADER = [
    "experiment_name",
    "instruction_item_id",
    "instruction_item",
    "model_id",
    "profile_id",
    "random_seed",
    "time",
    "answer",
]

SAVE_ANSWER_PARAMETERS = [
    "self",
    "experiment",
    "instruction_item_id",
    "instruction_item",
    "model_id",
    "profile_id",
    "random_seed",
    "time",
    "answer",
]


def make_item(question="How are you?"):
    return InstructionItem(question=question, attributes={"dimension": "1"})


def save(
    callback,
    answer="3",
    *,
    item_id=0,
    item=None,
    model_id="model-a",
    profile_id="profile-1",
    seed="7",
    time=0.25,
):
    """Call ``save_answer`` the way the run loop does (all arguments positional)."""
    callback.save_answer(
        EXPERIMENT,
        item_id,
        item or make_item(),
        model_id,
        profile_id,
        seed,
        time,
        answer,
    )


def read_csv_rows(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class Recorder(Callback):
    """Remembers every call it receives (and the thread it was made from)."""

    def __init__(self):
        self.calls = []
        self.threads = []

    def save_answer(self, *args):
        self.calls.append(args)
        self.threads.append(threading.get_ident())


FAILURE = "<<model error>>"


class ScriptedLLM(FakeListLLM):
    """A ``FakeListLLM`` whose call raises for the response ``FAILURE`` (a broken model call)."""

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        response = super()._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
        if response == FAILURE:
            raise RuntimeError("simulated model failure")
        return response


class PromptLengthLLM(LLM):
    """Answers with the length of its prompt: constant per call, different between calls."""

    delay: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "prompt-length"

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        sleep(self.delay * (1 + len(prompt) % 3))  # uneven: later calls may finish first
        return str(len(prompt))


# ===========================================================================
# the Callback interface
# ===========================================================================


def test_callbacks_package_exports_every_callback():
    import rupsycho.callbacks as callbacks

    assert sorted(callbacks.__all__) == [
        "CSVCallback",
        "Callback",
        "JSONLCallback",
        "PrintCallback",
        "PrintTableCallback",
    ]
    assert all(issubclass(getattr(callbacks, name), Callback) for name in callbacks.__all__)


def test_callbacks_are_reachable_from_the_top_level_package():
    import rupsycho.callbacks as callbacks

    assert rup.callbacks is callbacks  # sub-packages are resolved lazily on first access
    assert rup.callbacks.CSVCallback is CSVCallback


def test_callback_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        Callback()  # type: ignore[abstract]


def test_a_subclass_without_save_answer_cannot_be_instantiated():
    class Incomplete(Callback):
        pass

    with pytest.raises(TypeError, match="save_answer"):
        Incomplete()  # type: ignore[abstract]


def test_the_abstract_save_answer_does_nothing():
    class Delegating(Callback):
        def save_answer(self, *args):
            return super().save_answer(*args)

    assert Delegating().save_answer(EXPERIMENT, 0, make_item(), "m", "p", "1", 0.1, "a") is None


@pytest.mark.parametrize(
    "callback_class", [Callback, JSONLCallback, CSVCallback, PrintCallback, PrintTableCallback]
)
def test_save_answer_signature_includes_the_elapsed_time(callback_class):
    """The run loop passes ``time``; callbacks that lack it silently dropped every answer."""
    assert list(inspect.signature(callback_class.save_answer).parameters) == SAVE_ANSWER_PARAMETERS


def test_run_loop_calls_callbacks_with_the_documented_arguments(fake_experiment):
    recorder = Recorder()
    fake_experiment.run(callbacks=[recorder])

    items = fake_experiment.questionnaire.instruction_items
    profiles = list(fake_experiment.demographic_profiles)
    assert len(recorder.calls) == len(items) * len(profiles)

    expected_order = [(i, profile) for i in range(len(items)) for profile in profiles]
    assert [(c[1], c[4]) for c in recorder.calls] == expected_order
    for experiment, item_id, item, model_id, profile_id, seed, elapsed, answer in recorder.calls:
        assert experiment is fake_experiment
        assert item is items[item_id]
        assert (model_id, seed, answer) == ("fake", "7", "3")
        assert profile_id in profiles
        assert isinstance(elapsed, float)
        assert elapsed >= 0


def test_a_failing_callback_only_warns_and_the_others_still_run(fake_experiment):
    class Exploding(Callback):
        def save_answer(self, *args):
            raise RuntimeError("disk full")

    recorder = Recorder()
    with pytest.warns(UserWarning, match="Error while saving answer: disk full"):
        fake_experiment.run(callbacks=[Exploding(), recorder])

    assert len(recorder.calls) == 8


# ===========================================================================
# JSONLCallback
# ===========================================================================


def test_jsonl_callback_writes_one_json_object_per_answer(tmp_path):
    path = tmp_path / "out.jsonl"
    callback = JSONLCallback(str(path))

    save(callback, "3", item_id=0, profile_id="profile-1", seed="7", time=0.25)
    save(callback, "5", item_id=1, item=make_item("Second?"), profile_id="profile-2", seed="8")

    first, second = read_jsonl(path)
    assert first == {
        "experiment_name": "Demo experiment",
        "instruction_item_id": 0,
        "instruction_item": first["instruction_item"],
        "model_id": "model-a",
        "profile_id": "profile-1",
        "random_seed": "7",
        "time": 0.25,
        "answer": "3",
    }
    assert first["instruction_item"]["question"] == "How are you?"
    assert first["instruction_item"]["reversed"] is False
    assert first["instruction_item"]["answer_options"] is None
    assert first["instruction_item"]["attributes"] == {"dimension": "1"}
    assert (second["instruction_item_id"], second["answer"], second["profile_id"]) == (
        1,
        "5",
        "profile-2",
    )
    assert second["instruction_item"]["question"] == "Second?"


def test_jsonl_callback_creates_the_file_lazily_and_appends(tmp_path):
    path = tmp_path / "out.jsonl"
    callback = JSONLCallback(str(path))
    assert not path.exists()

    save(callback, "1")
    save(JSONLCallback(str(path)), "2")  # a second instance appends to the same file

    assert [row["answer"] for row in read_jsonl(path)] == ["1", "2"]


def test_jsonl_callback_default_file_is_experiment_output_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    callback = JSONLCallback()
    assert callback.file_path == "experiment_output.jsonl"

    save(callback, "3")

    assert (tmp_path / "experiment_output.jsonl").is_file()


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param("line one\nline two\r\nline three", id="line-breaks"),
        pytest.param('She said "3", then {"answer": 4}', id="quotes-and-braces"),
        pytest.param("Grüße \U0001f60a 日本語", id="unicode"),
        pytest.param("", id="empty"),
        pytest.param("x" * 100_000, id="very-long"),
    ],
)
def test_jsonl_callback_keeps_every_answer_on_a_single_line(tmp_path, answer):
    path = tmp_path / "out.jsonl"
    save(JSONLCallback(str(path)), answer)

    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    assert read_jsonl(path)[0]["answer"] == answer


def test_jsonl_callback_stores_a_failed_generation_as_null(tmp_path):
    path = tmp_path / "out.jsonl"
    save(JSONLCallback(str(path)), None)

    assert read_jsonl(path)[0]["answer"] is None


def test_jsonl_callback_does_not_embed_the_answers_collected_so_far(tmp_path):
    path = tmp_path / "out.jsonl"
    item = make_item()
    item.update_answer("model-a", "profile-1", "7", "first answer")
    save(JSONLCallback(str(path)), "second answer", item=item, profile_id="profile-2")

    row = read_jsonl(path)[0]
    assert "answers" not in row["instruction_item"]
    assert "first answer" not in path.read_text(encoding="utf-8")
    assert row["answer"] == "second answer"


def test_jsonl_callback_describes_the_item_by_its_configuration_only(tmp_path):
    path = tmp_path / "out.jsonl"
    item = InstructionItem(
        question="Q?",
        reversed=True,
        answer_options={"1": {"text": "1. no", "weight": 1}, "2": {"text": "2. yes", "weight": 2}},
        attributes={"dimension": "3"},
    )
    save(JSONLCallback(str(path)), "2", item=item)

    assert read_jsonl(path)[0]["instruction_item"] == item.model_dump(exclude={"answers"})
    assert set(read_jsonl(path)[0]["instruction_item"]) == {
        "question",
        "reversed",
        "answer_options",
        "attributes",
    }


def test_jsonl_rows_keep_their_size_while_the_item_collects_answers(tmp_path):
    # the accumulated answers used to be embedded in every row, so the file grew quadratically
    path = tmp_path / "out.jsonl"
    callback = JSONLCallback(str(path))
    item = make_item()
    for seed in range(60):
        item.update_answer("model-a", "profile-1", f"{seed:03d}", "an answer of some length")
        save(callback, "an answer of some length", item=item, seed=f"{seed:03d}")

    assert len({len(line) for line in path.read_text(encoding="utf-8").splitlines()}) == 1


# ===========================================================================
# CSVCallback
# ===========================================================================


def test_csv_callback_writes_the_header_when_it_creates_the_file(tmp_path):
    path = tmp_path / "out.csv"
    CSVCallback(str(path))

    assert read_csv_rows(path) == [CSV_HEADER]


def test_csv_callback_default_file_is_experiment_output_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    callback = CSVCallback()

    assert callback.file_path == "experiment_output.csv"
    assert read_csv_rows(tmp_path / "experiment_output.csv") == [CSV_HEADER]


def test_csv_callback_appends_rows_in_the_documented_column_order(tmp_path):
    path = tmp_path / "out.csv"
    callback = CSVCallback(str(path))

    save(callback, "3", item_id=0, profile_id="profile-1", seed="7", time=0.25)
    save(callback, "5", item_id=2, item=make_item("Other?"), model_id="model-b", seed=8, time=1)

    assert read_csv_rows(path) == [
        CSV_HEADER,
        ["Demo experiment", "0", "How are you?", "model-a", "profile-1", "7", "0.25", "3"],
        ["Demo experiment", "2", "Other?", "model-b", "profile-1", "8", "1", "5"],
    ]


def test_csv_callback_keeps_an_existing_file_and_does_not_repeat_the_header(tmp_path):
    path = tmp_path / "out.csv"
    save(CSVCallback(str(path)), "1")

    save(CSVCallback(str(path)), "2")  # e.g. a resumed experiment

    rows = read_csv_rows(path)
    assert [row[-1] for row in rows] == ["answer", "1", "2"]
    assert rows.count(CSV_HEADER) == 1


def test_csv_callback_writes_the_header_into_an_existing_empty_file(tmp_path):
    path = tmp_path / "out.csv"
    path.touch()

    save(CSVCallback(str(path)), "3")

    assert read_csv_rows(path) == [
        CSV_HEADER,
        ["Demo experiment", "0", "How are you?", "model-a", "profile-1", "7", "0.25", "3"],
    ]


def test_csv_callback_does_not_touch_the_content_of_a_pre_existing_file(tmp_path):
    path = tmp_path / "out.csv"
    path.write_text("my,own,header\n1,2,3\n", encoding="utf-8")

    save(CSVCallback(str(path)), "x")

    assert path.read_text(encoding="utf-8").startswith("my,own,header\n1,2,3\n")


def test_csv_callback_needs_an_existing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        CSVCallback(str(tmp_path / "missing" / "out.csv"))


def test_csv_callback_replaces_line_breaks_in_the_question(tmp_path):
    path = tmp_path / "out.csv"
    save(CSVCallback(str(path)), "3", item=make_item("First line\nsecond line"))

    assert read_csv_rows(path)[1][2] == "First line second line"


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param("plain", id="plain"),
        pytest.param("a, b, c", id="commas"),
        pytest.param('She said "3"', id="quotes"),
        pytest.param("line one\nline two", id="line-break"),
        pytest.param("Grüße \U0001f60a 日本語", id="unicode"),
        pytest.param(" padded ", id="padding-is-kept"),
        pytest.param("x" * 100_000, id="very-long"),
    ],
)
def test_csv_callback_round_trips_awkward_answers(tmp_path, answer):
    path = tmp_path / "out.csv"
    save(CSVCallback(str(path)), answer)

    assert read_csv_rows(path)[1][-1] == answer
    assert len(pd.read_csv(path)) == 1


def test_csv_callback_writes_a_failed_generation_as_an_empty_cell(tmp_path):
    path = tmp_path / "out.csv"
    save(CSVCallback(str(path)), None)

    assert read_csv_rows(path)[1][-1] == ""


def test_csv_callback_output_can_be_read_back_with_pandas(tmp_path):
    path = tmp_path / "out.csv"
    callback = CSVCallback(str(path))
    save(callback, "3", item_id=4, seed="9", time=0.5)

    frame = pd.read_csv(path, dtype={"answer": "string", "random_seed": "string"})

    assert list(frame.columns) == CSV_HEADER
    assert frame.loc[0, "instruction_item_id"] == 4
    assert frame.loc[0, "random_seed"] == "9"
    assert frame.loc[0, "time"] == 0.5
    assert frame.loc[0, "answer"] == "3"


def _use_a_cp1252_locale(monkeypatch):
    """Make ``open()`` behave like on a Windows machine when no encoding is given."""
    real_open = builtins.open

    def cp1252_open(file, mode="r", buffering=-1, encoding=None, *args, **kwargs):
        if "b" not in mode and encoding is None:
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", cp1252_open)


def test_csv_callback_writes_non_ascii_answers_whatever_the_locale_encoding(tmp_path, monkeypatch):
    path = tmp_path / "out.csv"
    with monkeypatch.context() as patch:
        _use_a_cp1252_locale(patch)
        callback = CSVCallback(str(path))
        save(callback, "I feel \U0001f60a", item=make_item("Fühlst du dich gut?"))

    rows = read_csv_rows(path)
    assert rows[1][-1] == "I feel \U0001f60a"
    assert rows[1][2] == "Fühlst du dich gut?"
    assert path.read_bytes().decode("utf-8")  # the file really is UTF-8


# ===========================================================================
# PrintCallback
# ===========================================================================


def test_print_callback_prints_every_detail(capsys):
    save(PrintCallback(), "because", item_id=3, item=make_item("Why?"), seed=5, time=1.234)

    assert capsys.readouterr().out.splitlines() == [
        "Experiment: Demo experiment",
        "Instruction Item ID: 3",
        "Instruction Item: Why?",
        "Model ID: model-a",
        "Profile ID: profile-1",
        "Random Seed: 5",
        "Time: 1.234s",
        "Answer: because",
        "=" * 50,
    ]


def test_print_callback_prints_a_failed_generation(capsys):
    save(PrintCallback(), None)

    assert "Answer: None" in capsys.readouterr().out.splitlines()


# ===========================================================================
# PrintTableCallback
# ===========================================================================

COLUMN_WIDTHS = (15, 20, 20, 12, 25, 25)


def table_row(*cells):
    """A table line, padded column by column (independent of the f-string used in the source)."""
    return " | ".join(str(cell).ljust(width) for cell, width in zip(cells, COLUMN_WIDTHS))


TABLE_HEADER = table_row(
    "Instruction ID",
    "Model ID",
    "Profile ID",
    "Random Seed",
    "Question (truncated)",
    "Answer (truncated)",
)


def test_print_table_callback_prints_the_header_once(capsys):
    callback = PrintTableCallback()
    assert callback.headers_printed is False

    save(callback, "1")
    save(callback, "2")
    save(callback, "3")

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == TABLE_HEADER
    assert lines[1] == "=" * len(TABLE_HEADER)
    assert len(lines) == 2 + 3
    assert lines.count(TABLE_HEADER) == 1
    assert callback.headers_printed is True


def test_print_table_callback_headers_are_per_instance(capsys):
    save(PrintTableCallback(), "1")
    save(PrintTableCallback(), "1")

    assert capsys.readouterr().out.splitlines().count(TABLE_HEADER) == 2


def test_print_table_callback_row_layout(capsys):
    save(PrintTableCallback(), "3", item_id=12, item=make_item("Short?"), seed=7)

    row = capsys.readouterr().out.splitlines()[2]
    assert row == table_row(12, "model-a", "profile-1", 7, "Short?", "3")


@pytest.mark.parametrize(
    ("question", "answer", "shown_question", "shown_answer"),
    [
        pytest.param("q" * 25, "a" * 25, "q" * 25, "a" * 25, id="at-the-limit"),
        pytest.param("q" * 26, "a" * 26, "q" * 22 + "...", "a" * 22 + "...", id="just-over"),
        pytest.param("q" * 40, "a" * 40, "q" * 22 + "...", "a" * 22 + "...", id="over-the-limit"),
        pytest.param("one\ntwo\rthree", "x\ny\rz", "one two three", "x y z", id="line-breaks"),
        pytest.param("", "", "", "", id="empty"),
        pytest.param("Grüße \U0001f60a", "日本語", "Grüße \U0001f60a", "日本語", id="unicode"),
    ],
)
def test_print_table_callback_truncates_long_cells(
    capsys, question, answer, shown_question, shown_answer
):
    save(PrintTableCallback(), answer, item=make_item(question))

    row = capsys.readouterr().out.splitlines()[2]
    assert row == table_row(0, "model-a", "profile-1", "7", shown_question, shown_answer)


def test_print_table_callback_long_answer_with_line_breaks_stays_on_one_line(capsys):
    save(PrintTableCallback(), "first line\nsecond line\nthird line which is long")

    assert len(capsys.readouterr().out.splitlines()) == 3


def test_print_table_callback_prints_a_row_for_a_failed_generation(capsys):
    save(PrintTableCallback(), None)

    row = capsys.readouterr().out.splitlines()[2]
    assert row == table_row(0, "model-a", "profile-1", "7", "How are you?", "(no answer)")


@pytest.mark.parametrize("length", [1, 22, 23, 24, 25])
def test_print_table_callback_shows_questions_that_fit_the_column_in_full(capsys, length):
    question = "q" * length
    save(PrintTableCallback(), "3", item=make_item(question))

    assert question in capsys.readouterr().out


def test_print_table_callback_keeps_the_columns_aligned_for_long_identifiers(capsys):
    callback = PrintTableCallback()
    save(callback, "3", model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", profile_id="P" * 40)
    save(callback, "3", model_id="fake", profile_id="p")

    header, _, long_row, short_row = capsys.readouterr().out.splitlines()

    separators = [index for index, char in enumerate(header) if char == "|"]
    assert len(separators) == 5
    for row in (long_row, short_row):
        assert [index for index, char in enumerate(row) if char == "|"] == separators


@pytest.mark.parametrize("length", [26, 27, 100])
def test_print_table_callback_shortens_questions_that_exceed_the_column(capsys, length):
    save(PrintTableCallback(), "3", item=make_item("q" * length))

    row = capsys.readouterr().out.splitlines()[2]
    assert row == table_row(0, "model-a", "profile-1", "7", "q" * 22 + "...", "3")


@pytest.mark.parametrize(
    ("answer", "shown"),
    [(3, "3"), (2.5, "2.5"), ("x" * 25, "x" * 25), ("x" * 26, "x" * 22 + "..."), ("", "")],
    ids=["int", "float", "at-the-limit", "over-the-limit", "empty"],
)
def test_print_table_callback_shows_answers_of_any_type(capsys, answer, shown):
    save(PrintTableCallback(), answer)

    row = capsys.readouterr().out.splitlines()[2]
    assert row == table_row(0, "model-a", "profile-1", "7", "How are you?", shown)


# ===========================================================================
# all callbacks together on a real run
# ===========================================================================


def test_all_callbacks_agree_with_the_collected_answers(fake_experiment, tmp_path, capsys):
    jsonl_path, csv_path = tmp_path / "out.jsonl", tmp_path / "out.csv"
    fake_experiment.run(
        callbacks=[
            JSONLCallback(str(jsonl_path)),
            CSVCallback(str(csv_path)),
            PrintCallback(),
            PrintTableCallback(),
        ]
    )

    frame = fake_experiment.get_answers_as_dataframe()
    expected = sorted(
        zip(frame["Instruction ID"], frame["Persona ID"], frame["Run Seed"], frame["Answer"])
    )

    csv_rows = read_csv_rows(csv_path)[1:]
    assert sorted((int(r[1]), r[4], r[5], r[7]) for r in csv_rows) == expected

    jsonl_rows = read_jsonl(jsonl_path)
    assert (
        sorted(
            (r["instruction_item_id"], r["profile_id"], r["random_seed"], r["answer"])
            for r in jsonl_rows
        )
        == expected
    )

    lines = capsys.readouterr().out.splitlines()
    assert lines.count("Answer: 3") == len(expected)
    assert lines.count(TABLE_HEADER) == 1
    assert sum(1 for line in lines if " | fake " in line) == len(expected)


# ===========================================================================
# failed model calls and concurrency
# ===========================================================================

# 8 calls: the BFI experiment asks item 0 for both profiles, then item 1, ...
SCRIPT = ["1", FAILURE, "3", "4", FAILURE, "2", "5", "1"]
FAILED_CALLS = [1, 4]


@pytest.fixture
def scripted_experiment(fake_experiment):
    """The BFI experiment with a model whose second and fifth call fail."""
    fake_experiment.clear_models()
    fake_experiment.add_model(ScriptedLLM(responses=list(SCRIPT)), identifier="scripted")
    return fake_experiment


def test_callbacks_receive_none_for_failed_model_calls(scripted_experiment):
    recorder = Recorder()

    summary = scripted_experiment.run(callbacks=[recorder], on_error="ignore", show_progress=False)

    assert (summary.n_calls, summary.n_failed) == (8, 2)
    assert [call[-1] for call in recorder.calls] == [None if a == FAILURE else a for a in SCRIPT]
    # a failed call still reports how long it took
    assert all(isinstance(call[-2], float) and call[-2] >= 0 for call in recorder.calls)


def test_every_callback_copes_with_failed_model_calls(scripted_experiment, tmp_path, capsys):
    jsonl_path, csv_path = tmp_path / "out.jsonl", tmp_path / "out.csv"

    scripted_experiment.run(
        callbacks=[
            JSONLCallback(str(jsonl_path)),
            CSVCallback(str(csv_path)),
            PrintCallback(),
            PrintTableCallback(),
        ],
        on_error="ignore",
        show_progress=False,
    )

    expected = [None if answer == FAILURE else answer for answer in SCRIPT]
    assert [row["answer"] for row in read_jsonl(jsonl_path)] == expected
    assert [row[-1] for row in read_csv_rows(csv_path)[1:]] == [a or "" for a in expected]
    lines = capsys.readouterr().out.splitlines()
    assert lines.count("Answer: None") == len(FAILED_CALLS)
    assert sum("(no answer)" in line for line in lines) == len(FAILED_CALLS)
    # no failure may stop the callbacks: all rows of all four callbacks are there
    assert sum(" | scripted " in line for line in lines) == len(SCRIPT)


def test_a_failed_call_does_not_reach_the_stored_answers(scripted_experiment):
    scripted_experiment.run(callbacks=[], on_error="ignore", show_progress=False)

    frame = scripted_experiment.get_answers_as_dataframe()
    assert sorted(frame["Answer"]) == sorted(a for a in SCRIPT if a != FAILURE)


def test_the_run_summary_and_the_callbacks_agree_about_the_failures(scripted_experiment):
    recorder = Recorder()

    with pytest.warns(RuntimeWarning, match="failed"):
        summary = scripted_experiment.run(callbacks=[recorder], show_progress=False)

    assert summary.n_failed == sum(call[-1] is None for call in recorder.calls) == 2


def test_callbacks_are_called_in_the_grid_order_and_on_the_calling_thread_when_calls_overlap(
    config_dict,
):
    def run(workers):
        experiment = rup.experiment_from_dict(copy.deepcopy(config_dict))
        experiment.add_model(PromptLengthLLM(delay=0.01), identifier="m")
        recorder = Recorder()
        experiment.run(callbacks=[recorder], max_concurrency=workers, show_progress=False)
        return recorder

    sequential, concurrent = run(1), run(4)

    def summary(recorder):
        return [(call[1], call[4], call[7]) for call in recorder.calls]

    assert summary(concurrent) == summary(sequential)
    assert len({answer for _, _, answer in summary(sequential)}) > 1  # the order is observable
    # file-writing callbacks never have to lock: they are never called from a worker thread
    assert set(concurrent.threads) == {threading.get_ident()}


def test_all_rows_of_a_multi_seed_run_have_the_same_shape(config_dict, tmp_path):
    config = copy.deepcopy(config_dict)
    config["parameters"]["seeds"] = ["1", "2", "3"]
    experiment = rup.experiment_from_dict(config)
    experiment.add_model(FakeListLLM(responses=["3"]), identifier="fake")
    jsonl_path, csv_path = tmp_path / "out.jsonl", tmp_path / "out.csv"

    summary = experiment.run(
        callbacks=[JSONLCallback(str(jsonl_path)), CSVCallback(str(csv_path))], show_progress=False
    )

    assert summary.n_calls == 4 * 2 * 3
    rows = read_jsonl(jsonl_path)
    assert len(rows) == len(read_csv_rows(csv_path)) - 1 == summary.n_calls
    assert all("answers" not in row["instruction_item"] for row in rows)
    assert {row["random_seed"] for row in rows} == {"1", "2", "3"}
    # the rows of one item do not grow while the item collects answers of further seeds/personas
    lengths_per_item = {}
    for row in rows:
        lengths_per_item.setdefault(row["instruction_item_id"], set()).add(
            len(json.dumps(row["instruction_item"]))
        )
    assert all(len(lengths) == 1 for lengths in lengths_per_item.values())
