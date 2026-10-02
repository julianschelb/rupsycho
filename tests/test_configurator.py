"""Tests for the Streamlit configurator (``src/rupsycho_configurator``).

The configurator is an optional extra (``pip install "rupsycho[configurator]"``), so this module
is skipped when Streamlit, pypdf, openai or langchain-openai are not installed.  Everything runs
offline: the app is driven headlessly with ``streamlit.testing.v1.AppTest``, PDFs are generated
in memory, and the OpenAI client / chat model used by the app are replaced by fakes.

A test marked ``xfail(strict=True)`` describes the *correct* behaviour of a known remaining bug in
the configurator.  It keeps the suite green today and starts failing (XPASS) once the bug is
fixed; remove the marker then.
"""

import builtins
import copy
import importlib.util
import io
import json
import os
import random
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("pypdf")
pytest.importorskip("openai")
pytest.importorskip("langchain_openai")

import httpx  # noqa: E402
import langchain_openai  # noqa: E402
import openai  # noqa: E402
import streamlit  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from langchain_core.prompt_values import ChatPromptValue  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import rupsycho as rup  # noqa: E402
from rupsycho.models.questionnaire import Questionnaire  # noqa: E402
from rupsycho_configurator import launcher, utils  # noqa: E402
from rupsycho_configurator.config_questionnaire import ConfigQuestionnaire  # noqa: E402

APP_PATH = Path(launcher.__file__).parent / "rupsycho_experiment_configurator.py"

needs_upload_support = pytest.mark.skipif(
    not hasattr(AppTest, "file_uploader"),
    reason="AppTest only supports st.file_uploader in newer Streamlit versions",
)


# ===========================================================================
#                                  Helpers
# ===========================================================================


def make_pdf(pages: list[str]) -> bytes:
    """Return a minimal valid PDF whose n-th page shows the (newline separated) lines of pages[n]."""
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    page_ids = []
    for index, text in enumerate(pages):
        page_id, content_id = 4 + 2 * index, 5 + 2 * index
        page_ids.append(page_id)
        operators = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
        for line in text.split("\n"):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            operators.append(f"({escaped}) Tj T*")
        operators.append("ET")
        stream = "\n".join(operators).encode("latin-1")
        objects[content_id] = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode()
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode()

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for number in sorted(objects):
        offsets[number] = len(pdf)
        pdf += b"%d 0 obj\n%s\nendobj\n" % (number, objects[number])
    xref_position = len(pdf)
    size = max(objects) + 1
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for number in range(1, size):
        pdf += b"%010d 00000 n \n" % offsets[number]
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref_position)
    return bytes(pdf)


def squash(text: str) -> str:
    """Collapse all whitespace so assertions do not depend on the PDF text extractor's layout."""
    return " ".join(text.split())


def llm_answer(title="BFI", instructions="Rate each statement.", questions=None, answers=None):
    """A language model answer in the format that ``utils.mk_prompt`` asks for."""
    questions = questions or ["Is talkative", "Is reserved", "Is helpful"]
    answers = answers or [["1. Disagree", "2. Neutral", "3. Agree"]]
    payload = json.dumps([title, instructions, questions, answers])
    return f"Sure, here is the questionnaire:\n```json\n{payload}\n```\nAnything else?"


# ===========================================================================
#                                  Fixtures
# ===========================================================================


@pytest.fixture
def bfi_config(config_dict):
    """The BFI test configuration with the default answer options spelled out for every item."""
    config = copy.deepcopy(config_dict)
    questionnaire = config["questionnaire"]
    for item in questionnaire["instruction_items"]:
        item["answer_options"] = copy.deepcopy(questionnaire["default_answer_options"])
    return config


@pytest.fixture(scope="module")
def pdf_bytes():
    pages = [
        "Title page",
        '1. I am "talkative" and/or loud\n2. I am reserved',
        "Answer options: agree, disagree",
    ]
    return make_pdf(pages)


@pytest.fixture
def downloads(monkeypatch):
    """Record the arguments of every ``st.download_button`` call (the file offered to the user)."""
    state = SimpleNamespace(calls=[])
    original = streamlit.download_button

    def spy(*args, **kwargs):
        state.calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(streamlit, "download_button", spy)
    return state


@pytest.fixture
def app(downloads):
    """A freshly started configurator app (every test gets its own session state)."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.fixture
def fake_openai():
    """Replace ``openai.OpenAI`` so that API keys can be 'validated' without network access.

    Request this fixture *before* ``app`` so the fake is in place from the first run on.
    """
    state = SimpleNamespace(outcome="valid", api_keys=[])

    class FakeOpenAI:
        def __init__(self, api_key=None, **kwargs):
            state.api_keys.append(api_key)
            self.models = self

        def list(self):
            if state.outcome == "rejected":
                request = httpx.Request("GET", "https://api.openai.com/v1/models")
                response = httpx.Response(401, request=request)
                raise openai.AuthenticationError("bad key", response=response, body=None)
            if state.outcome == "offline":
                raise ConnectionError("network is unreachable")
            return []

    original_client = openai.OpenAI
    saved_key = os.environ.pop("OPENAI_API_KEY", None)
    openai.OpenAI = FakeOpenAI
    try:
        yield state
    finally:
        openai.OpenAI = original_client
        # the app exports a validated key as environment variable: leave no trace of it
        os.environ.pop("OPENAI_API_KEY", None)
        if saved_key is not None:
            os.environ["OPENAI_API_KEY"] = saved_key


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace ``ChatOpenAI`` by a fake chat model that answers with ``fake_llm.response``
    (or fails with ``fake_llm.error`` if that is set).

    Request this fixture *before* ``app``: the app imports ``ChatOpenAI`` on every run.
    """
    state = SimpleNamespace(response=llm_answer(), kwargs=None, prompts=[], error=None)

    class FakeChatOpenAI(FakeListChatModel):
        def __init__(self, **kwargs):
            state.kwargs = kwargs
            super().__init__(responses=[state.response])

        def _generate(self, messages, *args, **kwargs):
            state.prompts.append(messages)
            if state.error is not None:
                raise state.error
            return super()._generate(messages, *args, **kwargs)

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    return state


# ===========================================================================
#                          Helpers to drive the app
# ===========================================================================


def result(at):
    """The configuration that the app currently shows as a dict."""
    value = at.json[0].value
    return json.loads(value) if isinstance(value, str) else value


def items_of(config):
    """Questionnaire items as (question, reversed, [(key, text, weight), ...]) tuples."""
    return [
        (
            item["question"],
            item["reversed"],
            [(key, o["text"], o["weight"]) for key, o in item["answer_options"].items()],
        )
        for item in config["questionnaire"]["instruction_items"]
    ]


def profiles_of(config):
    """Profiles as (key, title, name, ethnicity) tuples."""
    return [
        (key, p["attributes"]["title"], p["attributes"]["name"], p["attributes"]["ethnicity"])
        for key, p in config["demographic_profiles"].items()
    ]


def click(at, label):
    """Click the (unkeyed) button with the given label."""
    next(button for button in at.button if button.label == label).click().run()


def add_answer_button(at, item):
    """The 'Add answer option' button of the item-th item (0-based position, not widget key)."""
    buttons = [b for b in at.button if (b.key or "").startswith("add_ans_button_item")]
    return buttons[item]


def delete_answer_button(at, item, option):
    """The delete button of the option-th answer option of the item-th item (both 0-based)."""
    by_item: dict[str, list] = {}
    for button in at.button:
        match = re.fullmatch(r"del_ans_(\d+)item_(\d+)", button.key or "")
        if match:
            by_item.setdefault(match[2], []).append(button)
    return list(by_item.values())[item][option]


def weights_of(config):
    """The weights of the answer options of every item: [[w, ...], ...]."""
    return [[weight for _, _, weight in options] for _, _, options in items_of(config)]


def upload(at, key, name, content, mime):
    """Upload a file to the file uploader widget with the given key and rerun the app."""
    at.file_uploader(key=key).upload(name, content, mime).run()
    assert not at.exception, [e.value for e in at.exception]


def errors(at):
    return [e.value for e in at.error]


def successes(at):
    return [s.value for s in at.success]


def assert_loads_in_rupsycho(config):
    """The exported configuration must be accepted by rupsycho (models stripped: no downloads)."""
    config = copy.deepcopy(config)
    config["models"] = {}
    experiment = rup.experiment_from_dict(config)
    assert experiment.questionnaire is not None
    return experiment


# ===========================================================================
#                              utils: extract_json
# ===========================================================================


def test_extract_json_reads_fenced_json():
    assert utils.extract_json('```json\n["a", "b"]\n```') == ["a", "b"]


def test_extract_json_ignores_text_around_the_fence():
    text = 'Sure! Here you go:\n```json\n["a", {"b": 1}]\n```\nLet me know if you need more.'
    assert utils.extract_json(text) == ["a", {"b": 1}]


def test_extract_json_returns_the_first_block():
    text = '```json\n["first"]\n```\nand also\n```json\n["second"]\n```'
    assert utils.extract_json(text) == ["first"]


def test_extract_json_accepts_chat_messages():
    message = AIMessage(content='```json\n["x", "y"]\n```')
    assert utils.extract_json(message) == ["x", "y"]


def test_extract_json_parses_the_four_part_answer_of_the_prompt():
    title, instructions, questions, answers = utils.extract_json(llm_answer())
    assert (title, instructions) == ("BFI", "Rate each statement.")
    assert questions == ["Is talkative", "Is reserved", "Is helpful"]
    assert answers == [["1. Disagree", "2. Neutral", "3. Agree"]]


@pytest.mark.parametrize(
    "text",
    ['["a", "b"]', "I do not know.", "", '```json\n["a", "b"]'],
    ids=["plain-json", "prose-only", "empty", "unterminated-fence"],
)
def test_extract_json_requires_a_json_fence(text):
    with pytest.raises(ValueError, match="No JSON wrapper") as excinfo:
        utils.extract_json(text)

    assert excinfo.value.__cause__ is None  # nothing was parsed, so there is no underlying error


@pytest.mark.parametrize(
    "body",
    ['["a", "b"', '["a", "b",]', "['a', 'b']", "", "not json"],
    ids=["missing-bracket", "trailing-comma", "single-quotes", "empty-fence", "garbage"],
)
def test_extract_json_rejects_malformed_json(body):
    with pytest.raises(ValueError, match="Invalid JSON") as excinfo:
        utils.extract_json(f"```json\n{body}\n```")

    # the decoder's complaint is chained, and its position/reason is part of the message
    cause = excinfo.value.__cause__
    assert isinstance(cause, json.JSONDecodeError)
    assert str(cause) in str(excinfo.value)


def test_extract_json_reports_where_the_json_is_broken():
    text = '```json\n["fine", "also fine",\n"unterminated]\n```'

    with pytest.raises(ValueError, match=r"Invalid JSON \(.*line 3") as excinfo:
        utils.extract_json(text)

    assert "line 3" in str(excinfo.value.__cause__)


# ===========================================================================
#                    utils: other text helpers and prompt
# ===========================================================================


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('He said "hi" and/or left', "He said hi andor left"),
        ("back\\slash", "backslash"),
        ("nothing to remove", "nothing to remove"),
        ("", ""),
    ],
)
def test_rmv_special_chars_strips_quotes_slashes_and_backslashes(text, expected):
    assert utils.rmv_special_chars(text) == expected


@pytest.mark.parametrize(
    ("text", "nested", "expected"),
    [
        ('noise ["a", "b"] more noise', False, ["a", "b"]),
        ("a python style list ['a', 'b'] works too", False, ["a", "b"]),
        ('[["a", "b"], ["c"]]', True, [["a", "b"], ["c"]]),
        ('[1, 2] and then ["x"]', False, ["x"]),
        ('brackets in strings ["a]", "[b"]', False, ["a]", "[b"]),
        ('# ["comment"]\n["code"]', False, ["code"]),
        ('["flat", "list"]', True, []),
        ("no list here", False, []),
    ],
    ids=["flat", "quotes", "nested", "skip-non-str", "brackets", "comment", "wrong-shape", "none"],
)
def test_extract_json_array(text, nested, expected):
    assert utils.extract_json_array(text, nested) == expected


def test_list_pretty_print_numbers_the_entries(capsys):
    utils.list_pretty_print(["first", "second"], "questions")
    assert capsys.readouterr().out == "\nquestions = [\n    1 - first\n    2 - second\n]\n\n"


def test_mk_prompt_builds_a_system_and_a_human_message():
    quest_text = "1. I am talkative\n2. I am shy\nAnswers: agree, disagree"
    prompt = utils.mk_prompt(quest_text)

    assert isinstance(prompt, ChatPromptValue)
    system, human = prompt.to_messages()
    assert isinstance(system, SystemMessage)
    assert isinstance(human, HumanMessage)
    assert human.content == quest_text
    assert "four elements" in system.content
    assert "```json" in system.content.replace("\\", "")
    assert quest_text in prompt.to_string()  # the app prints the prompt this way


def test_prompt_and_extract_json_form_a_chain_like_the_app_uses():
    model = FakeListChatModel(responses=[llm_answer(title="Chain test")])
    chain = model | utils.extract_json

    title, _, questions, answers = chain.invoke(utils.mk_prompt("1. Is talkative"))

    assert title == "Chain test"
    assert len(questions) == 3
    assert len(answers) == 1


# ===========================================================================
#                              utils: PDF helpers
# ===========================================================================


def test_extract_quest_pages_returns_one_cleaned_text_per_page(pdf_bytes):
    pages = utils.extract_quest_pages(io.BytesIO(pdf_bytes))

    assert [squash(page) for page in pages] == [
        "Title page",
        "1. I am talkative andor loud 2. I am reserved",  # quotes and slash are removed
        "Answer options: agree, disagree",
    ]


def test_extract_quest_pages_accepts_a_path(pdf_bytes, tmp_path):
    path = tmp_path / "questionnaire.pdf"
    path.write_bytes(pdf_bytes)
    assert len(utils.extract_quest_pages(str(path))) == 3


def test_extract_quest_text_defaults_to_all_pages_in_order(pdf_bytes):
    text = squash(utils.extract_quest_text(io.BytesIO(pdf_bytes)))
    assert text == (
        "Title page 1. I am talkative andor loud 2. I am reserved Answer options: agree, disagree"
    )


def test_extract_quest_text_selects_pages_one_based_in_the_given_order(pdf_bytes):
    only_third = utils.extract_quest_text(io.BytesIO(pdf_bytes), [3])
    third_then_first = utils.extract_quest_text(io.BytesIO(pdf_bytes), [3, 1])

    assert squash(only_third) == "Answer options: agree, disagree"
    assert squash(third_then_first) == "Answer options: agree, disagree Title page"


def test_extract_quest_text_accepts_a_path(pdf_bytes, tmp_path):
    path = tmp_path / "questionnaire.pdf"
    path.write_bytes(pdf_bytes)
    assert squash(utils.extract_quest_text(str(path), [1])) == "Title page"


@pytest.mark.parametrize(
    "page",
    [0, -1, -3, 4, 9],
    ids=["zero", "negative", "before-the-first-page", "after-the-last-page", "far-out"],
)
def test_extract_quest_text_rejects_pages_outside_the_document(pdf_bytes, page, capsys):
    # page numbers are 1-based: 0 and negative numbers must not wrap around to the last pages
    assert utils.extract_quest_text(io.BytesIO(pdf_bytes), [page]) == ""

    assert f"{page} is an invalid page number (the PDF has 3 pages)" in capsys.readouterr().out


def test_extract_quest_text_keeps_the_valid_pages_of_a_mixed_selection(pdf_bytes, capsys):
    text = utils.extract_quest_text(io.BytesIO(pdf_bytes), [0, 3, -1, 1, 7])

    assert squash(text) == "Answer options: agree, disagree Title page"
    reported = [
        line.split(" is an invalid page number")[0] for line in capsys.readouterr().out.splitlines()
    ]
    assert reported == ["0", "-1", "7"]  # exactly the invalid pages, in the given order


# ===========================================================================
#                              ConfigQuestionnaire
# ===========================================================================


def test_constructor_stores_the_four_parts():
    quest = ConfigQuestionnaire("Title", "Instructions", ["q1", "q2"], [["a", "b"]])
    assert quest.get_as_list() == ["Title", "Instructions", ["q1", "q2"], [["a", "b"]]]


def test_from_dict_reads_title_instructions_questions_and_answers(bfi_config):
    quest = ConfigQuestionnaire.from_dict(bfi_config)
    source = bfi_config["questionnaire"]

    assert quest.title == source["name"]
    assert quest.instructions == source["general_instruction"]
    assert quest.questions == [item["question"] for item in source["instruction_items"]]
    expected_options = [option["text"] for option in source["default_answer_options"].values()]
    assert quest.answers == [expected_options] * 4


def default_option_texts(config):
    return [option["text"] for option in config["questionnaire"]["default_answer_options"].values()]


def test_from_dict_supports_default_answer_options(config_dict):
    # the repo's own BFI configuration gives the answer options once for all items
    assert all(
        "answer_options" not in item for item in config_dict["questionnaire"]["instruction_items"]
    )

    quest = ConfigQuestionnaire.from_dict(config_dict)

    defaults = default_option_texts(config_dict)
    assert defaults[0] == "1. Disagree strongly" and len(defaults) == 5
    assert len(quest.questions) == 4
    assert quest.answers == [defaults] * 4


def test_from_dict_prefers_the_own_answer_options_of_an_item(config_dict):
    config = copy.deepcopy(config_dict)
    own = {
        "1": {"text": "yes", "weight": 1, "ignored_for_scale": False},
        "2": {"text": "no", "weight": 2, "ignored_for_scale": False},
    }
    config["questionnaire"]["instruction_items"][1]["answer_options"] = own

    quest = ConfigQuestionnaire.from_dict(config)

    defaults = default_option_texts(config)
    assert quest.answers == [defaults, ["yes", "no"], defaults, defaults]


def test_from_dict_without_any_answer_options_gives_empty_answer_sets(bfi_config):
    config = bfi_config
    del config["questionnaire"]["default_answer_options"]
    for item in config["questionnaire"]["instruction_items"]:
        del item["answer_options"]

    quest = ConfigQuestionnaire.from_dict(config)  # no KeyError

    assert quest.answers == [[]] * 4
    assert [question for question, _ in quest.merge_questions_answers()] == quest.questions


def test_default_answer_options_survive_the_conversion_into_a_loadable_experiment(config_dict):
    quest = ConfigQuestionnaire.from_dict(config_dict)
    config = copy.deepcopy(config_dict)
    config["questionnaire"] = quest.mk_dict()["questionnaire"]

    experiment = assert_loads_in_rupsycho(config)

    expected = default_option_texts(config_dict)
    items = experiment.questionnaire.instruction_items
    assert [item.get_answer_options_as_list() for item in items] == [expected] * 4


def test_from_file_reads_a_json_file(bfi_config, tmp_path):
    path = tmp_path / "questionnaire.json"
    path.write_text(json.dumps(bfi_config), encoding="utf-8")

    quest = ConfigQuestionnaire.from_file(str(path))

    assert quest.get_as_list() == ConfigQuestionnaire.from_dict(bfi_config).get_as_list()


def test_mk_dict_is_accepted_by_the_questionnaire_model(bfi_config):
    quest = ConfigQuestionnaire.from_dict(bfi_config)

    questionnaire = Questionnaire(**quest.mk_dict()["questionnaire"])

    assert questionnaire.name == quest.title
    assert questionnaire.general_instruction == quest.instructions
    items = questionnaire.instruction_items
    assert items is not None and len(items) == 4
    assert items[0].question == "I see myself as someone who..."
    assert items[0].get_answer_options_as_list() == quest.answers[0]
    assert items[0].answer_options is not None
    assert list(items[0].answer_options.options) == ["1", "2", "3", "4", "5"]
    assert [o.weight for o in items[0].answer_options.options.values()] == [1, 2, 3, 4, 5]


def test_mk_dict_loads_as_part_of_a_full_experiment(bfi_config):
    quest = ConfigQuestionnaire.from_dict(bfi_config)
    config = copy.deepcopy(bfi_config)
    config["questionnaire"] = quest.mk_dict()["questionnaire"]

    experiment = assert_loads_in_rupsycho(config)

    assert len(experiment.questionnaire.instruction_items) == 4


def test_to_json_file_writes_and_returns_the_questionnaire_section(bfi_config, tmp_path):
    quest = ConfigQuestionnaire.from_dict(bfi_config)
    path = tmp_path / "out.json"

    serialized = quest.to_json_file(str(path))

    assert json.loads(serialized) == quest.mk_dict()
    assert json.loads(path.read_text(encoding="utf-8")) == quest.mk_dict()
    reread = ConfigQuestionnaire.from_file(str(path))  # what was written can be read back
    assert reread.get_as_list() == quest.get_as_list()


@pytest.fixture
def locale_encoding_is_ascii(monkeypatch, tmp_path):
    """Make ``open()`` of a text file in ``tmp_path`` default to ASCII, as it does on a C/POSIX
    locale or with a legacy Windows code page, when no ``encoding`` is given."""
    real_open = builtins.open

    def locale_open(
        file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs
    ):
        is_text_in_tmp_path = (
            "b" not in mode
            and isinstance(file, (str, os.PathLike))
            and Path(file).is_relative_to(tmp_path)
        )
        if is_text_in_tmp_path and encoding is None:
            encoding = "ascii"
        return real_open(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", locale_open)


def test_questionnaire_files_are_read_and_written_as_utf8(tmp_path, locale_encoding_is_ascii):
    quest = ConfigQuestionnaire(
        "Fragebogen über Persönlichkeit – 性格",
        "Bitte ehrlich antworten … danke ☺",
        ["Ich bin gern draußen", "我喜欢读书"],
        [["stimme zu ✓", "lehne ab"], ["sí", "não"]],
    )
    path = tmp_path / "questionnaire.json"

    quest.to_json_file(str(path))
    assert ConfigQuestionnaire.from_file(str(path)).get_as_list() == quest.get_as_list()

    # a configuration exported by rupsycho writes the characters out, as UTF-8
    path.write_bytes(json.dumps(quest.mk_dict(), ensure_ascii=False).encode("utf-8"))
    assert "über".encode() in path.read_bytes()
    assert ConfigQuestionnaire.from_file(str(path)).get_as_list() == quest.get_as_list()


@pytest.mark.parametrize(
    ("questions", "answers", "expected"),
    [
        (["q1", "q2"], [["a", "b"], ["c"]], [("q1", ["a", "b"]), ("q2", ["c"])]),
        (["q1", "q2"], [["a", "b"]], [("q1", ["a", "b"]), ("q2", ["a", "b"])]),
        (["q1", "q2", "q3"], [["a"], ["b"]], [("q1", ["a"]), ("q2", ["b"]), ("q3", [])]),
        (["q1"], [["a"], ["b"], ["c"]], [("q1", ["a"]), ("", ["b"]), ("", ["c"])]),
        (["q1", "q2"], [], [("q1", []), ("q2", [])]),
        ([], [["a"]], []),
        ([], [], []),
    ],
    ids=["individual", "global", "fewer-sets", "more-sets", "no-sets", "no-questions", "empty"],
)
def test_merge_questions_answers(questions, answers, expected):
    quest = ConfigQuestionnaire("T", "I", questions, answers)

    assert quest.merge_questions_answers() == expected
    assert quest.get_merged_questions_answers() == expected


def test_mk_dict_for_unbalanced_questionnaires_stays_valid():
    quest = ConfigQuestionnaire("T", "I", ["q1"], [["a"], ["b", "c"]])

    questionnaire = Questionnaire(**quest.mk_dict()["questionnaire"])

    assert [item.question for item in questionnaire.instruction_items] == ["q1", ""]
    assert questionnaire.instruction_items[1].get_answer_options_as_list() == ["b", "c"]


def test_get_as_dict_is_mk_dict():
    quest = ConfigQuestionnaire("T", "I", ["q1"], [["a", "b"]])
    assert quest.get_as_dict() == quest.mk_dict()


def test_str_lists_title_instructions_questions_and_answer_sets():
    quest = ConfigQuestionnaire("My title", "My instructions", ["q1", "q2"], [["a", "b"]])

    text = str(quest)

    assert "title: My title" in text
    assert "instructions: My instructions" in text
    assert "questionnaire items (2):" in text
    assert "q1\n    q2" in text
    assert "answer sets (1):" in text
    assert "['a', 'b']" in text


def test_from_raw_output_builds_a_questionnaire_from_model_outputs():
    quest = ConfigQuestionnaire.from_raw_output(
        '```json\n["My title", "My instructions"]\n```',
        '```json\n["q1", "q2"]\n```',
        '```json\n[["a", "b"]]\n```',
    )
    assert quest.get_as_list() == ["My title", "My instructions", ["q1", "q2"], [["a", "b"]]]
    assert quest.merge_questions_answers() == [("q1", ["a", "b"]), ("q2", ["a", "b"])]


def test_from_raw_output_accepts_text_around_the_json_blocks():
    quest = ConfigQuestionnaire.from_raw_output(
        'Title and instructions:\n```json\n["T", "I"]\n```\nDone.',
        'Here are the questions:\n```json\n["q1"]\n```',
        'And the answers:\n```json\n[["a"], ["b"]]\n```',
    )
    assert quest.get_as_list() == ["T", "I", ["q1"], [["a"], ["b"]]]


def test_from_raw_output_rejects_unparsable_title_and_instructions():
    with pytest.raises(ValueError, match="title and instructions") as excinfo:
        ConfigQuestionnaire.from_raw_output("no json here", "[]", "[]")

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "No JSON wrapper" in str(excinfo.value.__cause__)


@pytest.mark.parametrize(
    "title_instr",
    ['["only a title"]', '["a", "b", "c"]', '"just text"', "[]"],
    ids=["one-part", "three-parts", "string", "empty"],
)
def test_from_raw_output_requires_exactly_title_and_instructions(title_instr):
    with pytest.raises(ValueError, match="title and instructions") as excinfo:
        ConfigQuestionnaire.from_raw_output(f"```json\n{title_instr}\n```", "[]", "[]")

    assert excinfo.value.__cause__ is not None  # the unpacking error is chained


@pytest.mark.parametrize("broken", ["questions", "answers"])
def test_from_raw_output_reports_unparsable_questions_and_answers(broken):
    outputs = {"questions": '```json\n["q1"]\n```', "answers": '```json\n[["a"]]\n```'}
    outputs[broken] = "```json\n[oops\n```"

    with pytest.raises(ValueError, match="Invalid JSON"):
        ConfigQuestionnaire.from_raw_output(
            '```json\n["T", "I"]\n```', outputs["questions"], outputs["answers"]
        )


@pytest.mark.xfail(
    strict=True,
    reason="from_raw_output stores whatever JSON the model returned (a string of questions, a "
    "flat list of answers, numbers), which merge_questions_answers then splits into single "
    "characters; the app's run_model rejects exactly these shapes with a ValueError",
)
def test_from_raw_output_validates_the_shape_of_the_model_output():
    wrongly_shaped = (
        ('"q1 and q2"', '[["a", "b"]]'),  # the questions must be a list of strings
        ('["q1", 2]', '[["a", "b"]]'),
        ('["q1", "q2"]', '["a", "b"]'),  # the answer sets must be lists of lists of strings
        ('["q1", "q2"]', '[["a", 1]]'),
    )
    for questions, answers in wrongly_shaped:
        try:
            ConfigQuestionnaire.from_raw_output(
                '```json\n["T", "I"]\n```', f"```json\n{questions}\n```", f"```json\n{answers}\n```"
            )
        except ValueError:
            continue
        pytest.fail(f"accepted the questions {questions} with the answers {answers}")


# ===========================================================================
#                                  launcher
# ===========================================================================


@pytest.fixture
def streamlit_process(monkeypatch):
    """Replace the process that the launcher starts: records the commands, 'exits' with .exit_code."""
    state = SimpleNamespace(commands=[], exit_code=0)

    def call(command, *args, **kwargs):
        state.commands.append(list(command))
        return state.exit_code

    def run(command, *args, **kwargs):
        state.commands.append(list(command))
        return SimpleNamespace(returncode=state.exit_code)

    monkeypatch.setattr(subprocess, "call", call)
    monkeypatch.setattr(subprocess, "run", run)  # either may be used to spawn the server
    monkeypatch.setattr(sys, "argv", ["rup-configurator"])
    return state


def test_launcher_starts_streamlit_with_the_app_script(streamlit_process):
    assert launcher.main() == 0

    assert streamlit_process.commands == [[sys.executable, "-m", "streamlit", "run", str(APP_PATH)]]
    assert APP_PATH.is_file()


def test_launcher_passes_command_line_arguments_on_to_streamlit(streamlit_process, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["rup-configurator", "--server.port", "8600", "--server.headless", "true"]
    )

    launcher.main()

    (command,) = streamlit_process.commands
    assert command[:5] == [sys.executable, "-m", "streamlit", "run", str(APP_PATH)]
    assert command[5:] == ["--server.port", "8600", "--server.headless", "true"]


def test_launcher_prefers_explicitly_given_arguments_over_the_command_line(
    streamlit_process, monkeypatch
):
    monkeypatch.setattr(sys, "argv", ["rup-configurator", "--server.port", "8600"])

    launcher.main(["--server.port", "8700"])
    launcher.main([])  # an empty list means "no arguments", not "use the command line"
    launcher.main(argv=("--server.headless", "true"))

    ports, bare, headless = (command[5:] for command in streamlit_process.commands)
    assert ports == ["--server.port", "8700"]
    assert bare == []
    assert headless == ["--server.headless", "true"]


@pytest.mark.parametrize("exit_code", [0, 1, 2, 130])
def test_launcher_returns_the_exit_code_of_streamlit(streamlit_process, exit_code):
    streamlit_process.exit_code = exit_code

    assert launcher.main() == exit_code


def test_launcher_reports_a_missing_streamlit_instead_of_starting_it(
    streamlit_process, monkeypatch, capsys
):
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: None if name == "streamlit" else find_spec(name, *args),
    )

    assert launcher.main() == 1

    captured = capsys.readouterr()
    assert "Streamlit" in captured.err
    assert 'pip install "rupsycho[configurator]"' in captured.err
    assert captured.out == ""
    assert streamlit_process.commands == []  # nothing was started


@pytest.mark.xfail(
    strict=True,
    reason="launcher.main lets the KeyboardInterrupt from subprocess.call escape: stopping the "
    "app with Ctrl+C, as the tutorial advises, prints a Python traceback instead of ending "
    "quietly with the conventional exit code 130 (the 'rupsycho configurator' command does)",
)
def test_launcher_ends_quietly_when_the_app_is_stopped_with_ctrl_c(
    streamlit_process, monkeypatch, capsys
):
    def interrupted(command, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "call", interrupted)
    monkeypatch.setattr(subprocess, "run", interrupted)

    try:
        exit_code = launcher.main()
    except KeyboardInterrupt:  # pytest would treat a KeyboardInterrupt as 'abort the whole session'
        pytest.fail("the KeyboardInterrupt of Ctrl+C escapes from launcher.main()")
    assert exit_code == 130
    assert "Traceback" not in capsys.readouterr().err


@pytest.fixture
def real_streamlit(monkeypatch):
    """Let the launcher run the real Streamlit CLI, but only for commands that end by themselves.

    A launcher that dropped the options below would start a server (and open a browser) that never
    ends, so a command that does not end with ``state.options`` is refused before anything is
    started, and a process that is started is killed after a timeout.
    """
    state = SimpleNamespace(options=[])
    real_call, real_run = subprocess.call, subprocess.run

    def refuse_commands_that_would_start_a_server(command):
        options = state.options
        assert options and list(command[-len(options) :]) == options, command

    def call(command, *args, **kwargs):
        refuse_commands_that_would_start_a_server(command)
        return real_call(command, *args, timeout=60, **kwargs)

    def run(command, *args, **kwargs):
        refuse_commands_that_would_start_a_server(command)
        return real_run(command, *args, timeout=60, **kwargs)

    monkeypatch.setattr(subprocess, "call", call)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("STREAMLIT_SERVER_HEADLESS", "true")  # never open a browser
    monkeypatch.setenv("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    return state


def test_launcher_hands_the_app_and_the_arguments_to_the_real_streamlit_cli(real_streamlit, capfd):
    # no server is started: ``streamlit run <app> --help`` only prints the usage of ``run``
    real_streamlit.options = ["--help"]

    assert launcher.main(["--help"]) == 0

    assert "Usage: streamlit run" in capfd.readouterr().out


def test_launcher_returns_the_failure_of_the_real_streamlit_cli(real_streamlit, capfd):
    # Streamlit rejects the option before it starts anything
    real_streamlit.options = ["--server.port", "not-a-port"]

    exit_code = launcher.main(["--server.port", "not-a-port"])

    assert exit_code != 0
    assert "--server.port" in capfd.readouterr().err


def test_console_script_points_to_the_launcher():
    scripts = {ep.name: ep for ep in metadata.entry_points(group="console_scripts")}
    if "rup-configurator" not in scripts:
        pytest.skip("rupsycho is not installed with its console scripts")
    assert scripts["rup-configurator"].load() is launcher.main


# ===========================================================================
#                      App: start-up and manual configuration
# ===========================================================================


def test_app_starts_with_an_empty_configuration(app):
    assert not app.exception
    assert not app.error
    assert app.title[0].value == "R.U.Psycho Experiment Configurator"
    assert [tab.label.strip("*") for tab in app.tabs] == [
        "Tools",
        "Experiment Info",
        "Demographic Profiles",
        "Questionnaire Info",
        "Questionnaire Items",
        "Questionnaire Text",
        "Resulting Configuration",
    ]

    config = result(app)
    assert list(config) == [
        "name",
        "description",
        "parameters",
        "prompt_template",
        "models",
        "demographic_profiles",
        "questionnaire",
    ]
    assert config["name"] == ""
    assert config["parameters"] == {}
    assert config["models"] == {}
    assert config["prompt_template"]["type"] == "chat"
    assert "{persona_description}" in config["prompt_template"]["messages"][0]["content"]
    assert profiles_of(config) == [("1- ", "", "", "")]
    assert items_of(config) == [("", False, [("1", "", 0), ("2", "", 1)])]


def test_exported_configuration_contains_only_configuration_fields(app, downloads):
    click(app, "Add profile")
    app.button(key="add_item_button").click().run()
    app.text_input(key="question_1").set_value("Second").run()

    config = result(app)

    assert len(config["demographic_profiles"]) == 2
    assert len(config["questionnaire"]["instruction_items"]) == 2
    for profile in config["demographic_profiles"].values():
        assert set(profile) == {"attributes", "template"}
        assert set(profile["attributes"]) == {"title", "name", "ethnicity", "id"}
    for item in config["questionnaire"]["instruction_items"]:
        assert set(item) == {"question", "reversed", "answer_options", "attributes"}
        for option in item["answer_options"].values():
            assert set(option) == {"text", "weight", "ignored_for_scale"}
    # none of the widget bookkeeping of the app ends up in the file that the user downloads
    assert "widget_key" not in json.dumps(config)
    assert "widget_key" not in downloads.calls[-1]["data"]


DOCS_URL = "https://julianschelb.github.io/rupsycho/tutorials/configurator/"


def test_help_points_to_the_documentation_site(app):
    help_text = next(m.value for m in app.markdown if "This configurator facilitates" in m.value)

    assert DOCS_URL in help_text
    assert "README" not in help_text  # the guide is no longer a section of the README
    assert "doi.org/10.48550/arXiv.2503.10229" in help_text  # the paper stays referenced


def test_documentation_page_that_the_help_links_to_exists():
    docs = Path(__file__).parents[1] / "docs"
    if not docs.is_dir():
        pytest.skip("the documentation sources are not part of this installation")

    assert (docs / "tutorials" / "configurator.md").is_file()  # served at DOCS_URL


def test_manual_configuration_is_exported_as_a_loadable_experiment(app):
    # several widgets can be edited before the app reruns
    app.text_input(key="exp_name").set_value("My experiment")
    app.text_area(key="exp_descr").set_value("What it is about")
    app.text_input(key="quest_name").set_value("My questionnaire")
    app.text_area(key="quest_instr").set_value("Please answer honestly.")
    app.text_input(key="title_0").set_value("Dr.")
    app.text_input(key="name_0").set_value("Smith")
    app.text_input(key="ethnicity_0").set_value("Irish")
    app.text_input(key="question_0").set_value("Do you like tests?")
    app.text_input(key="ans_0_item_0").set_value("Yes")
    app.text_input(key="ans_1_item_0").set_value("No")
    app.run()

    click(app, "Add profile")
    app.text_input(key="title_1").set_value("Ms.")
    app.text_input(key="name_1").set_value("Jones")
    app.text_input(key="ethnicity_1").set_value("Welsh")
    app.run()

    app.button(key="add_item_button").click().run()
    app.button(key="add_ans_button_item1").click().run()
    app.text_input(key="question_1").set_value("How often?")
    app.text_input(key="ans_0_item_1").set_value("Never")
    app.text_input(key="ans_1_item_1").set_value("Sometimes")
    app.text_input(key="ans_2_item_1").set_value("Always")
    app.toggle(key="reversed_1").set_value(True)
    app.run()

    assert not app.exception
    config = result(app)
    assert config["name"] == "My experiment"
    assert config["description"] == "What it is about"
    assert config["questionnaire"]["name"] == "My questionnaire"
    assert config["questionnaire"]["general_instruction"] == "Please answer honestly."
    assert profiles_of(config) == [
        ("1-Dr. Smith", "Dr.", "Smith", "Irish"),
        ("2-Ms. Jones", "Ms.", "Jones", "Welsh"),
    ]
    assert [(q, r, [t for _, t, _ in a]) for q, r, a in items_of(config)] == [
        ("Do you like tests?", False, ["Yes", "No"]),
        ("How often?", True, ["Never", "Sometimes", "Always"]),
    ]

    experiment = assert_loads_in_rupsycho(config)
    assert config["parameters"] == {}  # the app does not edit them: rupsycho draws a seed itself
    assert len(experiment.parameters.seeds) == 1
    assert list(experiment.demographic_profiles) == ["1-Dr. Smith", "2-Ms. Jones"]
    assert str(experiment.demographic_profiles["2-Ms. Jones"]) == "Ms. Jones"
    first, second = experiment.questionnaire.instruction_items
    assert (first.reversed, second.reversed) == (False, True)
    assert second.get_answer_options_as_list() == ["Never", "Sometimes", "Always"]


def test_download_button_offers_the_displayed_configuration(app, downloads):
    tricky = 'He said "hi" \\ and left\nnew line / Müller'
    app.text_area(key="exp_descr").set_value(tricky)
    app.text_input(key="question_0").set_value(tricky)
    app.run()

    config = result(app)
    download = downloads.calls[-1]

    assert config["description"] == tricky
    assert config["questionnaire"]["instruction_items"][0]["question"] == tricky
    assert download["file_name"] == "rupsycho_experiment_config.json"
    assert download["mime"] == "application/json"
    assert json.loads(download["data"]) == config  # valid JSON, special characters escaped
    assert_loads_in_rupsycho(config)


def test_profiles_can_be_duplicated_and_deleted(app):
    app.text_input(key="title_0").set_value("Mr.")
    app.text_input(key="name_0").set_value("A")
    app.text_input(key="ethnicity_0").set_value("x")
    app.run()

    app.button(key="duplicate_profile_button_0").click().run()
    assert profiles_of(result(app)) == [("1-Mr. A", "Mr.", "A", "x"), ("2-Mr. A", "Mr.", "A", "x")]

    app.text_input(key="name_1").set_value("B").run()  # the copy is independent of the original
    click(app, "Add profile")
    assert profiles_of(result(app)) == [
        ("1-Mr. A", "Mr.", "A", "x"),
        ("2-Mr. B", "Mr.", "B", "x"),
        ("3- ", "", "", ""),
    ]

    app.button(key="delete_profile_button_1").click().run()
    assert [key for key, *_ in profiles_of(result(app))] == ["1-Mr. A", "2- "]  # ids are renumbered

    app.button(key="delete_profile_button_0").click().run()
    app.button(key="delete_profile_button_2").click().run()
    assert not app.exception
    assert result(app)["demographic_profiles"] == {}


def test_answer_options_can_be_added_and_deleted(app):
    app.button(key="add_ans_button_item0").click().run()
    app.text_input(key="ans_0_item_0").set_value("a1")
    app.text_input(key="ans_1_item_0").set_value("a2")
    app.text_input(key="ans_2_item_0").set_value("a3")
    app.run()
    assert items_of(result(app))[0][2] == [("1", "a1", 0), ("2", "a2", 1), ("3", "a3", 2)]

    app.button(key="del_ans_1item_0").click().run()  # delete the middle one
    assert items_of(result(app))[0][2] == [("1", "a1", 0), ("2", "a3", 1)]

    app.button(key="add_ans_button_item0").click().run()
    assert items_of(result(app))[0][2] == [("1", "a1", 0), ("2", "a3", 1), ("3", "", 2)]
    assert not app.exception


def test_items_can_be_duplicated_and_deleted(app):
    app.text_input(key="question_0").set_value("Q1")
    app.text_input(key="ans_0_item_0").set_value("yes")
    app.text_input(key="ans_1_item_0").set_value("no")
    app.run()

    app.button(key="duplicate_item_button_0").click().run()
    assert [(q, [t for _, t, _ in a]) for q, _, a in items_of(result(app))] == [
        ("Q1", ["yes", "no"]),
        ("Q1", ["yes", "no"]),
    ]

    app.text_input(key="question_1").set_value("Q2").run()  # the copy is independent
    assert [q for q, _, _ in items_of(result(app))] == ["Q1", "Q2"]

    app.button(key="delete_item_button_0").click().run()
    assert [q for q, _, _ in items_of(result(app))] == ["Q2"]
    app.button(key="delete_item_button_1").click().run()
    assert not app.exception
    assert result(app)["questionnaire"]["instruction_items"] == []


def test_duplicating_an_item_keeps_its_reversed_flag(app):
    app.toggle(key="reversed_0").set_value(True).run()
    assert [r for _, r, _ in items_of(result(app))] == [True]

    app.button(key="duplicate_item_button_0").click().run()

    assert [r for _, r, _ in items_of(result(app))] == [True, True]

    # the copy has a switch of its own: turning it off does not touch the original ...
    app.toggle(key="reversed_1").set_value(False).run()
    assert [r for _, r, _ in items_of(result(app))] == [True, False]

    # ... and the copy of a normal item stays a normal item
    app.button(key="duplicate_item_button_1").click().run()
    assert [r for _, r, _ in items_of(result(app))] == [True, False, False]
    assert not app.exception


def test_random_edits_of_items_and_answer_options_match_a_plain_python_model(app):
    """Replay edits on the app and on a simple model of the result, and compare them after each.

    This guards the bookkeeping of widget keys, widget values and weights: a duplicate, a deletion
    or an edit must never change another item or another answer option.
    """
    rng = random.Random(2024)
    model = [{"question": "", "reversed": False, "answers": [["", 0], ["", 1]]}]
    operations = ["add item", "edit question", "toggle", "duplicate", "delete"]
    operations += ["add answer", "edit answer", "delete answer"]
    # the first item is filled in and duplicated, then every kind of edit happens in random order
    prelude = ["edit question", "toggle", "edit answer", "add answer", "duplicate"]
    schedule = [(operation, 0) for operation in prelude]
    for _ in range(4):
        schedule += [(operation, None) for operation in rng.sample(operations, len(operations))]

    def widgets_of(widgets, pattern, item_key):
        return [w for w in widgets if re.fullmatch(pattern.format(item_key), w.key or "")]

    for number, (operation, target) in enumerate(schedule, start=1):
        candidates = list(range(len(model)))
        if operation in ("edit answer", "delete answer"):
            candidates = [i for i in candidates if model[i]["answers"]]
        if operation != "add item" and not candidates:
            continue
        i = target if target is not None else (rng.choice(candidates) if candidates else None)
        keys = [t.key[len("question_") :] for t in app.text_input if t.key.startswith("question_")]

        if operation == "add item":
            app.button(key="add_item_button").click().run()
            model.append({"question": "", "reversed": False, "answers": [["", 0], ["", 1]]})
        elif operation == "edit question":
            app.text_input(key=f"question_{keys[i]}").set_value(f"question {number}").run()
            model[i]["question"] = f"question {number}"
        elif operation == "toggle":
            toggle = app.toggle(key=f"reversed_{keys[i]}")
            toggle.set_value(not toggle.value).run()
            model[i]["reversed"] = not model[i]["reversed"]
        elif operation == "duplicate":
            app.button(key=f"duplicate_item_button_{keys[i]}").click().run()
            model.insert(i + 1, copy.deepcopy(model[i]))
        elif operation == "delete":
            app.button(key=f"delete_item_button_{keys[i]}").click().run()
            del model[i]
        elif operation == "add answer":
            app.button(key=f"add_ans_button_item{keys[i]}").click().run()
            weights = [weight for _, weight in model[i]["answers"]]
            model[i]["answers"].append(["", max(weights, default=-1) + 1])
        elif operation == "edit answer":
            j = rng.randrange(len(model[i]["answers"]))
            widgets_of(app.text_input, r"ans_\d+_item_{}", keys[i])[j].set_value(
                f"answer {number}"
            ).run()
            model[i]["answers"][j][0] = f"answer {number}"
        else:  # delete answer
            j = rng.randrange(len(model[i]["answers"]))
            answers = model[i]["answers"]
            weights = [weight for _, weight in answers]
            was_consecutive = weights == list(range(weights[0], weights[0] + len(weights)))
            widgets_of(app.button, r"del_ans_\d+item_{}", keys[i])[j].click().run()
            del answers[j]
            if was_consecutive:  # a consecutive scale stays consecutive
                for position, answer in enumerate(answers):
                    answer[1] = weights[0] + position

        assert not app.exception, (number, operation)
        expected = [
            (item["question"], item["reversed"], [tuple(a) for a in item["answers"]])
            for item in model
        ]
        shown = [
            (question, reversed_, [(text, weight) for _, text, weight in options])
            for question, reversed_, options in items_of(result(app))
        ]
        assert shown == expected, (number, operation)


def test_random_edits_of_personas_match_a_plain_python_model(app):
    rng = random.Random(2024)
    model = [{"title": "", "name": "", "ethnicity": ""}]
    operations = ["add", "edit title", "edit name", "edit ethnicity", "duplicate", "delete"]
    # the first persona is filled in and duplicated, then every kind of edit happens in random order
    schedule = [("edit title", 0), ("edit name", 0), ("edit ethnicity", 0), ("duplicate", 0)]
    for _ in range(4):
        schedule += [(operation, None) for operation in rng.sample(operations, len(operations))]

    for number, (operation, target) in enumerate(schedule, start=1):
        if operation != "add" and not model:
            continue
        i = target if target is not None else (rng.randrange(len(model)) if model else None)

        if operation == "add":
            click(app, "Add profile")
            model.append({"title": "", "name": "", "ethnicity": ""})
        elif operation.startswith("edit "):
            field = operation.removeprefix("edit ")
            inputs = [t for t in app.text_input if t.key.startswith(f"{field}_")]
            inputs[i].set_value(f"{field} {number}").run()
            model[i][field] = f"{field} {number}"
        elif operation == "duplicate":
            buttons = [b for b in app.button if (b.key or "").startswith("duplicate_profile")]
            buttons[i].click().run()
            model.insert(i + 1, dict(model[i]))
        else:
            buttons = [b for b in app.button if (b.key or "").startswith("delete_profile")]
            buttons[i].click().run()
            del model[i]

        assert not app.exception, (number, operation)
        expected = [
            (f"{n}-{p['title']} {p['name']}", p["title"], p["name"], p["ethnicity"])
            for n, p in enumerate(model, start=1)
        ]
        assert profiles_of(result(app)) == expected, (number, operation)


# ===========================================================================
#                          App: global answer set
# ===========================================================================


def test_global_answer_set_is_applied_to_every_item(app):
    app.button(key="add_item_button").click().run()
    app.text_input(key="question_0").set_value("Q1")
    app.text_input(key="question_1").set_value("Q2")
    app.toggle(key="use_global_answer_set").set_value(True).run()

    assert not app.exception
    assert [t.key for t in app.text_input if t.key.startswith("global_ans_")] == [
        "global_ans_0",
        "global_ans_1",
    ]
    app.button(key="add_global_ans_button").click().run()
    app.text_input(key="global_ans_0").set_value("Agree")
    app.text_input(key="global_ans_1").set_value("Disagree")
    app.text_input(key="global_ans_2").set_value("Neutral")
    app.toggle(key="global_reversed").set_value(True)
    app.run()

    expected = [("1", "Agree", 0), ("2", "Disagree", 1), ("3", "Neutral", 2)]
    assert items_of(result(app)) == [("Q1", True, expected), ("Q2", True, expected)]

    app.button(key="del_global_ans_1").click().run()  # delete 'Disagree'
    expected = [("1", "Agree", 0), ("2", "Neutral", 1)]
    assert items_of(result(app)) == [("Q1", True, expected), ("Q2", True, expected)]

    app.button(key="add_item_button").click().run()  # new items receive the global set, too
    assert [a for _, _, a in items_of(result(app))] == [expected] * 3
    assert_loads_in_rupsycho(result(app))

    # the scale stays 0, 1, 2, ... when options are appended and when the first one is deleted
    app.button(key="add_global_ans_button").click().run()
    expected = [("1", "Agree", 0), ("2", "Neutral", 1), ("3", "", 2)]
    assert [a for _, _, a in items_of(result(app))] == [expected] * 3
    app.button(key="del_global_ans_0").click().run()
    expected = [("1", "Neutral", 0), ("2", "", 1)]
    assert [a for _, _, a in items_of(result(app))] == [expected] * 3

    # switching back to local answer sets starts every item with two empty answer options
    app.toggle(key="use_global_answer_set").set_value(False).run()
    assert not app.exception
    assert [a for _, _, a in items_of(result(app))] == [[("1", "", 0), ("2", "", 1)]] * 3
    assert not [t for t in app.text_input if t.key.startswith("global_ans_")]


# ===========================================================================
#                      App: importing a configuration
# ===========================================================================


def make_importable_config(config_dict):
    """An experiment configuration in the shape that the configurator exports and imports."""
    options = {
        "1": {"text": "1. never", "weight": 1, "ignored_for_scale": False},
        "2": {"text": "2. sometimes", "weight": 2, "ignored_for_scale": False},
        "3": {"text": "3. often", "weight": 3, "ignored_for_scale": False},
    }
    profile = {"title": "Mr.", "name": "Tsosie", "ethnicity": "Native", "id": 1}
    return {
        "name": "Imported experiment",
        "description": "Imported description",
        "parameters": {"seeds": ["7"], "lazy_load_models": True},
        "prompt_template": copy.deepcopy(config_dict["prompt_template"]),
        "models": {"some-model": {"type": "ollama", "name": "llama3"}},
        "demographic_profiles": {
            "1-Mr. Tsosie": {"attributes": profile, "template": "{title} {name}"},
            "2-Ms. Nez": {
                "attributes": {**profile, "title": "Ms.", "name": "Nez", "id": 2},
                "template": "{title} {name}",
            },
        },
        "questionnaire": {
            "name": "Imported questionnaire",
            "general_instruction": "Rate yourself.",
            "attributes": {"dimension": {"1": "Promotion", "2": "Prevention"}},
            "instruction_items": [
                {
                    "question": "I often think about my goals.",
                    "reversed": False,
                    "answer_options": copy.deepcopy(options),
                    "attributes": {"dimension": "1"},
                },
                {
                    "question": "I rarely worry about failing.",
                    "reversed": True,
                    "answer_options": copy.deepcopy(options),
                    "attributes": {"dimension": "2"},
                },
            ],
        },
    }


def import_config(at, config):
    upload(at, "uploaded_config", "config.json", json.dumps(config).encode(), "application/json")


@pytest.fixture(scope="module")
def imported_app(config_dict):
    """An app into which ``make_importable_config`` was imported (tests must only read from it)."""
    if not hasattr(AppTest, "file_uploader"):
        pytest.skip("AppTest only supports st.file_uploader in newer Streamlit versions")
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    source = make_importable_config(config_dict)
    import_config(at, source)
    return at, source


@needs_upload_support
def test_imported_configuration_fills_the_widgets_and_passes_unsupported_sections(imported_app):
    app, source = imported_app

    assert successes(app) == ["Configuration loaded"]
    assert not app.error
    assert app.text_input(key="exp_name").value == "Imported experiment"
    assert app.text_area(key="exp_descr").value == "Imported description"
    assert app.text_input(key="quest_name").value == "Imported questionnaire"
    assert app.text_area(key="quest_instr").value == "Rate yourself."
    assert [t.value for t in app.text_input if t.key.startswith("question_")] == [
        "I often think about my goals.",
        "I rarely worry about failing.",
    ]

    config = result(app)
    for section in ("name", "description", "parameters", "prompt_template", "models"):
        assert config[section] == source[section]
    assert config["questionnaire"]["attributes"] == source["questionnaire"]["attributes"]
    assert profiles_of(config) == [
        ("1-Mr. Tsosie", "Mr.", "Tsosie", "Native"),
        ("2-Ms. Nez", "Ms.", "Nez", "Native"),
    ]
    items = config["questionnaire"]["instruction_items"]
    assert [item["question"] for item in items] == [
        "I often think about my goals.",
        "I rarely worry about failing.",
    ]
    for item in items:
        assert [(k, o["text"]) for k, o in item["answer_options"].items()] == [
            ("1", "1. never"),
            ("2", "2. sometimes"),
            ("3", "3. often"),
        ]
    assert [item["attributes"] for item in items] == [{"dimension": "1"}, {"dimension": "2"}]
    assert_loads_in_rupsycho(config)


@needs_upload_support
def test_imported_configuration_keeps_the_reversed_flags(imported_app):
    app, source = imported_app
    expected = [item["reversed"] for item in source["questionnaire"]["instruction_items"]]

    assert expected == [False, True]
    assert [r for _, r, _ in items_of(result(app))] == expected
    # the toggles of the items show the imported flags, too
    assert [t.value for t in app.toggle if t.key.startswith("reversed_")] == expected


@needs_upload_support
def test_imported_configuration_keeps_the_answer_weights(imported_app):
    app, _ = imported_app

    assert weights_of(result(app)) == [[1, 2, 3], [1, 2, 3]]  # not renumbered to 0, 1, 2


@needs_upload_support
def test_imported_configuration_exports_no_internal_widget_state(imported_app):
    app, _ = imported_app

    assert "widget_key" not in json.dumps(result(app))


@needs_upload_support
def test_import_preserves_custom_weights_and_ignored_options(app, config_dict):
    source = make_importable_config(config_dict)
    scale = {
        "1": {"text": "strongly agree", "weight": 5, "ignored_for_scale": False},
        "2": {"text": "agree", "weight": 3, "ignored_for_scale": False},
        "3": {"text": "no answer", "weight": 0, "ignored_for_scale": True},
    }
    source["questionnaire"]["instruction_items"][0]["answer_options"] = copy.deepcopy(scale)

    import_config(app, source)

    assert successes(app) == ["Configuration loaded"]
    exported = result(app)["questionnaire"]["instruction_items"][0]["answer_options"]
    assert exported == scale


@needs_upload_support
def test_import_converts_weights_given_as_text_and_rejects_other_weights(app, config_dict):
    source = make_importable_config(config_dict)
    first_option = source["questionnaire"]["instruction_items"][0]["answer_options"]["1"]

    first_option["weight"] = "7"
    import_config(app, source)
    assert successes(app) == ["Configuration loaded"]
    assert weights_of(result(app))[0] == [7, 2, 3]  # a number again, as the model requires
    assert_loads_in_rupsycho(result(app))

    first_option["weight"] = "heavy"
    import_config(app, source)
    assert errors(app) == ["Invalid configuration"]


@needs_upload_support
def test_answer_options_added_to_an_imported_scale_continue_it(app, config_dict):
    import_config(app, make_importable_config(config_dict))
    assert weights_of(result(app)) == [[1, 2, 3], [1, 2, 3]]

    add_answer_button(app, 0).click().run()
    add_answer_button(app, 0).click().run()

    assert weights_of(result(app)) == [[1, 2, 3, 4, 5], [1, 2, 3]]
    assert items_of(result(app))[0][2][3:] == [("4", "", 4), ("5", "", 5)]
    assert not app.exception


@needs_upload_support
def test_deleting_answer_options_keeps_an_imported_scale_consecutive(app, config_dict):
    import_config(app, make_importable_config(config_dict))

    delete_answer_button(app, 1, 0).click().run()  # the first option of the second item

    assert weights_of(result(app)) == [[1, 2, 3], [1, 2]]
    assert [text for _, text, _ in items_of(result(app))[1][2]] == ["2. sometimes", "3. often"]

    delete_answer_button(app, 1, 1).click().run()  # now its last one

    assert weights_of(result(app)) == [[1, 2, 3], [1]]
    assert not app.exception


@needs_upload_support
def test_deleting_answer_options_leaves_a_custom_scale_alone(app, config_dict):
    source = make_importable_config(config_dict)
    options = source["questionnaire"]["instruction_items"][0]["answer_options"]
    for option, weight in zip(options.values(), (1, 3, 5)):
        option["weight"] = weight
    import_config(app, source)
    assert weights_of(result(app))[0] == [1, 3, 5]

    add_answer_button(app, 0).click().run()
    assert weights_of(result(app))[0] == [1, 3, 5, 6]  # continues after the highest weight

    delete_answer_button(app, 0, 1).click().run()
    assert weights_of(result(app))[0] == [1, 5, 6]  # no renumbering: the scale was not 1, 2, 3, ...


@needs_upload_support
def test_importing_a_second_configuration_replaces_the_first_one_completely(app, config_dict):
    import_config(app, make_importable_config(config_dict))
    second = make_importable_config(config_dict)
    second["name"] = "Second experiment"
    del second["demographic_profiles"]["2-Ms. Nez"]
    del second["questionnaire"]["instruction_items"][0]
    only_item = second["questionnaire"]["instruction_items"][0]
    assert only_item["reversed"] is True

    import_config(app, second)

    config = result(app)
    assert config["name"] == "Second experiment"
    assert profiles_of(config) == [("1-Mr. Tsosie", "Mr.", "Tsosie", "Native")]
    assert items_of(config) == [
        (
            "I rarely worry about failing.",
            True,
            [("1", "1. never", 1), ("2", "2. sometimes", 2), ("3", "3. often", 3)],
        )
    ]
    # nothing of the first file is left in the widgets
    assert [t.value for t in app.text_input if t.key.startswith("question_")] == [
        "I rarely worry about failing."
    ]
    assert [t.value for t in app.toggle if t.key.startswith("reversed_")] == [True]

    # a reverse-scored item that was imported keeps its flag when it is duplicated
    duplicate = next(b for b in app.button if (b.key or "").startswith("duplicate_item_button_"))
    duplicate.click().run()
    assert [reversed_ for _, reversed_, _ in items_of(result(app))] == [True, True]
    assert weights_of(result(app)) == [[1, 2, 3], [1, 2, 3]]


@needs_upload_support
def test_import_accepts_items_without_reversed_flag_and_attributes(app, config_dict):
    source = make_importable_config(config_dict)
    for item in source["questionnaire"]["instruction_items"]:
        del item["reversed"], item["attributes"]

    import_config(app, source)

    assert successes(app) == ["Configuration loaded"]
    items = result(app)["questionnaire"]["instruction_items"]
    assert [item["reversed"] for item in items] == [False, False]
    assert [item["attributes"] for item in items] == [{}, {}]


@needs_upload_support
def test_import_applies_the_default_answer_options_to_items_without_their_own(app, config_dict):
    source = make_importable_config(config_dict)
    first, second = source["questionnaire"]["instruction_items"]
    defaults = first.pop("answer_options")
    own = {"1": {"text": "yes", "weight": 1, "ignored_for_scale": False}}
    second["answer_options"] = copy.deepcopy(own)
    source["questionnaire"]["default_answer_options"] = copy.deepcopy(defaults)

    import_config(app, source)

    assert successes(app) == ["Configuration loaded"]
    exported = result(app)["questionnaire"]["instruction_items"]
    assert exported[0]["answer_options"] == defaults
    assert exported[1]["answer_options"] == own
    assert_loads_in_rupsycho(result(app))


@needs_upload_support
def test_the_bundled_bfi_example_can_be_edited_once_its_personas_have_an_ethnicity(app):
    # the BFI example gives its answer options once (default_answer_options) and has reverse-scored
    # items; its personas need the 'ethnicity' that the configurator requires
    source = rup.load_example_config("bfi")
    for persona in source["demographic_profiles"].values():
        persona["attributes"]["ethnicity"] = "Swiss"
    source["models"] = {}
    items = source["questionnaire"]["instruction_items"]
    defaults = source["questionnaire"]["default_answer_options"]
    assert all("answer_options" not in item for item in items)
    assert any(item["reversed"] for item in items)

    import_config(app, source)

    assert successes(app) == ["Configuration loaded"]
    exported = result(app)
    exported_items = exported["questionnaire"]["instruction_items"]
    assert [item["question"] for item in exported_items] == [item["question"] for item in items]
    assert [item["reversed"] for item in exported_items] == [item["reversed"] for item in items]
    assert [item["attributes"] for item in exported_items] == [item["attributes"] for item in items]
    for item in exported_items:
        assert item["answer_options"] == defaults  # keys, texts and weights of the scale
    assert_loads_in_rupsycho(exported)


@needs_upload_support
def test_an_example_configuration_keeps_its_questionnaire_through_the_configurator(app):
    path = Path(__file__).parents[1] / "examples" / "data" / "bfi_small_and_mid.json"
    if not path.is_file():
        pytest.skip("the example configurations are not part of this installation")
    source = json.loads(path.read_text(encoding="utf-8"))

    upload(app, "uploaded_config", path.name, path.read_bytes(), "application/json")

    assert successes(app) == ["Configuration loaded"]
    exported = result(app)
    for section in ("name", "description", "parameters", "prompt_template", "models"):
        assert exported[section] == source[section]
    assert len(exported["demographic_profiles"]) == len(source["demographic_profiles"])
    items = exported["questionnaire"]["instruction_items"]
    source_items = source["questionnaire"]["instruction_items"]
    assert len(items) == len(source_items) and items
    assert [item["question"] for item in items] == [item["question"] for item in source_items]
    assert [item["reversed"] for item in items] == [item["reversed"] for item in source_items]
    assert any(item["reversed"] for item in items)
    defaults = source["questionnaire"].get("default_answer_options")
    for item, source_item in zip(items, source_items):
        own_or_default = source_item.get("answer_options") or defaults
        expected = [
            (option["text"], option["weight"], option["ignored_for_scale"])
            for option in own_or_default.values()
        ]
        actual = [
            (option["text"], option["weight"], option["ignored_for_scale"])
            for option in item["answer_options"].values()
        ]
        assert actual == expected
    assert_loads_in_rupsycho(exported)


@needs_upload_support
def test_invalid_configurations_are_rejected_and_reset_the_app(app):
    initial = result(app)
    invalid_files = (
        b"{not json",
        b"[]",
        b'{"name": "only a name"}',
        b"\x89PNG\r\n\x1a\n",  # a binary file instead of text
    )
    for content in invalid_files:
        app.text_input(key="exp_name").set_value("will be lost").run()

        upload(app, "uploaded_config", "bad.json", content, "application/json")

        assert errors(app) == ["Invalid configuration"], content
        config = result(app)
        assert config["name"] == ""
        assert len(config["demographic_profiles"]) == 1
        assert len(config["questionnaire"]["instruction_items"]) == 1
        assert config == initial, content  # exactly the empty initial state
        app.run()  # the app keeps working afterwards
        assert not app.exception
        assert not app.error


@needs_upload_support
def test_a_rejected_import_leaves_nothing_of_the_file_behind(app, config_dict):
    initial = result(app)
    source = make_importable_config(config_dict)
    last_item = source["questionnaire"]["instruction_items"][-1]
    del last_item["answer_options"]["3"]["ignored_for_scale"]  # the file fails at its very end

    import_config(app, source)

    assert errors(app) == ["Invalid configuration"]
    assert not successes(app)
    # name, parameters, models, personas and the first item of the file had already been read
    assert result(app) == initial
    assert app.text_input(key="exp_name").value == ""
    assert not [t for t in app.text_input if t.key.startswith("question_") and t.value]


def without(config, path):
    """A copy of the configuration from which the entry at the given path of keys is removed."""
    config = copy.deepcopy(config)
    parent = config
    for step in path[:-1]:
        parent = parent[step]
    del parent[path[-1]]
    return config


@needs_upload_support
def test_configurations_missing_a_required_key_are_rejected(app, config_dict):
    # the keys that the tutorial lists as required, one example from each level of the file
    required = {
        "top level": ("description",),
        "questionnaire": ("questionnaire", "instruction_items"),
        "item": ("questionnaire", "instruction_items", 0, "question"),
        "answer option": ("questionnaire", "instruction_items", 0, "answer_options", "1", "text"),
        "persona": ("demographic_profiles", "1-Mr. Tsosie", "attributes", "ethnicity"),
    }
    initial = result(app)
    for level, path in required.items():
        import_config(app, without(make_importable_config(config_dict), path))

        assert errors(app) == ["Invalid configuration"], level
        assert result(app) == initial, level
        assert not app.exception, level


@needs_upload_support
@pytest.mark.parametrize(
    ("tab", "corrupt"),
    [
        ("Experiment Info", lambda c: c.update(name={"not": "a string"})),
        (
            "Demographic Profiles",
            lambda c: c["demographic_profiles"]["1-Mr. Tsosie"]["attributes"].update(name=5),
        ),
        ("Questionnaire Info", lambda c: c["questionnaire"].update(general_instruction=["x"])),
        (
            "Questionnaire Items",
            lambda c: c["questionnaire"]["instruction_items"][0].update(question={"a": 1}),
        ),
        (
            "Questionnaire Items",
            lambda c: c["questionnaire"]["instruction_items"][1]["answer_options"]["2"].update(
                text=3
            ),
        ),
    ],
    ids=["experiment-name", "persona-name", "questionnaire-instruction", "question", "answer"],
)
def test_unexpected_value_types_in_an_imported_configuration_reset_the_app(
    app, config_dict, tab, corrupt
):
    initial = result(app)
    source = make_importable_config(config_dict)
    corrupt(source)

    import_config(app, source)

    assert errors(app) == [f"Unexpected value in '{tab}', app was reset"]
    assert result(app) == initial
    app.run()
    assert not app.error


@needs_upload_support
def test_importing_switches_off_the_global_answer_set_and_removing_the_file_changes_nothing(
    app, config_dict
):
    app.toggle(key="use_global_answer_set").set_value(True).run()

    import_config(app, make_importable_config(config_dict))

    assert app.toggle(key="use_global_answer_set").value is False
    assert len(items_of(result(app))) == 2

    app.file_uploader(key="uploaded_config").set_value(None).run()  # the user removes the file
    assert not app.exception
    assert result(app)["name"] == "Imported experiment"


# ===========================================================================
#                       App: importing profiles from CSV
# ===========================================================================


def import_csv(at, text):
    upload(at, "uploaded_csv", "profiles.csv", text.encode(), "text/csv")


@needs_upload_support
def test_profiles_can_be_imported_from_csv(app):
    app.text_input(key="name_0").set_value("replaced").run()

    import_csv(app, "title,name,ethnicity\nMr.,Tsosie,Native\nMs.,Nez,Native\n")

    assert not app.error
    assert profiles_of(result(app)) == [
        ("1-Mr. Tsosie", "Mr.", "Tsosie", "Native"),
        ("2-Ms. Nez", "Ms.", "Nez", "Native"),
    ]
    assert [t.value for t in app.text_input if t.key.startswith("name_")] == ["Tsosie", "Nez"]

    # column order and additional columns do not matter; a new file replaces the profiles
    import_csv(app, "ethnicity,comment,name,title\nIrish,x,Murphy,Dr.\n")
    assert profiles_of(result(app)) == [("1-Dr. Murphy", "Dr.", "Murphy", "Irish")]


@needs_upload_support
@pytest.mark.xfail(
    strict=True,
    reason="import_profiles rejects a file with an empty cell in any column, also in columns "
    "that are not used: a spreadsheet export with a partly empty 'comment' column or with a "
    "trailing delimiter is reported as 'Invalid file structure' (only title, name and ethnicity "
    "need values)",
)
def test_empty_cells_in_columns_that_are_not_used_do_not_matter(app):
    files = (
        "title,name,ethnicity,comment\nMr.,Tsosie,Native,\nMs.,Nez,Native,fine\n",
        "title,name,ethnicity,\nMr.,Tsosie,Native,\nMs.,Nez,Native,\n",  # a trailing delimiter
    )
    for text in files:
        import_csv(app, text)

        assert not app.error, text
        assert profiles_of(result(app)) == [
            ("1-Mr. Tsosie", "Mr.", "Tsosie", "Native"),
            ("2-Ms. Nez", "Ms.", "Nez", "Native"),
        ], text


@needs_upload_support
def test_invalid_csv_files_are_rejected_and_reset_the_profiles(app):
    invalid_files = (
        "title,name\nMr.,Tsosie\n",  # missing column
        "title,name,ethnicity\nMr.,Tsosie,\n",  # empty value
        "just some text\n",  # no table
    )
    for text in invalid_files:
        import_csv(app, "title,name,ethnicity\nMr.,Old,Native\n")
        assert len(result(app)["demographic_profiles"]) == 1

        import_csv(app, text)

        assert errors(app) == ["Invalid file structure"], text
        assert profiles_of(result(app)) == [("1- ", "", "", "")]


@needs_upload_support
@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ("Mr.,Smith,None", ("1-Mr. Smith", "Mr.", "Smith", "None")),
        ("Mr.,007,Native", ("1-Mr. 007", "Mr.", "007", "Native")),
        ("NA,N/A,null", ("1-NA N/A", "NA", "N/A", "null")),
        ("nan,NaN,<NA>", ("1-nan NaN", "nan", "NaN", "<NA>")),
        ("42,3.0,1e3", ("1-42 3.0", "42", "3.0", "1e3")),
        ('Mr.,"Smith, Jr.",Native', ("1-Mr. Smith, Jr.", "Mr.", "Smith, Jr.", "Native")),
    ],
    ids=[
        "na-like-string",
        "numeric-looking-name",
        "all-na-spellings",
        "nan-spellings",
        "numbers-stay-text",
        "comma-in-quotes",
    ],
)
def test_csv_values_are_imported_as_text(app, row, expected):
    import_csv(app, f"title,name,ethnicity\n{row}\n")

    assert not app.error
    assert profiles_of(result(app)) == [expected]
    assert [t.value for t in app.text_input if t.key.startswith("name_")] == [expected[2]]


@needs_upload_support
def test_csv_files_may_contain_any_unicode_and_be_saved_by_a_spreadsheet(app):
    # UTF-8 with a byte order mark and Windows line endings, as Excel writes CSV files
    text = "title,name,ethnicity\r\nDr.,Müller,Österreicher\r\nProf.,山田,日本\r\n"
    upload(app, "uploaded_csv", "profiles.csv", b"\xef\xbb\xbf" + text.encode("utf-8"), "text/csv")

    assert not app.error
    assert profiles_of(result(app)) == [
        ("1-Dr. Müller", "Dr.", "Müller", "Österreicher"),
        ("2-Prof. 山田", "Prof.", "山田", "日本"),
    ]


@needs_upload_support
def test_csv_files_that_are_not_utf8_are_rejected_with_a_message(app):
    latin1 = "title,name,ethnicity\nDr.,Müller,Irish\n".encode("latin-1")

    upload(app, "uploaded_csv", "profiles.csv", latin1, "text/csv")

    assert len(errors(app)) == 1  # a message instead of a crash
    assert profiles_of(result(app)) == [("1- ", "", "", "")]


# ===========================================================================
#                       App: PDF upload and page selection
# ===========================================================================


@needs_upload_support
def test_uploaded_pdf_fills_the_questionnaire_text_and_offers_a_page_slider(app, pdf_bytes):
    upload(app, "uploaded_pdf", "questionnaire.pdf", pdf_bytes, "application/pdf")

    assert not app.error
    assert squash(app.text_area(key="quest_text").value) == (
        "Title page 1. I am talkative andor loud 2. I am reserved Answer options: agree, disagree"
    )
    assert app.slider(key="page_range").value == (1, 3)

    app.slider(key="page_range").set_range(2, 3).run()
    assert squash(app.text_area(key="quest_text").value).startswith("1. I am talkative")
    assert "Title page" not in app.text_area(key="quest_text").value

    app.slider(key="page_range").set_range(3, 3).run()
    assert squash(app.text_area(key="quest_text").value) == "Answer options: agree, disagree"


@needs_upload_support
def test_removing_the_pdf_keeps_the_text_and_single_page_pdfs_have_no_slider(app, pdf_bytes):
    upload(app, "uploaded_pdf", "questionnaire.pdf", pdf_bytes, "application/pdf")
    assert len(app.slider) == 1
    text = app.text_area(key="quest_text").value

    app.file_uploader(key="uploaded_pdf").set_value(None).run()
    assert not app.exception
    assert app.text_area(key="quest_text").value == text
    assert len(app.slider) == 0

    upload(app, "uploaded_pdf", "one.pdf", make_pdf(["Only page"]), "application/pdf")
    assert squash(app.text_area(key="quest_text").value) == "Only page"
    assert len(app.slider) == 0


@needs_upload_support
def test_corrupt_pdf_is_reported_as_an_error_message(app):
    broken_files = (
        b"this is not a pdf",
        b"",
        b"%PDF-1.4\nthe header is right, nothing else is",
    )
    for content in broken_files:
        app.file_uploader(key="uploaded_pdf").upload("broken.pdf", content, "application/pdf").run()

        assert not app.exception, content
        assert errors(app) == ["Could not read the PDF"], content
        assert len(app.slider) == 0, content


@needs_upload_support
def test_a_pdf_can_be_loaded_after_a_corrupt_one(app, pdf_bytes):
    app.file_uploader(key="uploaded_pdf").upload("broken.pdf", b"nope", "application/pdf").run()
    assert errors(app) == ["Could not read the PDF"]

    upload(app, "uploaded_pdf", "questionnaire.pdf", pdf_bytes, "application/pdf")

    assert not app.error
    assert squash(app.text_area(key="quest_text").value).startswith("Title page 1. I am talkative")
    assert app.slider(key="page_range").value == (1, 3)

    # and the other way round: a corrupt file does not leave pages of the PDF behind
    app.file_uploader(key="uploaded_pdf").upload("broken.pdf", b"nope", "application/pdf").run()
    assert errors(app) == ["Could not read the PDF"]
    assert len(app.slider) == 0


@needs_upload_support
@pytest.mark.xfail(
    strict=True,
    reason="set_or_delete_pdf falls through to concat_pages after a failed read and overwrites "
    "quest_text with '': the text that the user has already put into the field is lost "
    "(removing a PDF leaves the text untouched, a PDF that cannot be read should, too)",
)
def test_a_corrupt_pdf_leaves_the_questionnaire_text_untouched(app):
    app.text_area(key="quest_text").set_value("1. Is talkative\n2. Is reserved").run()

    app.file_uploader(key="uploaded_pdf").upload("broken.pdf", b"nope", "application/pdf").run()

    assert errors(app) == ["Could not read the PDF"]
    assert app.text_area(key="quest_text").value == "1. Is talkative\n2. Is reserved"


# ===========================================================================
#                 App: API key validation and LLM-assisted import
# ===========================================================================


@pytest.mark.parametrize(
    ("outcome", "message", "valid"),
    [
        ("valid", "Valid API key", True),
        ("rejected", "Invalid API key", False),
        ("offline", "Something went wrong", False),
    ],
)
def test_api_key_validation(fake_openai, app, outcome, message, valid):
    fake_openai.outcome = outcome

    app.text_input(key="api_key").set_value("sk-test").run()

    assert not app.exception
    assert (successes(app) if valid else errors(app)) == [message]
    assert app.session_state["valid_api_key"] is valid
    assert fake_openai.api_keys == ["sk-test"]
    if valid:
        assert app.session_state["api_key"] == "sk-test"
        assert os.environ["OPENAI_API_KEY"] == "sk-test"
    else:  # a rejected key is wiped from the input field and not exported
        assert app.session_state["api_key"] == ""
        assert "OPENAI_API_KEY" not in os.environ


def test_api_key_input_masks_the_key(app):
    text_input_proto = pytest.importorskip("streamlit.proto.TextInput_pb2").TextInput

    assert app.text_input(key="api_key").proto.type == text_input_proto.PASSWORD


def test_api_key_is_not_part_of_the_configuration_or_its_download(fake_openai, app, downloads):
    app.text_input(key="api_key").set_value("sk-secret-test-key").run()
    assert app.session_state["valid_api_key"]
    app.text_input(key="exp_name").set_value("Experiment").run()

    config = result(app)

    assert config["name"] == "Experiment"
    assert "sk-secret-test-key" not in json.dumps(config)
    assert "sk-secret-test-key" not in downloads.calls[-1]["data"]
    assert "api_key" not in downloads.calls[-1]["data"].lower()


def test_run_button_is_enabled_only_with_a_valid_key_and_a_questionnaire_text(fake_openai, app):
    assert app.button[0].label == "**Run**"
    assert app.button[0].disabled

    app.text_area(key="quest_text").set_value("1. Is talkative").run()
    assert app.button[0].disabled  # text but no key

    app.text_input(key="api_key").set_value("sk-test").run()
    assert not app.button[0].disabled

    app.text_area(key="quest_text").set_value("").run()
    assert app.button[0].disabled  # key but no text


def start_model_run(app, quest_text="1. Is talkative\n2. Is reserved\n3. Is helpful"):
    app.text_input(key="api_key").set_value("sk-test").run()
    assert app.session_state["valid_api_key"]
    app.text_area(key="quest_text").set_value(quest_text).run()
    app.button[0].click().run()


def test_model_run_fills_the_questionnaire_from_the_model_output(
    fake_openai, fake_llm, app, downloads
):
    app.toggle(key="use_global_answer_set").set_value(True).run()  # will be switched off
    app.text_input(key="question_0").set_value("to be replaced").run()
    app.text_input(key="exp_name").set_value("My experiment").run()  # not part of a run
    app.text_input(key="name_0").set_value("Smith").run()
    questionnaire_text = "1. Is talkative\n2. Is reserved\n3. Is helpful"

    start_model_run(app, questionnaire_text)

    assert not app.exception
    assert successes(app) == ["Model run was successful"]
    assert fake_llm.kwargs["model"] == "gpt-4o-mini"
    assert fake_llm.kwargs["api_key"] == "sk-test"
    (messages,) = fake_llm.prompts
    assert messages[-1].content.startswith("1. Is talkative")  # the questionnaire text is the input

    config = result(app)
    assert config["questionnaire"]["name"] == "BFI"
    assert config["questionnaire"]["general_instruction"] == "Rate each statement."
    # the single answer set is used for every question, as a scale that starts at 0
    scale = [("1", "1. Disagree", 0), ("2", "2. Neutral", 1), ("3", "3. Agree", 2)]
    assert items_of(config) == [
        ("Is talkative", False, scale),
        ("Is reserved", False, scale),
        ("Is helpful", False, scale),
    ]
    assert app.text_input(key="quest_name").value == "BFI"
    assert [t.value for t in app.text_input if t.key.startswith("question_")] == [
        "Is talkative",
        "Is reserved",
        "Is helpful",
    ]
    assert app.toggle(key="use_global_answer_set").value is False
    assert app.text_area(key="quest_text").value == questionnaire_text  # the text stays
    # a run replaces the questionnaire only
    assert config["name"] == "My experiment"
    assert profiles_of(config) == [("1- Smith", "", "Smith", "")]
    # the API key that the run used is not exported
    assert "sk-test" not in json.dumps(config)
    assert "sk-test" not in downloads.calls[-1]["data"]
    assert_loads_in_rupsycho(config)


def test_model_run_with_one_answer_set_per_question(fake_openai, fake_llm, app):
    fake_llm.response = llm_answer(
        questions=["Q1", "Q2"], answers=[["yes", "no"], ["true", "false", "unsure"]]
    )

    start_model_run(app)

    assert [[t for _, t, _ in a] for _, _, a in items_of(result(app))] == [
        ["yes", "no"],
        ["true", "false", "unsure"],
    ]


def test_model_run_sends_the_text_without_quotes_slashes_and_backslashes(
    fake_openai, fake_llm, app
):
    text = '1. I am "talkative" and/or loud\n2. I am \\reserved'

    start_model_run(app, text)

    (messages,) = fake_llm.prompts
    sent = messages[-1].content
    assert not set('"/\\') & set(sent)  # the model does not have to escape anything
    assert all(word in sent for word in ("talkative", "loud", "reserved"))
    assert app.text_area(key="quest_text").value == text  # the displayed text is not changed


def test_model_run_reports_a_failing_model_call_and_can_be_retried(fake_openai, fake_llm, app):
    fake_llm.error = RuntimeError("rate limit exceeded")

    start_model_run(app)

    assert not app.exception
    assert errors(app) == ["Model error, please try again"]
    assert not successes(app)
    assert len(result(app)["questionnaire"]["instruction_items"]) == 1  # the empty questionnaire

    fake_llm.error = None
    app.button[0].click().run()  # the key and the text are still there

    assert successes(app) == ["Model run was successful"]
    assert [q for q, _, _ in items_of(result(app))] == ["Is talkative", "Is reserved", "Is helpful"]


def test_model_run_validates_the_shape_of_the_model_output(fake_openai, fake_llm, app):
    wrongly_shaped = (
        ["T", "I", ["q1", "q2"], ["Yes", "No"]],  # the answer sets must be lists of lists
        ["T", "I", ["q1", "q2"], "Yes/No"],
        ["T", "I", ["q1", "q2"], {"set": ["Yes", "No"]}],
        ["T", "I", "q1", [["Yes", "No"]]],  # the questions must be a list of strings
        ["T", "I", ["q1", 2], [["Yes", "No"]]],
        ["T", "I", ["q1", "q2"], [["Yes", 1]]],  # ... and the answer options strings
        [1, "I", ["q1"], [["Yes"]]],  # the title and the instruction must be strings
        ["T", None, ["q1"], [["Yes"]]],
        ["T", "I", ["q1"], [["Yes"]], "and one more"],  # exactly four parts
        {"title": "T"},
    )
    for number, output in enumerate(wrongly_shaped):
        fake_llm.response = f"```json\n{json.dumps(output)}\n```"

        if number == 0:
            start_model_run(app)
        else:
            app.button[0].click().run()

        assert not app.exception, output
        assert errors(app) == ["Model error, please try again"], output
        assert not successes(app), output
        config = result(app)  # nothing of the output is used: the app is reset
        assert config["questionnaire"]["name"] == "", output
        assert items_of(config) == [("", False, [("1", "", 0), ("2", "", 1)])], output


def test_model_run_can_be_repeated_and_replaces_the_previous_result(fake_openai, fake_llm, app):
    start_model_run(app)
    assert [q for q, _, _ in items_of(result(app))] == ["Is talkative", "Is reserved", "Is helpful"]

    fake_llm.response = llm_answer(
        title="Second", questions=["Only one question"], answers=[["yes", "no"]]
    )
    app.button[0].click().run()

    assert not app.exception
    assert successes(app) == ["Model run was successful"]
    config = result(app)
    assert config["questionnaire"]["name"] == "Second"
    assert items_of(config) == [("Only one question", False, [("1", "yes", 0), ("2", "no", 1)])]
    assert [t.value for t in app.text_input if t.key.startswith("question_")] == [
        "Only one question"
    ]


def test_model_run_with_unusable_output_reports_an_error_and_resets(fake_openai, fake_llm, app):
    unusable_responses = (
        "I cannot do that.",  # no JSON fence
        '```json\n["only", "two"]\n```',  # not the four parts that were asked for
        "```json\nnot json\n```",
    )
    for number, response in enumerate(unusable_responses):
        fake_llm.response = response

        if number == 0:
            start_model_run(app)
        else:
            app.button[0].click().run()  # the key and the text are still there

        assert not app.exception
        assert errors(app) == ["Model error, please try again"], response
        config = result(app)
        assert len(config["questionnaire"]["instruction_items"]) == 1  # the empty questionnaire
        assert config["questionnaire"]["name"] == ""
