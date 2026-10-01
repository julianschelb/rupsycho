"""Tests for the Streamlit configurator (``src/rupsycho_configurator``).

The configurator is an optional extra (``pip install "rupsycho[configurator]"``), so this module
is skipped when Streamlit, pypdf or openai are not installed.  Everything runs offline: the app
is driven headlessly with ``streamlit.testing.v1.AppTest``, PDFs are generated in memory, and the
OpenAI client / chat model used by the app are replaced by fakes.

Tests marked ``xfail(strict=True)`` describe the *correct* behaviour of a known bug in the
configurator.  They keep the suite green today and start failing (XPASS) once the bug is fixed;
remove the marker then.
"""

import copy
import io
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("pypdf")
pytest.importorskip("openai")

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
    """Replace ``ChatOpenAI`` by a fake chat model that answers with ``fake_llm.response``.

    Request this fixture *before* ``app``: the app imports ``ChatOpenAI`` on every run.
    """
    state = SimpleNamespace(response=llm_answer(), kwargs=None, prompts=[])

    class FakeChatOpenAI(FakeListChatModel):
        def __init__(self, **kwargs):
            state.kwargs = kwargs
            super().__init__(responses=[state.response])

        def _generate(self, messages, *args, **kwargs):
            state.prompts.append(messages)
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
    with pytest.raises(ValueError, match="No JSON wrapper"):
        utils.extract_json(text)


@pytest.mark.parametrize(
    "body",
    ['["a", "b"', '["a", "b",]', "['a', 'b']", "", "not json"],
    ids=["missing-bracket", "trailing-comma", "single-quotes", "empty-fence", "garbage"],
)
def test_extract_json_rejects_malformed_json(body):
    with pytest.raises(ValueError, match="Invalid JSON"):
        utils.extract_json(f"```json\n{body}\n```")


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


def test_extract_quest_text_skips_pages_that_do_not_exist(pdf_bytes, capsys):
    assert utils.extract_quest_text(io.BytesIO(pdf_bytes), [9]) == ""
    assert "invalid page number" in capsys.readouterr().out


@pytest.mark.parametrize("page", [0, -1], ids=["zero", "negative"])
@pytest.mark.xfail(
    strict=True,
    reason="extract_quest_text uses pages[p - 1], so page 0 / negative pages wrap around "
    "to the last pages instead of being reported as invalid",
)
def test_extract_quest_text_rejects_non_positive_page_numbers(pdf_bytes, page):
    assert utils.extract_quest_text(io.BytesIO(pdf_bytes), [page]) == ""


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


@pytest.mark.xfail(
    strict=True,
    reason="from_dict ignores questionnaire.default_answer_options and raises "
    "KeyError('answer_options') for items without their own answer options (documented "
    "limitation of the configurator, but the repo's own BFI test config uses it)",
)
def test_from_dict_supports_default_answer_options(config_dict):
    quest = ConfigQuestionnaire.from_dict(config_dict)
    assert len(quest.questions) == 4
    assert all(len(answer_set) == 5 for answer_set in quest.answers)


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


@pytest.mark.xfail(
    strict=True,
    reason="from_raw_output calls extract_json(title_instr, False) but extract_json only takes "
    "one argument, so it always raises ValueError",
)
def test_from_raw_output_builds_a_questionnaire_from_model_outputs():
    quest = ConfigQuestionnaire.from_raw_output(
        '```json\n["My title", "My instructions"]\n```',
        '```json\n["q1", "q2"]\n```',
        '```json\n[["a", "b"]]\n```',
    )
    assert quest.get_as_list() == ["My title", "My instructions", ["q1", "q2"], [["a", "b"]]]


def test_from_raw_output_rejects_unparsable_title_and_instructions():
    with pytest.raises(ValueError, match="title and instructions"):
        ConfigQuestionnaire.from_raw_output("no json here", "[]", "[]")


# ===========================================================================
#                                  launcher
# ===========================================================================


def test_launcher_starts_streamlit_with_the_app_script(monkeypatch):
    calls = []

    def record(command, *args, **kwargs):
        calls.append(command)
        return 0

    monkeypatch.setattr(subprocess, "run", record)
    monkeypatch.setattr(subprocess, "call", record)  # either may be used to spawn the server

    launcher.main()

    (command,) = calls
    assert command[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert Path(command[4]) == APP_PATH
    assert APP_PATH.is_file()


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


@pytest.mark.xfail(
    strict=True,
    reason="duplicate_item does not pre-set the 'reversed_<key>' toggle of the copy, so a "
    "duplicated reverse-scored item silently becomes a normal item",
)
def test_duplicating_an_item_keeps_its_reversed_flag(app):
    app.toggle(key="reversed_0").set_value(True).run()
    assert [r for _, r, _ in items_of(result(app))] == [True]

    app.button(key="duplicate_item_button_0").click().run()

    assert [r for _, r, _ in items_of(result(app))] == [True, True]


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
@pytest.mark.xfail(
    strict=True,
    reason="imported 'reversed' flags are lost: the 'reversed_<key>' toggles are not pre-set "
    "and overwrite item['reversed'] with False on the first render",
)
def test_imported_configuration_keeps_the_reversed_flags(imported_app):
    app, _ = imported_app

    assert [r for _, r, _ in items_of(result(app))] == [False, True]


@needs_upload_support
@pytest.mark.xfail(
    strict=True,
    reason="imported answer weights are overwritten by the 0-based position of the option "
    "(a 1..3 scale is exported as 0..2)",
)
def test_imported_configuration_keeps_the_answer_weights(imported_app):
    app, _ = imported_app

    for _, _, options in items_of(result(app)):
        assert [weight for _, _, weight in options] == [1, 2, 3]


@needs_upload_support
def test_invalid_configurations_are_rejected_and_reset_the_app(app):
    for content in (b"{not json", b"[]", b'{"name": "only a name"}'):
        app.text_input(key="exp_name").set_value("will be lost").run()

        upload(app, "uploaded_config", "bad.json", content, "application/json")

        assert errors(app) == ["Invalid configuration"], content
        config = result(app)
        assert config["name"] == ""
        assert len(config["demographic_profiles"]) == 1
        assert len(config["questionnaire"]["instruction_items"]) == 1
        app.run()  # the app keeps working afterwards
        assert not app.exception
        assert not app.error


@needs_upload_support
def test_unexpected_value_types_in_an_imported_configuration_reset_the_app(app, config_dict):
    source = make_importable_config(config_dict)
    source["name"] = {"not": "a string"}

    import_config(app, source)

    assert errors(app) == ["Unexpected value in 'Experiment Info', app was reset"]
    assert result(app)["name"] == ""
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
    ],
    ids=["na-like-string", "numeric-looking-name"],
)
@pytest.mark.xfail(
    strict=True,
    reason="pd.read_csv is used without dtype=str / keep_default_na=False: values such as "
    "'None' or 'NA' are treated as missing and numeric-looking values break the widgets",
)
def test_csv_values_are_imported_as_text(app, row, expected):
    import_csv(app, f"title,name,ethnicity\n{row}\n")

    assert profiles_of(result(app)) == [expected]


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
@pytest.mark.xfail(
    strict=True,
    reason="set_or_delete_pdf has no error handling: a corrupt / non-PDF upload raises an "
    "uncaught PdfStreamError instead of showing an error message like the other importers",
)
def test_corrupt_pdf_is_reported_as_an_error_message(app):
    app.file_uploader(key="uploaded_pdf").upload(
        "broken.pdf", b"this is not a pdf", "application/pdf"
    ).run()

    assert not app.exception
    assert app.error


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


def test_model_run_fills_the_questionnaire_from_the_model_output(fake_openai, fake_llm, app):
    app.toggle(key="use_global_answer_set").set_value(True).run()  # will be switched off
    app.text_input(key="question_0").set_value("to be replaced").run()

    start_model_run(app)

    assert not app.exception
    assert successes(app) == ["Model run was successful"]
    assert fake_llm.kwargs["model"] == "gpt-4o-mini"
    assert fake_llm.kwargs["api_key"] == "sk-test"
    (messages,) = fake_llm.prompts
    assert messages[-1].content.startswith("1. Is talkative")  # the questionnaire text is the input

    config = result(app)
    assert config["questionnaire"]["name"] == "BFI"
    assert config["questionnaire"]["general_instruction"] == "Rate each statement."
    expected = ["1. Disagree", "2. Neutral", "3. Agree"]  # the single answer set is used everywhere
    assert [(q, [t for _, t, _ in a]) for q, _, a in items_of(config)] == [
        ("Is talkative", expected),
        ("Is reserved", expected),
        ("Is helpful", expected),
    ]
    assert app.text_input(key="quest_name").value == "BFI"
    assert [t.value for t in app.text_input if t.key.startswith("question_")] == [
        "Is talkative",
        "Is reserved",
        "Is helpful",
    ]
    assert app.toggle(key="use_global_answer_set").value is False
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


@pytest.mark.xfail(
    strict=True,
    reason="run_model does not validate the shape of the model output: a flat list of answers "
    "(or a string) is split into single characters and the run is reported as successful",
)
def test_model_run_validates_the_shape_of_the_model_output(fake_openai, fake_llm, app):
    wrongly_shaped = (
        ["T", "I", ["q1", "q2"], ["Yes", "No"]],  # the answer sets must be lists of lists
        ["T", "I", ["q1", "q2"], "Yes/No"],
        ["T", "I", "q1", [["Yes", "No"]]],  # the questions must be a list of strings
    )
    for number, output in enumerate(wrongly_shaped):
        fake_llm.response = f"```json\n{json.dumps(output)}\n```"

        if number == 0:
            start_model_run(app)
        else:
            app.button[0].click().run()

        assert not app.exception
        assert errors(app) == ["Model error, please try again"], output


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
