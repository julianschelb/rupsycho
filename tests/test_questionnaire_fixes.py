import pytest

from rupsycho.models.questionnaire import InstructionItem, Questionnaire


@pytest.fixture
def questionnaire(config_dict) -> Questionnaire:
    return Questionnaire(**config_dict["questionnaire"])


def test_print_questionnaire_lists_default_and_item_options(questionnaire, capsys):
    questionnaire.instruction_items.append(
        InstructionItem(
            question="Extra?", answer_options={"1": {"text": "own option", "weight": 1}}
        )
    )
    questionnaire.print_questionnaire()
    output = capsys.readouterr().out
    assert "Default Answer Options" in output
    assert "1. Disagree strongly" in output
    assert "own option" in output


def test_answers_can_be_added_after_a_plain_dict_was_loaded():
    item = InstructionItem(question="Q", answers={"model": {"persona": {"1": "a"}}})
    item.update_answer("other-model", "persona", "1", "b")
    item.update_answer("model", "second-persona", "2", "c")
    assert item.get_all_answers() == {
        "model": {"persona": {"1": "a"}, "second-persona": {"2": "c"}},
        "other-model": {"persona": {"1": "b"}},
    }


def test_none_answers_are_not_stored():
    item = InstructionItem(question="Q")
    item.update_answer("m", "p", "1", None)
    assert not item.get_all_answers()


def test_reading_a_missing_answer_raises_a_key_error():
    item = InstructionItem(question="Q")
    item.update_answer("m", "p", "1", "a")
    with pytest.raises(KeyError):
        item.get_answer("m", "p", "2")
