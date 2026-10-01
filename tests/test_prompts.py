"""The default prompt decides what the model sees when a configuration has no prompt template.

Its text - including whitespace - is part of the reproducibility contract, so it is pinned.
"""

from rupsycho import prompts

GOLDEN_SYSTEM = (
    '\nObjective: "{general_instruction}"\n'
    "Answer with respect to the following persona description and question.\n"
)
GOLDEN_USER = (
    " \nQuestion:\n{persona_description} was asked the following question. {question}\n\n"
    "Answer Options: \n{answer_options}\n\n"
    "Instructions: Choose from the list of answer options to answer the question. Answer the "
    "question using only the provided answer options. If none of the options are correct, "
    "choose the option that is closest to being correct.\n\nAnswer:\n"
)


def test_default_prompt_text_is_unchanged():
    assert prompts.system_message_template == GOLDEN_SYSTEM
    assert prompts.user_message_template == GOLDEN_USER


def test_default_prompt_config_uses_the_pinned_messages():
    messages = prompts.DEFAULT_CHAT_PROMPT_TEMPLATE_CONFIG.messages
    assert [(m.role, m.content) for m in messages] == [
        ("system", GOLDEN_SYSTEM),
        ("user", GOLDEN_USER),
    ]


def test_default_prompt_renders_all_placeholders():
    prompt = prompts.DEFAULT_CHAT_PROMPT_TEMPLATE_CONFIG.load_prompt_template()
    text = prompt.format(
        general_instruction="I",
        persona_description="P",
        question="Q",
        answer_options="A, B",
    )
    assert "{" not in text and "P was asked the following question. Q" in text
