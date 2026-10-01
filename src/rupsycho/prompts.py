# prompts.py
"""Default prompt template used when an experiment defines no ``prompt_template``.

The texts are spelled out with explicit escapes on purpose: they contain significant
whitespace (a leading space before ``Question:`` and a trailing space after
``Answer Options:``) that editors, formatters and pre-commit hooks would otherwise strip.
Changing a single character changes what the model sees and therefore the results of every
experiment that relies on the default, so ``tests/test_prompts.py`` pins the exact text.
"""

from __future__ import annotations

from rupsycho.models.prompt import ChatMessageConfig, ChatPromptTemplateConfig

__all__ = [
    "DEFAULT_CHAT_PROMPT_TEMPLATE_CONFIG",
    "system_message_template",
    "user_message_template",
]

system_message_template = (
    "\n"
    'Objective: "{general_instruction}"\n'
    "Answer with respect to the following persona description and question.\n"
)
"""System message of the default chat prompt."""

user_message_template = (
    " \n"
    "Question:\n"
    "{persona_description} was asked the following question. {question}\n"
    "\n"
    "Answer Options: \n"
    "{answer_options}\n"
    "\n"
    "Instructions: Choose from the list of answer options to answer the question. "
    "Answer the question using only the provided answer options. "
    "If none of the options are correct, choose the option that is closest to being correct.\n"
    "\n"
    "Answer:\n"
)
"""User message of the default chat prompt."""

DEFAULT_CHAT_PROMPT_TEMPLATE_CONFIG = ChatPromptTemplateConfig(
    type="chat",
    messages=[
        ChatMessageConfig(role="system", content=system_message_template),
        ChatMessageConfig(role="user", content=user_message_template),
    ],
)
"""The default chat prompt: a system message with the instruction and a user message that asks
one question as one persona."""
