# ===========================================================================
#                      Prompt Template Data Model
# ===========================================================================
# This file contains the data model for the prompt templates.

from typing import Any

from pydantic import BaseModel, Field

from rupsycho._compat import load_serialized

# ================================= Prompt Template Configs ================================


class NormalPromptTemplateConfig(BaseModel):
    """
    Configuration for a normal LangChain PromptTemplate.
    This model defines the template string with placeholders.
    """

    type: str = Field("normal", description="The type of the prompt configuration.")

    template: str = Field(
        ...,
        description="The template string with placeholders (e.g., 'Tell me a {adjective} joke about {content}').",
    )

    def load_prompt_template(self):
        """Method to create the actual PromptTemplate object."""
        from langchain_core.prompts import PromptTemplate

        return PromptTemplate.from_template(self.template)


# ---------------------------------------------------------------------------


class ChatMessageConfig(BaseModel):
    """
    Configuration for a single message in a ChatPromptTemplate.
    """

    role: str = Field(..., description="The role of the message (e.g., 'system', 'human', 'ai').")
    content: str = Field(
        ..., description="The content of the message, with placeholders if necessary."
    )


class ChatPromptTemplateConfig(BaseModel):
    """
    Configuration for a LangChain ChatPromptTemplate.
    This model defines the sequence of messages, each associated with a role.
    """

    type: str = Field("chat", description="The type of the prompt configuration.")

    messages: list[ChatMessageConfig] = Field(
        ..., description="A list of messages, each with a role and content."
    )

    def load_prompt_template(self):
        """Method to create the actual ChatPromptTemplate object."""
        from langchain_core.prompts import ChatPromptTemplate

        return ChatPromptTemplate.from_messages([(msg.role, msg.content) for msg in self.messages])


# ---------------------------------------------------------------------------


class LangchainPromptTemplateConfig(BaseModel):
    """
    Configuration for a LangChain ChatPromptTemplate.
    This model defines the sequence of messages, each associated with a role.
    """

    type: str = Field("langchain", description="The type of the prompt configuration.")

    definition: dict[str, Any] = Field(
        ...,
        description="A dictionary containing the serialized LangChain prompt or other runnable configuration.",
    )

    def load_prompt_template(self):
        """Method to deserialize and create the actual LangChain PromptTemplate object."""
        try:
            return load_serialized(self.definition)
        except Exception as e:
            raise ValueError(f"Failed to create the LangChain prompt template: {e}") from e
