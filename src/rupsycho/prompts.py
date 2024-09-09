# ===========================================================================
#                       Default Prompt Templates
# ===========================================================================
# This module provides a set of default prompt templates for use in experiments.
# The templates are designed for consistent and structured interactions with
# models, including system and user prompts tailored for various tasks.

from rupsycho.models.prompt import ChatMessageConfig, ChatPromptTemplateConfig, NormalPromptTemplateConfig

# ================================= Default Chat Template ================================


system_message_template = """
Objective: "{general_instruction}"
Answer with respect to the following persona description and question.
"""

user_message_template = """ 
Question:
{persona_description} was asked the following question. {question}

Answer Options: 
{answer_options}

Instructions: Choose from the list of answer options to answer the question. Answer the question using only the provided answer options. If none of the options are correct, choose the option that is closest to being correct.

Answer:
"""

# Define Pydantic model for the Default Chat Prompt Template
DEFAULT_CHAT_PROMPT_TEMPLATE_CONFIG = ChatPromptTemplateConfig(
    type="chat",
    messages=[
        ChatMessageConfig(role="system", content=system_message_template),
        ChatMessageConfig(role="user", content=user_message_template)
    ]
)

# # Create the actual ChatPromptTemplate using the Pydantic model
# DEFAULT_CHAT_TEMPLATE = DEFAULT_CHAT_PROMPT_TEMPLATE_CONFIG.load_prompt_template()

# # Define Pydantic model for the Default Prompt Template
# DEFAULT_PROMPT_TEMPLATE_CONFIG = NormalPromptTemplateConfig(
#     type="normal",
#     template=system_message_template + user_message_template
# )

# # Create the actual PromptTemplate using the Pydantic model
# DEFAULT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATE_CONFIG.load_prompt_template()

# ================================= Simple Prompt Template ================================


simple_template = """
{general_instructions}
{persona} was asked the following question. {question}
{answer_options}

Answer:
"""

# Define Pydantic model for the Simple Prompt Template
SIMPLE_PROMPT_TEMPLATE_CONFIG = NormalPromptTemplateConfig(
    type="normal",
    template=simple_template
)

# Create the actual PromptTemplate using the Pydantic model
PROMPT_TEMPLATE_SIMPLE = SIMPLE_PROMPT_TEMPLATE_CONFIG.load_prompt_template()

# Define Pydantic model for the Simple Chat Prompt Template
SIMPLE_CHAT_PROMPT_TEMPLATE_CONFIG = ChatPromptTemplateConfig(
    type="chat",
    messages=[ChatMessageConfig(role="user", content=simple_template)]
)

# Create the actual ChatPromptTemplate using the Pydantic model
CHAT_PROMPT_TEMPLATE_SIMPLE = SIMPLE_CHAT_PROMPT_TEMPLATE_CONFIG.load_prompt_template()


# =================================  Optimized Prompt Template ================================


optimized_template = """
Objective: {general_instructions}

Question:
{persona} was asked the following question. {question}

Answer Options:
{answer_options}

Instructions: Choose from the list of answer options to answer the question. Answer the question using only the provided answer options. If none of the options are correct, choose the option that is closest to being correct.

Answer:
"""

# Define Pydantic model for the Optimized Prompt Template
OPTIMIZED_PROMPT_TEMPLATE_CONFIG = NormalPromptTemplateConfig(
    type="normal",
    template=optimized_template
)

# Create the actual PromptTemplate using the Pydantic model
PROMPT_TEMPLATE_OPTIMIZED = OPTIMIZED_PROMPT_TEMPLATE_CONFIG.load_prompt_template()

# Define Pydantic model for the Optimized Chat Prompt Template
OPTIMIZED_CHAT_PROMPT_TEMPLATE_CONFIG = ChatPromptTemplateConfig(
    type="chat",
    messages=[
        ChatMessageConfig(
            role="system", content="Objective: {general_instructions}"),
        ChatMessageConfig(role="user", content=optimized_template)
    ]
)

# Create the actual ChatPromptTemplate using the Pydantic model
CHAT_PROMPT_TEMPLATE_OPTIMIZED = OPTIMIZED_CHAT_PROMPT_TEMPLATE_CONFIG.load_prompt_template()


# =================================  JSON Output Prompt Templat ================================


json_output_template = """
Objective: {general_instructions}

Question:
{persona} was asked the following question. {question}

Answer Options:
{answer_options}

Instructions: Choose from the list of answer options to answer the question. Answer the question using only the provided answer options. If none of the options are correct, choose the option that is closest to being correct. The solution must be provided in this format: {{"answer": "answer_option"}}.

Answer:
"""

# Define Pydantic model for the JSON Output Prompt Template
JSON_OUTPUT_PROMPT_TEMPLATE_CONFIG = NormalPromptTemplateConfig(
    type="normal",
    template=json_output_template
)

# Create the actual PromptTemplate using the Pydantic model
PROMPT_TEMPLATE_JSON_OUTPUT = JSON_OUTPUT_PROMPT_TEMPLATE_CONFIG.load_prompt_template()

# Define Pydantic model for the JSON Output Chat Prompt Template
JSON_OUTPUT_CHAT_PROMPT_TEMPLATE_CONFIG = ChatPromptTemplateConfig(
    type="chat",
    messages=[
        ChatMessageConfig(
            role="system", content="Objective: {general_instructions}"),
        ChatMessageConfig(role="user", content=json_output_template)
    ]
)

# Create the actual ChatPromptTemplate using the Pydantic model
CHAT_PROMPT_TEMPLATE_JSON_OUTPUT = JSON_OUTPUT_CHAT_PROMPT_TEMPLATE_CONFIG.load_prompt_template()
