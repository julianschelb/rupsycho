# Data Models

The classes below validate the experiment configuration. Their attributes are the keys of
the configuration, see the [Configuration Reference](../configuration.md) for how to use them.

## Questionnaire

::: rupsycho.models.questionnaire
    options:
      members:
        - Questionnaire
        - InstructionItem
        - AnswerOptions
        - AnswerOption
        - DemographicProfile
        - DemographicAttributes
      show_if_no_docstring: true
      show_labels: false
      filters: ["!^_", "!^model_config$"]

## Parameters

::: rupsycho.models.parameters.ExperimentParameters
    options:
      show_if_no_docstring: true
      show_labels: false
      filters: ["!^_", "!^model_config$"]

## Prompt templates

::: rupsycho.models.prompt
    options:
      show_if_no_docstring: true
      show_labels: false
      filters: ["!^_", "!^model_config$"]

## Model configurations

::: rupsycho.models.model
    options:
      members:
        - LangChainModelConfig
        - LocalHuggingFaceModelConfig
        - RemoteHuggingFaceModelConfig
        - OllamaModelConfig
        - OpenAIModelConfig
        - GoogleModelConfig
        - DeepSeekModelConfig
      show_if_no_docstring: true
      show_labels: false
      filters: ["!^_", "!^model_config$"]
