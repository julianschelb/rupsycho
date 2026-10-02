# Experiment

::: rupsycho.experiment.ExperimentDocument
    options:
      members:
        - name
        - description
        - parameters
        - prompt_template
        - models
        - demographic_profiles
        - questionnaire
        - metadata
        - set_questionnaire
        - set_parser
      show_if_no_docstring: true
      show_labels: false

::: rupsycho.experiment_collection.ExperimentCollection

## Mixins

::: rupsycho.mixins.experiment_processing.ExperimentProcessingMixin

::: rupsycho.mixins.experiment_exporting.ExperimentExportMixin

::: rupsycho.mixins.model_managing.ModelManagementMixin

::: rupsycho.mixins.persona_managing.PersonaManagementMixin

::: rupsycho.mixins.prompt_managing.PromptTemplateMixin

## Run summary

::: rupsycho.mixins.experiment_processing.RunSummary
