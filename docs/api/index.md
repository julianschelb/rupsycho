# API Reference

The API reference is generated from the docstrings in the source code. The configuration
file format is described in the [Configuration Reference](../configuration.md).

| Page                                | Contents                                                                                          |
| ----------------------------------- | ------------------------------------------------------------------------------------------------- |
| [Experiment](experiment.md)         | `ExperimentDocument`, `ExperimentCollection` and their mixins (running, models, personas, prompt, export) |
| [Loading](loading.md)               | `experiment_from_file`, `experiment_from_dict`, `experiments_from_files`, `experiments_from_dicts`, the loader, the example experiment and utility functions |
| [Data Models](models.md)            | Pydantic models of the configuration (questionnaire, parameters, prompts, models)                 |
| [Parsers](parsers.md)               | Cleaners, validators and judges                                                                   |
| [Callbacks](callbacks.md)           | Answer-saving callbacks                                                                           |
| [Postprocessing](postprocessing.md) | `PostprocessingPipeline`                                                                          |
| [Scoring](scoring.md)               | Item and scale scores from judged answers                                                         |
| [Seeding](seeding.md)               | How seeds reach each model back-end                                                               |
