# Concepts

## Anatomy of an experiment

An **experiment** ([`ExperimentDocument`][rupsycho.experiment.ExperimentDocument]) bundles
everything needed to repeat a psychometric test on a language model:

```mermaid
flowchart LR
    Q[Questionnaire<br/>items + answer options] --> P
    D[Demographic profiles<br/>personas] --> P
    T[Prompt template] --> P[Assembled prompt]
    P --> M[Model(s)<br/>+ seeds]
    M --> A[Free-text answer]
    A --> C[Cleaner → Validator → Judge]
    C --> R[Scorable response]
```

| Component              | What it is                                                                 |
| ---------------------- | -------------------------------------------------------------------------- |
| **Questionnaire**      | A general instruction, items (questions), and answer options with weights. |
| **Demographic profile**| A persona the model is asked to answer *as*, rendered from a template.     |
| **Prompt template**    | A plain, chat or LangChain template with the placeholders below.           |
| **Model**              | One or more models with their generation parameters.                       |
| **Seeds**              | Random seeds passed to the model for each run.                             |

## Prompt placeholders

Prompt templates can use these variables:

| Placeholder             | Filled with                                               |
| ----------------------- | --------------------------------------------------------- |
| `{general_instruction}` | The questionnaire's general instruction                   |
| `{persona_description}` | The rendered demographic profile                          |
| `{question}`            | The current item's question                               |
| `{answer_options}`      | The item's answer options (or the questionnaire default), joined with the configured delimiter |

## The run loop

`experiment.run()` visits every combination of

```
model × seed × item × persona
```

builds `prompt | model | parser` for it, records the answer and its latency on the item,
and passes it to all [callbacks](tutorials/callbacks.md). A failing invocation is logged
and recorded as *no answer* instead of aborting the whole experiment.

Prompt inputs are computed once per experiment, and models are loaded lazily and
released after use so that several large models can be tested one after another on
the same machine.

**Cumulative mode** (`run(cumulative=True)`) keeps a per-persona "response memory":
each earlier question and answer is appended to the persona's prompt, which lets you
study consistency across a questionnaire. This mode requires a user message template
that only contains `{question}`.

## Reproducibility

- Every experiment is a **single configuration** that can be versioned and shared.
- `parameters.seeds` are passed to each model call. **Set them explicitly** – if omitted,
  one random seed is drawn per experiment.
- `print_assembled_prompt` shows exactly what the model receives.
- Generation parameters live next to the model in the config, not in code.

## From text to scores

Language models answer in free text ("I would say 4, because …"). The
[`rupsycho.parsers`](tutorials/postprocessing.md) package turns such output into a
valid response in three steps:

1. **Cleaners** strip noise, repeated prompt text or extract a pattern.
2. **Validators** flag refusals, apologies and "as an AI" answers.
3. **Judges** choose the answer option that best matches the cleaned text.
