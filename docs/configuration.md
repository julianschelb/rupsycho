# Configuration Reference

An experiment configuration is a JSON object with the following top-level keys.

```json
{
  "name": "...",
  "description": "...",
  "parameters": {},
  "models": {},
  "prompt_template": {},
  "demographic_profiles": {},
  "questionnaire": {}
}
```

## `parameters`

| Key                | Type        | Description                                                        |
| ------------------ | ----------- | ------------------------------------------------------------------ |
| `seeds`            | list of str | Seeds for the model calls. Defaults to one random seed.            |
| `lazy_load_models` | bool        | Load each model only when it is needed (default `true`).           |

Extra keys are allowed and stored with the experiment.

## `models`

A mapping from a free-form **model id** to a model configuration. If the key is omitted
entirely, a small default model (`HuggingFaceTB/SmolLM-1.7b-Instruct`) is used; set
`"models": {}` to configure none. The `type` field
selects the back-end; `parameters` are passed to the model.

| `type`               | Key fields                                                                   |
| -------------------- | ---------------------------------------------------------------------------- |
| `local_huggingface`  | `name_or_path`, `task`, `device_map`, `revision`, `bitsandbytes_config`      |
| `remote_huggingface` | `repo_id`, `task`                                                            |
| `ollama`             | `model`, `base_url`                                                          |
| `openai`             | `name_or_path`, `api_key`, `base_url`, `organization`                        |
| `google`             | `name_or_path`, `api_key`                                                    |
| `deepseek`           | `name_or_path`, `api_key`                                                    |
| `langchain`          | `definition` (a serialised LangChain runnable)                               |

!!! danger "Keep secrets out of configs"
    Do not commit API keys. Pass them via environment variables or add the model
    programmatically with `experiment.add_model(...)`.

## `prompt_template`

| `type`      | Fields                                                              |
| ----------- | ------------------------------------------------------------------- |
| `normal`    | `template` – a single template string                               |
| `chat`      | `messages` – list of `{role, content}` (`system`, `user`, …)        |
| `langchain` | `definition` – a serialised LangChain prompt                        |

Available placeholders are listed in [Concepts](concepts.md#prompt-placeholders).
Literal braces must be doubled (`{{answer: "…"}}`).

## `demographic_profiles`

A mapping from a profile id to a persona:

```json
"Optimistic Persona": {
  "attributes": {"age": 18, "title": "Ms", "name": "Muller"},
  "template": "{title} {name} is {age} years old and very open minded."
}
```

`attributes` may contain any keys; the `template` is formatted with them.

## `questionnaire`

| Key                     | Description                                                       |
| ----------------------- | ----------------------------------------------------------------- |
| `name`                  | Name of the questionnaire                                         |
| `general_instruction`   | Instruction shown to every participant                            |
| `attributes`            | Free-form metadata, e.g. a mapping of scale dimensions            |
| `default_answer_options`| Answer options used by items without their own                    |
| `instruction_items`     | List of items                                                     |

**Answer options** map an id to `{text, weight, ignored_for_scale}`, or can be wrapped
as `{options, delimiter, prepend_delimiter}` to control how they are joined in the
prompt.

**Items** have a `question`, optional `reversed` flag (for reverse-keyed scoring),
optional `answer_options` and optional `attributes` (e.g. `{"dimension": "1"}`).

A complete example is
[`examples/data/bfi_demo_config.json`](https://github.com/julianschelb/rupsycho/blob/main/examples/data/bfi_demo_config.json).
