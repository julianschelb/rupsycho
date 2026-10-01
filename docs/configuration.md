# Configuration Reference

An experiment configuration is a JSON object with the following top-level keys (`{}` stands for
the content of the section, described below; this skeleton is not loadable as it is).

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

| Key                    | Required                | Description                                                         |
| ---------------------- | ----------------------- | ------------------------------------------------------------------- |
| `name`                 | no                      | Name of the experiment; shown in the progress bar and in callbacks  |
| `description`          | no                      | Free text                                                           |
| `parameters`           | **yes** (may be `{}`)   | Seeds and loading behaviour, see [below](#parameters)               |
| `models`               | no                      | Omitted: a default model is used. `{}`: no config-defined models   |
| `prompt_template`      | no                      | Omitted: the [default chat prompt](#prompt_template) is used        |
| `demographic_profiles` | to get any answers      | The personas. Without a persona, `run()` generates nothing          |
| `questionnaire`        | **yes**                 | Items and answer options, see [below](#questionnaire)              |
| `metadata`             | no                      | Free-form object stored with the experiment                         |

Unknown keys are ignored without a warning, so a misspelt key (for example `instruction_item`
instead of `instruction_items`) only shows up later as a missing value.

A complete configuration that needs no model download (add a model in Python with
`experiment.add_model(...)`):

```json
{
  "name": "Minimal example",
  "parameters": {"seeds": ["1"]},
  "models": {},
  "demographic_profiles": {
    "Anna": {"attributes": {"name": "Anna", "age": 30}, "template": "a {age}-year-old person named {name}"}
  },
  "questionnaire": {
    "name": "Mini",
    "general_instruction": "Rate the statement.",
    "default_answer_options": {
      "1": {"text": "1. Disagree", "weight": 1},
      "2": {"text": "2. Agree", "weight": 2}
    },
    "instruction_items": [{"question": "I like tests."}]
  }
}
```

If a configuration is invalid (a missing required key, a wrong type, an unknown `type` of a
model or prompt), the error is printed and no experiment is created, see
[Running Experiments](tutorials/running-experiments.md#load).

## `parameters`

The `parameters` key must be present in every configuration; use `{}` to accept the defaults.

| Key                | Type        | Default                              | Description |
| ------------------ | ----------- | ------------------------------------ | ----------- |
| `seeds`            | list of str | one random seed (`"0"` – `"999999"`) | Seeds for the model calls, one run per seed. They must be strings (`["1", "2"]`) that can be converted to integers. A new random seed is drawn for every experiment; an empty list or `null` falls back to the seed `42`. |
| `lazy_load_models` | bool        | `true`                               | `true`: a model is loaded when `run()` reaches it and released afterwards. `false`: all models are loaded when the experiment is created; a model that fails to load is skipped with a printed message. |

Extra keys are allowed and stored with the experiment, but they are not passed to the models.
Put generation settings into the `parameters` of the model instead.

## `models`

A mapping from a free-form **model id** to a model configuration. The model id appears in
the results (`Model ID`). If the `models` key is omitted entirely, a small default model
(`HuggingFaceTB/SmolLM-1.7b-Instruct` on the CPU, id `default_model`) is used; set
`"models": {}` to configure none.

Every model configuration needs a `type`, which selects the back-end, and may contain
`parameters`: a mapping that is passed on to the model (the Hugging Face pipeline, or the
LangChain model class).

| `type`               | Required fields           | Optional fields (default)                                                                   |
| -------------------- | ------------------------- | ------------------------------------------------------------------------------------------- |
| `local_huggingface`  | `name_or_path`            | `task` (`"text-generation"`), `device_map` (`"auto"`), `tokenizer_name_or_path`, `bitsandbytes_config` |
| `remote_huggingface` | `repo_id`, `task`         |                                                                                             |
| `ollama`             | `model`                   | `base_url` (`"http://localhost:11434"`)                                                     |
| `openai`             | `name_or_path`            | `api_key`, `base_url`, `organization`                                                       |
| `google`             | `name_or_path`            | `api_key`                                                                                   |
| `deepseek`           | `name_or_path`            | `api_key`                                                                                   |
| `langchain`          | `definition`              |                                                                                             |

Details:

- `local_huggingface` models are loaded with `AutoModelForCausalLM`, wrapped in a
  Transformers pipeline (`parameters` are the pipeline arguments) and used through a chat
  interface, so the tokenizer needs a chat template. `bitsandbytes_config` is forwarded to
  `transformers.BitsAndBytesConfig`. The fields `revision`, `cache_dir` and
  `huggingfacehub_api_token` are accepted but currently not used when loading.
- `remote_huggingface` uses the Hugging Face Inference API; the access token is read from
  the `HUGGINGFACEHUB_API_TOKEN` environment variable.
- `openai` also works with OpenAI-compatible servers through `base_url`.
- `google` needs the key in `api_key`: the `GOOGLE_API_KEY` environment variable is not used
  when the model is created from the configuration. Google models do not receive the run seed.
- `deepseek` reads the key from the `DEEPSEEK_API_KEY` environment variable; the `api_key`
  field is currently not used.
- `langchain` holds a serialised LangChain object (as produced by
  `langchain_core.load.dumpd`). Recent `langchain-core` versions only deserialise classes of
  `langchain-core` itself by default, so add partner models such as `ChatOpenAI` in Python
  with `experiment.add_model(...)` instead.
- A `prompt_template` field is accepted on model configurations but currently ignored; use
  the experiment-level `prompt_template`.

See [Models](tutorials/models.md) for examples, API keys and seeds.

!!! danger "Keep secrets out of configs"
    Do not commit API keys, and note that `export_to_file` writes the model configurations
    including any `api_key` to the exported file. Pass keys through environment variables
    where the provider supports it (see [Models](tutorials/models.md#api-keys)) or add the
    model programmatically with `experiment.add_model(...)`.

## `prompt_template`

If the key is given, it needs a `type`.

| `type`      | Fields                                                                         |
| ----------- | ------------------------------------------------------------------------------ |
| `normal`    | `template` – a single template string                                          |
| `chat`      | `messages` – list of `{role, content}`, roles such as `system`, `user` (`human`), `assistant` (`ai`) |
| `langchain` | `definition` – a serialised LangChain prompt (as produced by `langchain_core.load.dumpd`) |

Available placeholders are listed in [Concepts](concepts.md#prompt-placeholders).
Literal braces must be doubled (`{{"answer": "…"}}`). A placeholder that is not on that list
makes every model call fail, see [Running Experiments](tutorials/running-experiments.md#troubleshooting).

[Cumulative mode](concepts.md#the-run-loop) requires a `chat` prompt with a system message
followed by a user message, and the user message may only use `{question}`.

??? note "Default prompt (used when `prompt_template` is omitted)"
    A `chat` prompt with this system message

    ```text
    Objective: "{general_instruction}"
    Answer with respect to the following persona description and question.
    ```

    and this user message:

    ```text
    Question:
    {persona_description} was asked the following question. {question}

    Answer Options:
    {answer_options}

    Instructions: Choose from the list of answer options to answer the question. Answer the question using only the provided answer options. If none of the options are correct, choose the option that is closest to being correct.

    Answer:
    ```

## `demographic_profiles`

A mapping from a profile id to a persona. The profile id appears in the results (`Persona ID`).

```json
"Optimistic Persona": {
  "attributes": {"age": 18, "title": "Ms", "name": "Muller"},
  "template": "{title} {name} is {age} years old and very open minded."
}
```

| Key          | Required | Description |
| ------------ | -------- | ----------- |
| `attributes` | yes      | A mapping of any keys. `age`, `title` and `name` are always defined and default to `null` (printed as `None`) |
| `template`   | no       | Format string that is filled with the attributes to give `{persona_description}`. Default: `"{title} {name} is {age} years old."` |

Every placeholder in `template` must be an attribute; an unknown one raises a `KeyError`
when the prompts are built.

## `questionnaire`

| Key                      | Required | Description |
| ------------------------ | -------- | ----------- |
| `name`                   | yes      | Name of the questionnaire |
| `general_instruction`    | yes      | Instruction shown to every participant (`{general_instruction}`) |
| `instruction_items`      | to run   | List of items |
| `default_answer_options` | see note | Answer options used by items without their own |
| `attributes`             | no       | Free-form metadata, e.g. a mapping of scale dimensions (default `{}`) |
| `demographic_profiles`   | no       | Accepted but not used; define the personas at the top level |

Every item needs answer options, either its own `answer_options` or the questionnaire's
`default_answer_options`; without both, the prompts cannot be built.

**Answer options** map an id to `{text, weight, ignored_for_scale}`:

| Key                 | Type | Default                | Description |
| ------------------- | ---- | ---------------------- | ----------- |
| `text`              | str  | `"Choose an option"`   | The text shown in the prompt |
| `weight`            | int  | `0`                    | Numeric value of the option |
| `ignored_for_scale` | bool | `false`                | Marks options that do not count towards a scale |

The options are shown in the order of the mapping. To control how they are joined, wrap
them as `{options, delimiter, prepend_delimiter}`:

```json
"answer_options": {
  "options": {
    "1": {"text": "1. Never", "weight": 1},
    "2": {"text": "2. Always", "weight": 2}
  },
  "delimiter": "\n",
  "prepend_delimiter": true
}
```

| Key                 | Type | Default | Description |
| ------------------- | ---- | ------- | ----------- |
| `options`           | map  | `{}`    | The answer options |
| `delimiter`         | str  | `", "`  | Inserted between the options in `{answer_options}` |
| `prepend_delimiter` | bool | `false` | Also put the delimiter before the first option |

**Items** have these keys:

| Key              | Type   | Default                      | Description |
| ---------------- | ------ | ---------------------------- | ----------- |
| `question`       | str    | `"Enter question text here"` | The question (`{question}`) |
| `reversed`       | bool   | `false`                      | Flag for reverse-keyed items |
| `answer_options` | object | none                         | Answer options of this item |
| `attributes`     | object | `{}`                         | Free-form metadata, e.g. `{"dimension": "1"}` |
| `answers`        | object | `{}`                         | Filled by `run()` (model id → persona id → seed → answer); part of the exported file, not something you write |

R.U.Psycho stores `weight`, `reversed`, `ignored_for_scale` and the `attributes` with the
questionnaire so that you can compute scale scores yourself. It does not compute scores.

A complete example is
[`examples/data/bfi_demo_config.json`](https://github.com/julianschelb/rupsycho/blob/main/examples/data/bfi_demo_config.json).
