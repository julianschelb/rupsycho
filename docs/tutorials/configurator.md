# Configurator App

The configurator is a [Streamlit](https://streamlit.io) app for building the personas and the
questionnaire of an experiment configuration without writing JSON. It is meant to be run
locally.

```bash
pip install "rupsycho[configurator] @ git+https://github.com/julianschelb/rupsycho.git"
rup-configurator
```

The command starts Streamlit and the app opens in your browser (by default at
`http://localhost:8501`); stop it with `Ctrl+C` in the terminal.

The page has two parts. The **Configurator** (left) has a tab for each part of the experiment.
**Input / Output** (right) has two tabs: **Questionnaire Text**, the text that the
LLM-assisted import reads, and **Resulting Configuration**, the JSON that the app builds, which
is updated continuously. At any time the **Download** button saves the current configuration as
`rupsycho_experiment_config.json`.

| Tab                  | Purpose                                                                            |
| -------------------- | ---------------------------------------------------------------------------------- |
| Tools                | *Language Model*: LLM-assisted questionnaire import. *Import Configuration*: load an existing configuration. *Help* |
| Experiment Info      | Name and description of the experiment                                             |
| Demographic Profiles | Add, edit, delete and duplicate personas (title, name, ethnicity); import them from a CSV file |
| Questionnaire Info   | Name and general instruction of the questionnaire                                  |
| Questionnaire Items  | Add, edit, delete and duplicate items and their answer options, optionally with one global answer set |

## Workflows

**Manual** – fill in the tabs; you can ignore *Tools* and *Questionnaire Text*. The
*Resulting Configuration* updates as you type. With many profiles the app needs a moment to
respond to changes. If all items share the same answer options, switch on *Global answer set*
in *Questionnaire Items*: you enter the options once (and can mark all items as reverse-scored)
and they are applied to every item.

**LLM-assisted questionnaire import** – in *Tools* → *Language Model*:

1. Enter an OpenAI API key. It is checked immediately and, if valid, kept in the running app
   (as `OPENAI_API_KEY` in its process environment); it is not written to the configuration.
2. Provide the questionnaire: upload a PDF, or paste or edit the text in *Questionnaire Text*.
   Cleaning the text (removing irrelevant parts, fixing broken formatting) improves the result,
   especially for lists of answer options that PDF extraction tends to scramble. For a PDF with
   several pages, a slider selects the range of pages that is used.
3. Press **Run**. It is enabled once the key is valid and there is text. The text is sent to
   OpenAI GPT-4o mini in one request (double quotes, slashes and backslashes are removed from
   the text first), which can take a moment for a long questionnaire; a run typically costs a
   fraction of a cent.
4. The model's output replaces the questionnaire name, the instruction and all items. A single
   answer set in the text is applied to every item; otherwise the answer sets are matched
   to the questions by position. Continue editing as usual, or press **Run** again; the text
   stays in its field.

A run **overwrites** the questionnaire part of the configuration, so do it first. If the model
does not return a usable result ("Model error, please try again"), the whole configuration is
reset to its empty initial state.

**Import an existing configuration** – in *Tools* → *Import Configuration*, upload a JSON
file. A successful import replaces the entire current configuration; a rejected file
("Invalid configuration") resets the app to its empty initial state. The file is accepted only if
all of these keys are present:

- top level: `name`, `description`, `parameters`, `prompt_template`, `models`,
  `demographic_profiles`, `questionnaire`
- every persona: `attributes` with `title`, `name` and `ethnicity`
- `questionnaire`: `name`, `general_instruction`, `attributes`, `instruction_items`
- every item: `question`, `reversed`, `answer_options` (each option with `text`, `weight` and
  `ignored_for_scale`) and `attributes`

Other keys are ignored.

The app does not support `default_answer_options`, so configurations that use them (such as
`bfi_demo_config.json`) are rejected; give every item its own `answer_options` instead. The
sections `parameters`, `prompt_template`, `models` and the questionnaire `attributes` cannot be
edited in the app. They are imported as they are and written back unchanged on download. Of the
configurations in `examples/data/`, `bdi_qwen72.json` and the three `rfq_*_small.json` files
can be imported.

**Import personas from CSV** – in *Demographic Profiles* → *Import from CSV*. The header must
contain `title`, `name` and `ethnicity`, and no value may be empty. The profiles replace the
current ones; a rejected file ("Invalid file structure") leaves a single empty profile.

## Things to know

- **Weights.** Answer options are numbered from 0 in the order in which they are listed
  (`0`, `1`, `2`, …) and `ignored_for_scale` is always `false`; neither can be edited. An import
  renumbers the weights of the imported options in the same way. Adjust them in the downloaded
  JSON if your scoring needs other values.
- **Reversed items.** After importing a configuration, all *Reversed scoring* switches are off,
  so set them again for reverse-keyed items (or fix `reversed` in the downloaded JSON).
- **Personas.** The app writes each persona with the template `{title} {name}` and the
  attributes `title`, `name`, `ethnicity` and `id`. Other attributes of imported personas
  (for example `age`) and their templates are dropped. `ethnicity` is only stored: add
  `{ethnicity}` to the persona `template` in the JSON if the prompt should mention it.
- **Models, seeds and prompt.** The downloaded configuration has `"parameters": {}` (a random
  seed is drawn per experiment), `"models": {}` and a default chat `prompt_template` that asks
  the model to answer in the format `{"answer": "answer option"}`. Add a model and, if you
  like, seeds before running the experiment.
