# Configurator App

The configurator is a [Streamlit](https://streamlit.io) app for building experiment
configurations without writing JSON.

```bash
pip install "rupsycho[configurator] @ git+https://github.com/julianschelb/rupsycho.git"
rup-configurator
```

The page has two parts: the **Configurator** with a tab for each part of the experiment,
and **Input/Output** showing the questionnaire text and the resulting configuration.

| Tab                    | Purpose                                                          |
| ---------------------- | ---------------------------------------------------------------- |
| Experiment Info        | Name and description                                             |
| Demographic Profiles   | Add, edit, delete and duplicate personas                         |
| Questionnaire Info     | Name and instruction                                             |
| Questionnaire Items    | Items and answer options, optionally with a global answer set    |
| Tools                  | LLM-assisted import, configuration import                        |

At any time the **download** button saves the current configuration as JSON.

## Workflows

**Manual** – fill in the tabs. The *Resulting Configuration* updates continuously.

**LLM-assisted questionnaire import** – in *Tools*, enter an OpenAI API key and either
upload the questionnaire as PDF or paste its text. Select the relevant page range to
reduce noise, then run. GPT-4o mini fills the questionnaire section (a run costs a
fraction of a cent) and you continue editing as usual. A run **overwrites** the
questionnaire section, so do it first.

**Import an existing configuration** – upload a JSON file. It is accepted only if all
expected keys are present; unsupported sections (`parameters`, `prompt_template`,
`models`, `attributes`) pass through unchanged.

**Import personas from CSV** – the header must contain `title`, `name` and `ethnicity`, and
no value may be empty. The profiles replace the current ones.

!!! note
    The app edits the questionnaire and personas. Models, prompt template and parameters are
    not editable in the app – add them to the downloaded JSON.
