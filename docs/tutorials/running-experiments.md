# Running Experiments

## Load

```python
import rupsycho as rup

experiment = rup.experiment_from_file("config.json")          # one file
experiment = rup.experiment_from_dict(config_dict)            # one dict
collection = rup.experiments_from_files("configs/*.json")     # many files (glob)
experiments = rup.experiments_from_dicts([cfg_a, cfg_b])      # many dicts (lazy)
experiment = rup.load_example_experiment("bfi")               # bundled example
```

`experiments_from_files` returns an
[`ExperimentCollection`][rupsycho.experiment_collection.ExperimentCollection] ordered by file
name; `experiments_from_dicts` returns a generator that builds each experiment when you iterate
over it.

Loading **one** experiment never fails silently:

- `experiment_from_file` raises `FileNotFoundError` for a missing file and `ValueError` for
  invalid JSON or an invalid configuration (pydantic's `ValidationError` is a `ValueError`; the
  message names the offending section).
- `experiment_from_dict` raises `ValueError` the same way.

When loading **several** experiments, invalid ones are skipped and reported through the
`rupsycho.reader` logger (`strict=True` raises instead). A path or glob that matches no file
raises a `ValueError`. A configuration without a `parameters` section is valid.

## Inspect before you spend compute

```python
print(experiment.assemble_prompt(item_idx=0, persona_idx=1))   # the text as a string
experiment.print_assembled_prompt(item_idx=0, persona_idx=1)   # the same, printed
print(experiment.list_models(), experiment.list_personas())
```

`assemble_prompt` raises on an out-of-range index or an unusable template;
`print_assembled_prompt` emits a `UserWarning` instead.

## Run

```python
summary = experiment.run()             # every model × seed × item × persona
print(summary)                         # "160 model calls in 12.3s"
```

`run` returns a [`RunSummary`][rupsycho.mixins.experiment_processing.RunSummary] with
`n_calls`, `n_failed`, `n_succeeded`, `elapsed` and the first distinct `errors`.

| Argument | Meaning |
| --- | --- |
| `callbacks` | [Callbacks](callbacks.md) that receive every answer as soon as it is generated |
| `cumulative` | Response-memory mode, see [Concepts](../concepts.md#the-run-loop) |
| `max_concurrency` | Calls in parallel (API models); results and callbacks keep their order. Local Hugging Face models always run sequentially |
| `on_error` | `"warn"` (default): log each failed call, continue, warn once at the end; `"raise"`: stop at the first failure; `"ignore"` |
| `show_progress` | Show a progress bar (`False` for scripts and tests) |

```python
from rupsycho.callbacks import CSVCallback

experiment.run(callbacks=[CSVCallback("answers.csv")], max_concurrency=8, on_error="warn")
```

A failed call has no stored answer (callbacks receive `None`), so a broken setup shows up in the
summary instead of looking like a silent success:

```python
summary = experiment.run(on_error="ignore")
if summary.n_failed:
    print(summary.errors)              # e.g. ['AuthenticationError: invalid API key']
```

For a collection, `collection.run_all(**run_kwargs)` returns one `RunSummary` per experiment.

Experiments can be run **repeatedly**: lazily loaded models are released after use and loaded
again from their configuration; models added with `add_model` are kept.

## Collect results

```python
df = experiment.get_answers_as_dataframe()   # flat: one row per generated answer
raw = experiment.get_answers()               # nested: per item → model → persona → seed
experiment.export_to_file("experiment.json") # configuration including all answers, as JSON
config = experiment.to_config(include_answers=False)   # the definition only, as a dict
```

The data frame has the columns `Instruction ID`, `Instruction Question`, `Model ID`,
`Persona ID`, `Run Seed` and `Answer` (also when it is empty). `get_answers()` returns one
dictionary per item, in the order of the questionnaire: `{model_id: {persona_id: {seed: answer}}}`.

The export contains only the configuration (and the answers): runtime objects are left out and
**API keys are masked**, so the file can be shared and loaded again with
`rup.experiment_from_file`. Models added with `add_model` cannot always be serialised; add them
again after loading.

## Parse answers while they run

By default the answer is the model's text output. Set a LangChain output parser to
transform every answer as it is generated, for example to extract a value:

```python
from rupsycho.parsers.cleaners import RegexExtractorCleaner

experiment.set_parser(RegexExtractorCleaner(pattern=r'"answer":\s*"?([^"]*?)"?\s*}'))
```

## Manage experiments in code

```python
from langchain_core.prompts import ChatPromptTemplate
from rupsycho.models.questionnaire import DemographicProfile

profile = DemographicProfile(attributes={"name": "Bob", "age": 40}, template="{name}, {age}")
experiment.add_persona(profile, identifier="Optimistic")
experiment.add_model(my_langchain_model, identifier="my-model")  # any chat model or LLM
experiment.remove_model("my-model")
experiment.set_prompt(
    ChatPromptTemplate.from_messages(
        [("system", "{general_instruction}"), ("user", "{persona_description}: {question}")]
    )
)
```

| What     | Methods                                                                                      |
| -------- | -------------------------------------------------------------------------------------------- |
| Models   | `add_model`, `replace_model`, `get_model`, `has_model`, `list_models`, `count_models`, `remove_model`, `clear_models` |
| Personas | `add_persona`, `get_persona`, `list_personas`, `remove_persona`, `clear_personas`            |
| Prompt   | `set_prompt`, `set_prompt_config`, `get_prompt`, `get_prompt_config`, `has_prompt`, `reset_prompt` |
| Parser   | `set_parser`                                                                                 |

Adding a model or persona under an existing identifier **replaces** it and emits a warning;
removing an unknown one only emits a warning. See the [API reference](../api/experiment.md).

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `RuntimeWarning: N of M model calls failed …` | Read `summary.errors` and the log messages of the `rupsycho.mixins.experiment_processing` logger. Typical causes: authentication, connection, a prompt placeholder that does not exist (check with `assemble_prompt()`). |
| An empty data frame but no errors | The experiment has no personas in `demographic_profiles`. |
| `ValueError: 'questionnaire' must be an object …` | A section of the configuration has the wrong type; the message names it. |
| `ValueError: Item(s) […] have no answer options …` | Neither the items nor the questionnaire define answer options. |
| `ValueError: The questionnaire has no instruction_items to ask.` | Check the spelling of the `instruction_items` key. |
| `ValueError: No models have been set in runnable_models.` | No model is configured (`"models": {}`) and none was added with `add_model`. |
| `ImportError: … Install it with: pip install 'rupsycho[huggingface]'` | The extra of the model back-end is not installed. |

## Tips for robust studies

- Run **several seeds** and report the spread, not a single run; see
  [Reproducibility](reproducibility.md) for how seeds reach each back-end.
- Test **paraphrased prompts** and **multiple personas** – effects that disappear under
  small prompt changes are not robust.
- Keep the exported experiment file together with your results.
