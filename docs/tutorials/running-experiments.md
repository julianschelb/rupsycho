# Running Experiments

## Load

```python
import rupsycho as rup

experiment = rup.experiment_from_file("config.json")          # one file
experiment = rup.experiment_from_dict(config_dict)            # one dict
collection = rup.experiments_from_files("configs/*.json")     # many files (glob)
experiments = rup.experiments_from_dicts([cfg_a, cfg_b])      # many dicts (lazy)
```

`experiments_from_files` returns an
[`ExperimentCollection`][rupsycho.experiment_collection.ExperimentCollection];
`experiments_from_dicts` returns a generator that builds each experiment when you iterate
over it.

An invalid configuration is reported by printing the pydantic validation error. What you get
back depends on the function:

- `experiment_from_dict` raises a `RuntimeError` with the message
  `Failed to create experiment from dict:` and nothing after the colon. Read the printed
  error above it.
- `experiment_from_file` returns `None`.
- `experiments_from_files` and `experiments_from_dicts` skip the invalid entries and
  continue with the others.

A path or glob that matches no file raises a `RuntimeError`.

## Inspect before you spend compute

```python
experiment.print_assembled_prompt(item_idx=0, persona_idx=1)
print(experiment.list_models(), experiment.list_personas())
```

`print_assembled_prompt` prints the prompt for one item and one persona exactly as the model
receives it. If the prompt cannot be built it emits a `UserWarning` instead of raising.

## Run

```python
experiment.run()                       # every model × seed × item × persona
```

Instead of the plain call, pass [callbacks](callbacks.md) to receive every answer as soon as
it is generated:

```python
from rupsycho.callbacks import PrintTableCallback

experiment.run(callbacks=[PrintTableCallback()])
```

`run(cumulative=True)` switches to the response-memory mode described in
[Concepts](../concepts.md#the-run-loop).

For a collection, `collection.run_all()` calls `run()` on every experiment without
callbacks; loop over `collection.experiments` to pass arguments.

A progress bar is shown (the notebook variant is chosen automatically). Each model is
loaded just before its turn and released afterwards.

!!! warning "An experiment can be run once"
    Because models are released after use, calling `run()` a second time on the same
    experiment fails with `AttributeError: 'NoneType' object has no attribute 'load_model'`.
    Load the experiment again to repeat it (for models added with `add_model`, adding them
    again also works).

## Collect results

```python
df = experiment.get_answers_as_dataframe()   # flat: one row per generated answer
raw = experiment.get_answers()               # nested: per item → model → persona → seed
experiment.export_to_file("experiment.json") # configuration including all answers, as JSON
```

The data frame has the columns `Instruction ID`, `Instruction Question`, `Model ID`,
`Persona ID`, `Run Seed` and `Answer`. `get_answers()` returns one dictionary per item, in
the order of the questionnaire: `{model_id: {persona_id: {seed: answer}}}`. Calls that
failed have no entry.

`export_to_file` writes the whole experiment, including the model configurations (and
therefore any `api_key` you put there), so do not share the file without checking it. A
failure to write the file is printed rather than raised.

## Parse answers while they run

By default the answer is the model's text output. Set a LangChain output parser to
transform every answer as it is generated, for example to extract a value:

```python
from rupsycho.parsers.cleaners import RegexExtractorCleaner

experiment.set_parser(RegexExtractorCleaner(pattern=r'"answer":\s*"?([^"]*?)"?\s*}'))
```

## Manage experiments in code

```python
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.prompts import ChatPromptTemplate
from rupsycho.models.questionnaire import DemographicProfile

profile = DemographicProfile(attributes={"name": "Bob", "age": 40}, template="{name}, {age}")
experiment.add_persona(profile, identifier="Optimistic")
experiment.add_model(FakeListLLM(responses=["2"]), identifier="my-model")  # any chat model or LLM
experiment.remove_model("my-model")
experiment.set_prompt(
    ChatPromptTemplate.from_messages(
        [("system", "{general_instruction}"), ("user", "{persona_description}: {question}")]
    )
)
```

| What     | Methods                                                                                      |
| -------- | -------------------------------------------------------------------------------------------- |
| Models   | `add_model`, `get_model`, `has_model`, `list_models`, `count_models`, `remove_model`, `clear_models` |
| Personas | `add_persona`, `get_persona`, `list_personas`, `remove_persona`, `clear_personas`            |
| Prompt   | `set_prompt`, `get_prompt`, `has_prompt`, `reset_prompt`                                     |
| Parser   | `set_parser`                                                                                 |

Identifiers must be unique: adding a model or persona under an existing identifier, or
removing an unknown one, emits a warning. See the [API reference](../api/experiment.md) for
the details.

## Troubleshooting

Calls to the model that fail are logged and skipped, so a broken setup can look like a
successful run with no results. Python prints `ERROR` log messages to the standard error
stream; use `logging.basicConfig(level=logging.INFO)` to see more.

| Symptom                                                                 | Likely cause |
| ----------------------------------------------------------------------- | ------------ |
| `Error invoking chain for run: …` for every call, and an empty data frame | The prompt uses a placeholder that does not exist, or the model/API call fails (authentication, connection). Check with `print_assembled_prompt()` and the logged message. |
| An empty data frame but no errors                                       | The experiment has no personas in `demographic_profiles`. |
| `RuntimeError: Failed to create experiment from dict:` (nothing after the colon) | The configuration is invalid; read the validation error printed before it. Typical causes: no `parameters` key, a model or prompt without `type`, seeds that are not strings. |
| `AttributeError: 'NoneType' object has no attribute 'join_options'`     | Neither the item nor the questionnaire has answer options. |
| `TypeError: object of type 'NoneType' has no len()`                     | The questionnaire has no `instruction_items` (check the spelling of the key). |
| `ValueError: No models have been set in runnable_models.`               | No model is configured (`"models": {}`) and none was added with `add_model`, or models failed to load with `lazy_load_models` set to `false`. |
| `AttributeError: 'NoneType' object has no attribute 'load_model'`       | The experiment was already run once, see above. |

## Tips for robust studies

- Run **several seeds** and report the spread, not a single run. Check that the seed has an
  effect for your back-end, see [Reproducibility](../concepts.md#reproducibility).
- Test **paraphrased prompts** and **multiple personas** – effects that disappear under
  small prompt changes are not robust.
- Keep the exported experiment file together with your results.
