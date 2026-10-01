# Running Experiments

## Load

```python
import rupsycho as rup

experiment = rup.experiment_from_file("config.json")          # one file
experiment = rup.experiment_from_dict(config_dict)            # one dict
collection = rup.experiments_from_files("configs/*.json")     # many files (glob)
experiments = rup.experiments_from_dicts([cfg_a, cfg_b])      # many dicts (lazy)
```

Invalid configurations are reported (with the pydantic validation error) and skipped when
loading several at once.

## Inspect before you spend compute

```python
experiment.print_assembled_prompt(item_idx=0, persona_idx=1)
print(experiment.list_models(), experiment.list_personas())
```

## Run

```python
experiment.run()                       # every model × seed × item × persona
experiment.run(cumulative=True)        # response-memory mode, see Concepts
collection.run_all()                   # run all experiments of a collection
```

A progress bar is shown (the notebook variant is chosen automatically). Each model is
loaded just before its turn and released afterwards.

## Collect results

```python
df = experiment.get_answers_as_dataframe()   # flat: one row per combination
raw = experiment.get_answers()               # nested: per item → model → persona → seed
experiment.export_to_file("experiment.json") # the configuration including all answers
```

## Manage experiments in code

```python
experiment.add_persona(profile, identifier="Optimistic")
experiment.add_model(model, identifier="my-model")
experiment.remove_model("my-model")
experiment.set_prompt(chat_prompt_template)
```

## Tips for robust studies

- Run **several seeds** and report the spread, not a single run.
- Test **paraphrased prompts** and **multiple personas** – effects that disappear under
  small prompt changes are not robust.
- Keep the exported experiment file together with your results.
