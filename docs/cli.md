# CLI Reference

The package installs the `rupsycho` command; `python -m rupsycho` is equivalent. It covers the
everyday workflow without writing any Python: check a configuration, preview the prompts, run
the experiment, score the answers.

```bash
rupsycho --help              # all commands
rupsycho run --help          # options and examples of one command
rupsycho --version
```

| Command | Purpose |
| --- | --- |
| [`run`](#run) | Run an experiment and collect the answers |
| [`validate`](#validate) | Check configuration files without loading any model |
| [`prompt`](#prompt) | Print the fully assembled prompt of one item and persona |
| [`examples`](#examples) | List, show and copy the bundled example configurations |
| [`postprocess`](#postprocess) | Clean, validate and score the answers of a run |
| [`configurator`](#configurator) | Launch the Streamlit configurator app |

## Conventions

**Data on stdout, diagnostics on stderr.** Answer tables, prompts, validation verdicts and
example files go to stdout, so they can be piped or redirected. Progress bars, summaries,
warnings and errors go to stderr and always start with `rupsycho:`.

**Exit codes** are shell friendly:

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | Invalid configuration, missing or protected files, a model that cannot be loaded, any other error. The message is one line on stderr, without a traceback. |
| `2` | Usage error (unknown option, missing argument, bad value); printed by the argument parser |
| `3` | `run` finished, but some model calls failed and their answers are missing |
| `130` | Interrupted with Ctrl-C (answers already streamed to a file are kept) |

**Nothing is overwritten by accident.** `run`, `postprocess` and `examples copy` refuse to
replace an existing file unless you pass `-f` / `--force`.

**Environment variable.** `RUPSYCHO_DEBUG=1` prints the full traceback of an unexpected error
instead of the one-line message.

**Models load lazily.** Whatever `parameters.lazy_load_models` says in the file, the command
loads models one at a time when the run reaches them. `validate`, `prompt` and `--dry-run`
therefore never download or load a model.

## `run`

Run an experiment: every model answers every item as every persona, once per seed.

```bash
rupsycho run config.json -o results.csv                 # stream the answers into a CSV file
rupsycho run config.json --seeds 1 2 3 -o runs.jsonl    # three repetitions as JSON lines
rupsycho run config.json --model small --dry-run        # preview the work, load no model
rupsycho run config.json                                # print the answers as a table
```

| Option | Description |
| --- | --- |
| `CONFIG` | Experiment configuration (JSON file, see the [Configuration Reference](configuration.md)) |
| `-o`, `--output FILE` | Write the answers to `FILE`. The format follows the extension: `.csv`, `.jsonl` or `.json`. Without it the answers are printed as a table. |
| `-f`, `--force` | Overwrite `FILE` if it exists |
| `--model ID [ID ...]` | Run only these model ids (the keys of `models`); may be repeated. Models that are not selected are not loaded. |
| `--seeds S [S ...]` | Use these seeds instead of `parameters.seeds` (non-negative integers) |
| `--cumulative` | Let each persona remember its earlier answers (needs a chat prompt) |
| `--max-concurrency N` | Calls to run at the same time. Only API models run in parallel; local Hugging Face models always run one call at a time and the command says so. |
| `--on-error {warn,raise,ignore}` | What to do when a model call fails: log it and go on (`warn`, default), stop at the first failure (`raise`) or go on silently (`ignore`) |
| `--dry-run` | Load no model and write nothing; print the number of calls and the first assembled prompt |
| `-q`, `--quiet` | Hide the progress bar and the final summary (failures are still reported) |

### Output formats

| Extension | Content | Written |
| --- | --- | --- |
| `.csv` | One row per answer: `experiment_name`, `instruction_item_id`, `instruction_item`, `model_id`, `profile_id`, `random_seed`, `time`, `answer`. This is the input of [`postprocess`](#postprocess). | while running |
| `.jsonl` | The same fields, one JSON object per line | while running |
| `.json` | The whole experiment including the answers, as written by `experiment.export_to_file`; load it again with `rup.experiment_from_file` | at the end |
| none | A table with the columns `Instruction ID`, `Instruction Question`, `Model ID`, `Persona ID`, `Run Seed`, `Answer` on stdout | at the end |

`.csv` and `.jsonl` files grow with the run, so an interrupted or crashed run keeps the
answers it has already collected. A failed call has an empty answer in the CSV file, `null` in
the JSON lines, and is left out of the table and of the `.json` export.

### Reproducibility

A run is determined by the configuration and its seeds: running the same command twice
produces the same answers (the `time` column is the only difference). The seed of every
answer is recorded in the `random_seed` column. If the configuration defines no `seeds`, a
random one is drawn and the command says so; pass `--seeds` to make the run reproducible.

### Failed calls

A failed call does not stop the run (unless `--on-error raise`). Its answer is missing, a
summary line names the first error, and the exit code is `3`:

```text
rupsycho: 8 model calls in 0.1s, 4 failed (first error: RuntimeError: cannot answer); answers in results.csv
```

The exit code is `3` for `warn` and `ignore` alike, so scripts never mistake an incomplete
result for a complete one.

```bash
rupsycho run study.json -o results.csv
case $? in
  0) echo "complete" ;;
  3) echo "finished, but some answers are missing" ;;
  *) echo "aborted" ;;
esac
```

## `validate`

Check one or more configuration files without loading any model. Prints one line per file and
exits with `1` if any file is invalid.

```bash
rupsycho validate config.json
rupsycho validate configs/*.json
```

```text
OK  bfi.json: Generative Models for Big Five Inventory - 4 items x 2 personas x 1 models x 1 seeds = 8 calls
ERROR typo.json: unknown top-level key(s): 'modles' (did you mean 'models'?)
ERROR missing.json: no such file
```

A file is valid if it is a JSON object, passes the schema, has no unknown top-level keys (a
misspelled `models` would silently fall back to the default model) and can actually be run:
it needs a questionnaire with items, personas, at least one model, integer seeds and a prompt
template that loads, and every persona and every item must produce a prompt. `run` makes the
same checks before it loads the first model. The `OK` line shows the size of the experiment:
items x personas x models x seeds.

## `prompt`

Print the fully assembled prompt of one item and persona, exactly as the model would receive
it. The prompt is the only thing on stdout, so it can be redirected into a file.

```bash
rupsycho prompt config.json
rupsycho prompt config.json --item 3 --persona "Conservative Persona" > prompt.txt
```

| Option | Description |
| --- | --- |
| `CONFIG` | Experiment configuration (JSON file) |
| `--item I` | Index of the instruction item, counting from 0 (default `0`) |
| `--persona P` | Index or id of the persona (default `0`) |

An index outside the questionnaire is an error that names the valid range. Items without
answer options of their own use the questionnaire's `default_answer_options`. No model is
needed, so the configuration may have `"models": {}`.

## `examples`

Work with the example configurations that ship with the package.

```bash
rupsycho examples list                      # names, one per line
rupsycho examples show bfi                  # print one as JSON
rupsycho examples copy bfi my-study.json    # a starting point for your own experiment
rupsycho examples copy bfi studies/         # into a directory: studies/bfi.json
```

| Action | Description |
| --- | --- |
| `list` | Print the names of the bundled examples |
| `show NAME` | Print the example as JSON |
| `copy NAME DEST [-f]` | Write the example to `DEST` (a file or a directory). Existing files are only replaced with `-f` / `--force`. |

## `postprocess`

Turn free-text answers into scores. The command reads the CSV files written by `run -o`
and applies the rule-based defaults of the [postprocessing
pipeline](tutorials/postprocessing.md): a cleaner, the combined validator (apologies,
"as an AI" and refusals) and the multiple-choice judge, which picks the answer option the
text matches best. Items without answer options of their own are judged against the
questionnaire's default options.

```bash
rupsycho postprocess config.json results.csv -o scored.csv
rupsycho postprocess config.json 'runs/*.csv' -o scored.csv --pattern 'answer:\s*"([^"]*)"'
```

| Option | Description |
| --- | --- |
| `CONFIG` | The configuration the results come from (it provides the answer options) |
| `RESULTS ...` | CSV files written by `run -o`; globs are allowed and may be quoted |
| `-o`, `--output OUT` | The processed CSV file (required) |
| `--cleaner {basic,regex}` | `basic` (default) removes line breaks and non-ASCII characters; `regex` extracts the answer with `--pattern` |
| `--pattern REGEX` | Regular expression with one capturing group that extracts the answer; implies `--cleaner regex` |
| `-f`, `--force` | Overwrite `OUT` if it exists |
| `-q`, `--quiet` | Hide the progress bars and the summary |

The output has all columns of the results plus four new ones:

| Column | Content |
| --- | --- |
| `cleaned_answer` | The answer after the cleaner |
| `validation_status` | The verdict of the validator, with details |
| `valid` | `True` if the validator found nothing wrong |
| `decision` | The answer option chosen by the judge; `not present` if the text matches no option, `inconclusive` if it matches several equally well |

`postprocess` never overwrites a results file, not even with `--force`. For other cleaners,
validators and judges (for example the model-based ones) use the
[`PostprocessingPipeline`](tutorials/postprocessing.md) from Python.

## `configurator`

Launch the [Streamlit configurator app](tutorials/configurator.md), a graphical editor for
experiment configurations. It needs the `configurator` extra:

```bash
pip install 'rupsycho[configurator]'
rupsycho configurator          # the same as rup-configurator
rupsycho configurator -- --server.port 8502
```

Arguments after `--` are passed on to `streamlit run`.

## Worked example: from configuration to scores

The bundled Big Five example, from a fresh install to a scored table.

**1. Get a configuration and check it.** Copying gives you a file to edit; `validate` loads
no model.

```bash
rupsycho examples copy bfi bfi.json
rupsycho validate bfi.json
```

```text
rupsycho: wrote bfi.json
OK  bfi.json: Generative Models for Big Five Inventory - 4 items x 2 personas x 1 models x 1 seeds = 8 calls
```

**2. Preview the work.** The dry run shows how many model calls three seeds mean and what
the model will be asked. Nothing is loaded and nothing is written.

```bash
rupsycho run bfi.json --seeds 1 2 3 --dry-run
```

```text
Dry run: no model was loaded and nothing was written.
experiment: Generative Models for Big Five Inventory
models:     Qwen/Qwen2.5-0.5B-Instruct
seeds:      1, 2, 3
calls:      4 items x 2 personas x 1 models x 3 seeds = 24 calls
output:     stdout (table)

Assembled prompt (item 0, persona 0 'Optimistic Persona'):
System: Objective: Act like you are Ms Muller is 18 years old and very open minded with a optimistic personality., a survey participant answering a questionnaire.
...
The solution must be provided in this format: {answer: "answer option"}
Human: Question: I see myself as someone who...
Answer Options: 1. Disagree strongly, 2. Disagree a little, 3. Neither agree nor disagree, 4. Agree a little, 5. Agree strongly
Answer:
```

**3. Run it.** The model is downloaded on first use. The answers stream into the CSV file
while the run is going.

```bash
rupsycho run bfi.json --seeds 1 2 3 -o results.csv
```

```text
Generative Models for Big Five Inventory: 100%|██████████| 24/24 [00:41<00:00, 1.72s/ prompts]
rupsycho: 24 model calls in 41.3s; answers in results.csv
```

The instruction asks for the format `{answer: "answer option"}`, so each row of `results.csv`
holds the model's raw text:

```text
experiment_name,instruction_item_id,instruction_item,model_id,profile_id,random_seed,time,answer
Generative Models for Big Five Inventory,0,I see myself as someone who...,Qwen/Qwen2.5-0.5B-Instruct,Optimistic Persona,1,1.912,"{answer: ""5. Agree strongly""}"
Generative Models for Big Five Inventory,0,I see myself as someone who...,Qwen/Qwen2.5-0.5B-Instruct,Conservative Persona,1,1.874,"{answer: ""3. Neither agree nor disagree""}"
...
```

**4. Score it.** The regular expression pulls the chosen option out of the braces; the judge
maps it onto the answer options of the questionnaire.

```bash
rupsycho postprocess bfi.json results.csv -o scored.csv --pattern 'answer:\s*"([^"]*)"'
```

```text
rupsycho: wrote scored.csv (24 rows)
```

**5. Analyse it** with any tool that reads CSV, for example pandas:

```python
import pandas as pd

scores = pd.read_csv("scored.csv")
scores = scores[scores["valid"]]
print(scores.groupby("profile_id")["decision"].value_counts())
```

Everything the command does is available in Python as well:

| Command | Python |
| --- | --- |
| `run` | `rup.experiment_from_file(...)`, then `experiment.run(callbacks=[CSVCallback(...)])` |
| `validate` | `rup.experiment_from_file(...)` |
| `prompt` | `experiment.print_assembled_prompt(item_idx, persona_idx)` |
| `examples` | `rup.list_examples()`, `rup.load_example_config(name)` |
| `postprocess` | `PostprocessingPipeline(...).run()` |
| `configurator` | `rup-configurator` |

See [Running Experiments](tutorials/running-experiments.md) and
[Callbacks](tutorials/callbacks.md) for the Python side.
