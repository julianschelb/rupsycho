# Callbacks

Callbacks receive every answer **as soon as it is generated**, so partial results survive
a crash and long runs can be monitored.

```python
from rupsycho.callbacks import CSVCallback, JSONLCallback, PrintTableCallback

experiment.run(callbacks=[
    JSONLCallback("answers.jsonl"),
    CSVCallback("answers.csv"),
    PrintTableCallback(),
])
```

| Callback             | Output                                                                              |
| -------------------- | ----------------------------------------------------------------------------------- |
| `JSONLCallback`      | one JSON object per answer, appended to the file (default `experiment_output.jsonl`) |
| `CSVCallback`        | one CSV row per answer, appended to the file; the header is written once when the file is created (default `experiment_output.csv`) |
| `PrintCallback`      | verbose console output, one block per answer                                        |
| `PrintTableCallback` | compact table in the console, long questions and answers are truncated              |

Both file callbacks append to an existing file, so use a new file name (or delete the old
file) for a fresh run.

The CSV file has the columns `experiment_name`, `instruction_item_id`, `instruction_item`
(the question), `model_id`, `profile_id`, `random_seed`, `time` and `answer`. Each line of the
JSONL file has the keys `experiment_name`, `instruction_item_id`, `instruction_item` (the
complete item, including the answers collected so far), `model_id`, `profile_id`,
`random_seed`, `time` and `answer`. The CSV file is the input of the
[postprocessing pipeline](postprocessing.md#the-pipeline).

## Writing your own

Subclass [`Callback`][rupsycho.callbacks.answer_saving_callbacks.Callback] and implement
`save_answer`:

```python
from rupsycho.callbacks import Callback

class CollectCallback(Callback):
    def __init__(self):
        self.rows = []

    def save_answer(self, experiment, instruction_item_id, instruction_item,
                    model_id, profile_id, random_seed, time, answer):
        self.rows.append((model_id, profile_id, random_seed, instruction_item.question, answer, time))

collector = CollectCallback()
experiment.run(callbacks=[collector])
```

| Argument              | Meaning                                                              |
| --------------------- | -------------------------------------------------------------------- |
| `experiment`          | The running experiment                                               |
| `instruction_item_id` | Position of the item in the questionnaire, starting at 0             |
| `instruction_item`    | The item (`question`, `answer_options`, `attributes`, …)             |
| `model_id`            | Identifier of the model                                              |
| `profile_id`          | Identifier of the persona                                            |
| `random_seed`         | The seed of this run                                                 |
| `time`                | Generation time in seconds                                           |
| `answer`              | The generated answer, or `None` if the call failed                   |

A callback is also called when the model call failed, with `answer=None`; make sure your
callback can handle that (`PrintTableCallback` cannot and reports a warning for such
answers). Errors inside a callback are reported as warnings and never stop the experiment.
