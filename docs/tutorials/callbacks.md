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

| Callback             | Output                                         |
| -------------------- | ---------------------------------------------- |
| `JSONLCallback`      | one JSON object per answer                     |
| `CSVCallback`        | CSV rows, header written once                  |
| `PrintCallback`      | verbose console output                         |
| `PrintTableCallback` | compact table in the console                   |

## Writing your own

Subclass [`Callback`][rupsycho.callbacks.answer_saving_callbacks.Callback] and implement
`save_answer`:

```python
from rupsycho.callbacks import Callback

class SqliteCallback(Callback):
    def save_answer(self, experiment, instruction_item_id, instruction_item,
                    model_id, profile_id, random_seed, time, answer):
        ...  # `time` is the generation time in seconds
```

Errors inside a callback are reported as warnings and never stop the experiment.
