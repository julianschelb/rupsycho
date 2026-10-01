# Parsing & Postprocessing

Raw model answers are free text. The `rupsycho.parsers` package provides LangChain output
parsers that you can use on their own (`parser.invoke(text)`), as an answer parser while the
experiment runs (`experiment.set_parser(parser)`), or via the
[`PostprocessingPipeline`][rupsycho.postprocessing.PostprocessingPipeline].

## Cleaners

```python
from rupsycho.parsers.cleaners import BasicCleaner, PromptRemovalCleaner, RegexExtractorCleaner

BasicCleaner().invoke("Hello,\nworld 😊")                                 # 'Hello, world'
RegexExtractorCleaner(pattern=r'"answer":\s*"?([^"]*?)"?\s*}').invoke('{"answer": "4"}')  # '4'

PromptRemovalCleaner(prompt="Question: I like tests. Answer:", similarity_threshold=0.5).invoke(
    "Question: I like tests. Answer: 4. Agree a little"
)
# {'completion': '4. agree a little', 'similarity_score': 0.775, 'similarity_threshold': 0.5}
```

- `BasicCleaner` replaces line breaks and runs of whitespace by single spaces and removes all
  non-ASCII characters, including accented letters.
- `RegexExtractorCleaner` returns the first capturing group of the pattern, or the unchanged
  text if the pattern does not match.
- `PromptRemovalCleaner` removes a copy of the prompt that the model echoed. It lower-cases the
  text and returns a dictionary (`completion`, `similarity_score`, `similarity_threshold`)
  instead of a string, so its result cannot be passed to a validator or to the pipeline
  as it is.

## Validators

Validators return a dictionary with the original text and a `validation_status` of `"valid"`
or `"invalid"`.

```python
from rupsycho.parsers.validators import ValidatorParser

ValidatorParser().invoke("As an AI, I cannot answer that.")
# {'text': 'As an AI, I cannot answer that.',
#  'validation_status': 'invalid',
#  'details': {'apologies': False, 'being_ai': True, 'refusal': True}}
```

`ApologiesValidatorParser`, `BeingAiValidatorParser` and `RefusalValidatorParser` each check
one kind of answer, and `ValidatorParser` combines them (the answer is invalid if any of
them flags it, and `details` shows which ones did):

| Validator                   | Flags answers that …                                                              |
| --------------------------- | --------------------------------------------------------------------------------- |
| `ApologiesValidatorParser`  | start with an apology ("Sorry", "I apologize", …)                                 |
| `BeingAiValidatorParser`    | start with "As an AI", "I am an AI", …                                            |
| `RefusalValidatorParser`    | start with a refusal ("No,", "I cannot", "I can't", …) or contain phrases such as "I do not have a personal opinion" |

`ModelBasedValidator` uses a rejection classifier from the Hugging Face hub (by default
`ProtectAI/distilroberta-base-rejection-v1`, downloaded on first use). It returns
`text`, `validation_status` and a `confidence_score`.

## Judges

Judges select the answer option that best matches the text.

```python
from rupsycho.parsers.judges import MultipleChoiceJudge

judge = MultipleChoiceJudge(["1. Disagree", "2. Neutral", "3. Agree"])
judge.invoke("I would choose option 3 because I agree.")   # '3. Agree'
judge.invoke("Banana")                                      # 'not present'
judge.invoke("Either 1 or 3")                               # 'inconclusive'
```

`MultipleChoiceJudge` looks for the number (or letter) and for the text of each option in the
answer. It returns the option with the most matches, `"not present"` if no option matches and
`"inconclusive"` if several options match equally often.

`ModelBasedAnswerJudge` does the same with a fine-tuned RoBERTa classifier, for example
`julian-schelb/rup-answer-option-likert-scale`; it returns `"inconclusive"` if the model's
probabilities are too spread out (`entropy_threshold`). Pass `device="cpu"` if you have no GPU,
because the default is `cuda:0`:

```python
from rupsycho.parsers.judges import ModelBasedAnswerJudge

judge = ModelBasedAnswerJudge(
    model_name="julian-schelb/rup-answer-option-likert-scale",
    possible_answers=["1. never or seldom", "2.", "3. sometimes", "4.", "5. always"],
    device="cpu",
)
judge.parse("I always do my best to be honest.")   # '5. always'
```

`DemographicsJudge` extracts demographic information from answers to questions such as "What
gender do you identify with?". The task is chosen by the answer option of the item: its text
must be `gender` (result: `male`, `female` or `other`, or `not present` / `inconclusive`) or `age`
(result: the age as digits, or `inconclusive`). The age judge needs the `num2words` package.
See `examples/data/example_config_for_demographics_judge.json` for a configuration.

## The pipeline

[`PostprocessingPipeline`][rupsycho.postprocessing.PostprocessingPipeline] applies a cleaner,
a validator and a judge to the CSV files written by the
[`CSVCallback`](callbacks.md):

```python
import rupsycho as rup
from rupsycho.callbacks import CSVCallback
from rupsycho.parsers.cleaners import BasicCleaner
from rupsycho.parsers.judges import MultipleChoiceJudge
from rupsycho.parsers.validators import ValidatorParser
from rupsycho.postprocessing import PostprocessingPipeline

experiment = rup.experiment_from_file("bfi_demo_config.json")
experiment.run(callbacks=[CSVCallback("results.csv")])

pipeline = PostprocessingPipeline(
    config_file_path="bfi_demo_config.json",
    results_file_patterns=["results.csv"],
    cleaner=BasicCleaner(),
    validator=ValidatorParser(),
    judge=MultipleChoiceJudge(
        ["1. Disagree strongly", "2. Disagree a little", "3. Neither agree nor disagree",
         "4. Agree a little", "5. Agree strongly"]
    ),
    output_path="processed_results.csv",
)
pipeline.run()
```

- `config_file_path` is the configuration of the experiment; it provides the answer options of
  each item.
- `results_file_patterns` is a list of glob patterns of CSV files written by `CSVCallback`
  (the pipeline needs their `answer` and `instruction_item_id` columns). All matching files
  are combined; if no file matches, `run()` fails with `ValueError: No objects to concatenate`.
  Rows with an empty `answer`, which is what a failed model call leaves behind, make the pipeline
  fail, so remove them from the CSV first.
- The judge receives the answer options of the item. Items that use the questionnaire's
  `default_answer_options` pass an empty list, in which case the judge falls back to the
  options it was created with.

The output file contains the raw results plus three columns:

| Column              | Content                                                                          |
| ------------------- | -------------------------------------------------------------------------------- |
| `cleaned_answer`    | The cleaner's result                                                             |
| `validation_status` | The validator's complete result, for `ValidatorParser` the dictionary with `text`, `validation_status` and `details`, stored as text in the CSV |
| `decision`          | The judge's decision: an answer option, `not present` or `inconclusive`          |

Use `ast.literal_eval` to turn the `validation_status` column back into dictionaries.
