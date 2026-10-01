# Parsing & Postprocessing

Raw model answers are free text. The `rupsycho.parsers` package provides LangChain output
parsers that you can use on their own (`parser.invoke(text)`), inside a chain, or via the
[`PostprocessingPipeline`][rupsycho.postprocessing.PostprocessingPipeline].

## Cleaners

```python
from rupsycho.parsers.cleaners import BasicCleaner, PromptRemovalCleaner, RegexExtractorCleaner

BasicCleaner().invoke("Hello,\nworld 😊")                      # 'Hello, world'
PromptRemovalCleaner(prompt="Once upon a time").invoke(text)   # drop an echoed prompt
RegexExtractorCleaner(pattern=r'"answer":\s*"?([^"]*?)"?\s*}').invoke('{"answer": "4"}')
```

## Validators

Validators return the text together with a `validation_status`.

```python
from rupsycho.parsers.validators import ValidatorParser

ValidatorParser().invoke("As an AI, I cannot answer that.")
# {'validation_status': 'invalid', ...}
```

`ApologiesValidatorParser`, `BeingAiValidatorParser` and `RefusalValidatorParser` check one
pattern each; `ValidatorParser` combines them. `ModelBasedValidator` uses a
classifier from the Hugging Face hub (for example
`protectai/distilroberta-base-rejection-v1`).

## Judges

Judges select the answer option that best matches the text.

```python
from rupsycho.parsers.judges import MultipleChoiceJudge

judge = MultipleChoiceJudge(["1. Disagree", "2. Neutral", "3. Agree"])
judge.invoke("I would choose option 3 because I agree.")   # '3. Agree'
```

`ModelBasedAnswerJudge` does the same with a fine-tuned classifier, and `DemographicsJudge`
extracts demographic information such as age or gender.

## The pipeline

```python
from rupsycho.postprocessing import PostprocessingPipeline
from rupsycho.callbacks import CSVCallback

experiment.run(callbacks=[CSVCallback("results.csv")])

pipeline = PostprocessingPipeline(
    config_file_path="config.json",
    results_file_patterns=["results.csv"],
    cleaner=cleaner,
    validator=validator,
    judge=judge,
    output_path="processed_results.csv",
)
pipeline.run()
```

The output adds the columns `cleaned_answer`, `validation_status` and `decision` to the
raw results.
