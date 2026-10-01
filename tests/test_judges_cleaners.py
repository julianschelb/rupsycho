"""Unit tests for the output parsers: cleaners, validators and judges.

Everything runs offline. The model-based classes are tested twice:

* with tiny stand-ins for the ``transformers`` factories (``monkeypatch``), which pins how the
  classes call the library and makes every probability exactly known, and
* with real but tiny, randomly initialised models written to ``tmp_path``, which proves that the
  real ``from_pretrained`` / pipeline / tokenizer code paths work without downloading anything.

Tests marked ``xfail(strict=True)`` describe the *correct* behaviour of a known defect: they fail
today and turn into a loud XPASS as soon as the defect is fixed, at which point the marker has to
be removed.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import ValidationError

import rupsycho.parser as legacy_parser_module
import rupsycho.parsers as parsers_package
import rupsycho.parsers.judges as judges_module
import rupsycho.parsers.validators as validators_module
from rupsycho.parsers.cleaners import BasicCleaner, PromptRemovalCleaner, RegexExtractorCleaner
from rupsycho.parsers.judges import DemographicsJudge, ModelBasedAnswerJudge, MultipleChoiceJudge
from rupsycho.parsers.parser import BasicParser
from rupsycho.parsers.validators import (
    APOLOGIES_HINTS,
    BEING_AI_HINTS,
    CATCH_ALL_HINTS,
    REFUSAL_HINTS,
    ApologiesValidatorParser,
    BeingAiValidatorParser,
    ModelBasedValidator,
    RefusalValidatorParser,
    ValidatorParser,
    normalize,
)

from .test_parser_utils import fake_num2words  # noqa: F401  (fixture, used via usefixtures)

# ===========================================================================
# package surface
# ===========================================================================


@pytest.mark.parametrize(
    "name",
    [
        "BasicCleaner",
        "PromptRemovalCleaner",
        "RegexExtractorCleaner",
        "MultipleChoiceJudge",
        "DemographicsJudge",
        "ModelBasedAnswerJudge",
        "ApologiesValidatorParser",
        "BeingAiValidatorParser",
        "RefusalValidatorParser",
        "ValidatorParser",
        "ModelBasedValidator",
        "BasicParser",
    ],
)
def test_parsers_package_exports_every_parser(name):
    assert hasattr(parsers_package, name)


def test_parsers_package_exports_the_very_same_classes():
    assert parsers_package.BasicCleaner is BasicCleaner
    assert parsers_package.MultipleChoiceJudge is MultipleChoiceJudge
    assert parsers_package.ValidatorParser is ValidatorParser
    assert parsers_package.BasicParser is BasicParser


def test_legacy_parser_module_is_a_thin_alias():
    assert legacy_parser_module.BasicParser is BasicParser
    assert legacy_parser_module.__all__ == ["BasicParser"]
    assert not hasattr(legacy_parser_module, "re")  # no longer a verbatim copy of the module


@pytest.mark.parametrize(
    ("parser", "expected"),
    [
        (BasicCleaner(), "basic_parser"),
        (BasicParser(), "basic_parser"),
        (PromptRemovalCleaner(prompt="p"), "prompt_removal_cleaner_parser"),
        (RegexExtractorCleaner(pattern="(x)"), "regex_extractor_cleaner"),
        (ApologiesValidatorParser(), "apologies_validator_parser"),
        (BeingAiValidatorParser(), "being_ai_validator_parser"),
        (RefusalValidatorParser(), "refusal_validator_parser"),
        (ValidatorParser(), "validator_parser"),
        (MultipleChoiceJudge(["1. a"]), "multiple_choice_parser"),
        (DemographicsJudge(), "demographic_parser"),
    ],
)
def test_parser_type_identifiers(parser, expected):
    assert parser._type == expected


# ===========================================================================
# BasicCleaner / BasicParser
# ===========================================================================

BASIC_CLEANER_CASES = [
    pytest.param("Hello,\nworld \U0001f60a", "Hello, world", id="emoji-and-line-break"),
    pytest.param("", "", id="empty"),
    pytest.param("   ", "", id="only-spaces"),
    pytest.param("\n\t \r\n", "", id="only-whitespace-characters"),
    pytest.param(" a  b \t c ", "a b c", id="whitespace-is-collapsed"),
    pytest.param("line1\r\nline2", "line1 line2", id="windows-line-break"),
    pytest.param("x\n\n\ny", "x y", id="several-line-breaks"),
    pytest.param("ASCII only.", "ASCII only.", id="ascii-is-untouched"),
    pytest.param("3", "3", id="number"),
    pytest.param('{"answer": "3"}', '{"answer": "3"}', id="json-is-untouched"),
    pytest.param("\U0001f60a", "", id="only-emoji"),
    pytest.param("emoji only \U0001f60a\U0001f60a", "emoji only", id="trailing-emoji"),
    pytest.param("日本語", "", id="non-latin-script"),
    pytest.param("naïve café", "nave caf", id="accented-letters-are-dropped"),
    pytest.param("a​b", "ab", id="zero-width-space"),
    pytest.param("word " * 5000, ("word " * 5000).strip(), id="very-long"),
]


@pytest.mark.parametrize("parser_class", [BasicCleaner, BasicParser])
@pytest.mark.parametrize(("text", "expected"), BASIC_CLEANER_CASES)
def test_basic_cleaner_and_basic_parser(parser_class, text, expected):
    assert parser_class().parse(text) == expected


def test_basic_cleaner_works_on_chat_messages():
    assert BasicCleaner().invoke(AIMessage(content="Hello\nworld \U0001f60a")) == "Hello world"


def test_basic_cleaner_is_idempotent():
    cleaner = BasicCleaner()
    once = cleaner.parse("  Mixed \n content \U0001f60a here\r\n")
    assert cleaner.parse(once) == once


@pytest.mark.parametrize("parser_class", [BasicCleaner, BasicParser])
@pytest.mark.parametrize(
    "text",
    [
        "plain",
        "\ufeffbyte order mark",
        "e\u0301 combining accent",
        "left\u200fto\u200eright marks",
        "family \U0001f468\u200d\U0001f469\u200d\U0001f467",
        "tabs\tand\x0bvertical\x0cform feeds",
        "\n\n  leading and trailing  \r\n\r\n",
        "mixed \u00e4\u00f6\u00fc \u65e5\u672c \U0001f60a\nlines",
    ],
)
def test_basic_cleaner_output_is_single_line_ascii(parser_class, text):
    result = parser_class().parse(text)

    assert result.isascii()
    assert "\n" not in result
    assert "\r" not in result
    assert "  " not in result
    assert result == result.strip()


# ===========================================================================
# PromptRemovalCleaner
# ===========================================================================


def test_prompt_removal_cleaner_fields():
    default = PromptRemovalCleaner(prompt="Once upon a time")
    assert (default.prompt, default.similarity_threshold, default.fast) == (
        "Once upon a time",
        0.7,
        True,
    )

    custom = PromptRemovalCleaner(prompt="x", similarity_threshold=0.9, fast=False)
    assert (custom.prompt, custom.similarity_threshold, custom.fast) == ("x", 0.9, False)


def test_prompt_removal_cleaner_prompt_is_required():
    with pytest.raises(TypeError):
        PromptRemovalCleaner()  # type: ignore[call-arg]


@pytest.mark.xfail(
    strict=True,
    reason="PromptRemovalCleaner.parse returns prompt_cleaner's dict instead of the cleaned string",
)
def test_prompt_removal_cleaner_returns_the_cleaned_completion_as_a_string():
    cleaner = PromptRemovalCleaner(prompt="Question: what? Answer:", similarity_threshold=0.5)

    assert cleaner.parse("Question: what? Answer: 3") == "3"


@pytest.mark.xfail(
    strict=True,
    reason="PromptRemovalCleaner.parse returns prompt_cleaner's dict instead of the cleaned string",
)
def test_prompt_removal_cleaner_can_be_chained_with_other_parsers():
    chain = PromptRemovalCleaner(prompt="Question: what? Answer:") | BasicCleaner()

    assert chain.invoke("Question: what? Answer: 3\n") == "3"


# ===========================================================================
# RegexExtractorCleaner
# ===========================================================================

ANSWER_PATTERN = r'"answer":\s*"?([^"]*?)"?\s*}'


@pytest.mark.parametrize(
    ("pattern", "text", "expected"),
    [
        pytest.param(ANSWER_PATTERN, '{"answer": "4"}', "4", id="quoted-value"),
        pytest.param(ANSWER_PATTERN, '{"answer": 4}', "4", id="unquoted-value"),
        pytest.param(
            ANSWER_PATTERN, '{"answer":"4. Agree a little"}', "4. Agree a little", id="text"
        ),
        pytest.param(
            ANSWER_PATTERN, 'x {"answer": "a"} and {"answer": "b"}', "a", id="first-match"
        ),
        pytest.param(ANSWER_PATTERN, '{"answer": ""}', "", id="empty-value"),
        pytest.param(ANSWER_PATTERN, "no json here", "no json here", id="no-match-keeps-the-text"),
        pytest.param(ANSWER_PATTERN, '{"answer": "x"', '{"answer": "x"', id="truncated-json"),
        pytest.param(ANSWER_PATTERN, "", "", id="empty-text"),
        pytest.param(r"(\d+)", "abc", "abc", id="digits-no-match"),
        pytest.param(r"(\d+)", "a 12 b 34", "12", id="digits-first-match"),
        pytest.param(r"(\d+)-(\d+)", "a 12-34", "12", id="only-group-one-is-returned"),
        pytest.param(r"answer:\s*(.+)", "ok\nanswer: 4\nnext", "4", id="multi-line-text"),
        pytest.param(r"(?i)answer:\s*(\w+)", "ANSWER: yes", "yes", id="inline-flags"),
        pytest.param(r"(\w+)", "\U0001f60a x", "x", id="emoji-before-the-match"),
    ],
)
def test_regex_extractor_cleaner(pattern, text, expected):
    assert RegexExtractorCleaner(pattern=pattern).parse(text) == expected


def test_regex_extractor_cleaner_stores_its_pattern():
    assert RegexExtractorCleaner(pattern=r"(\d)").pattern == r"(\d)"


def test_regex_extractor_cleaner_pattern_is_required():
    with pytest.raises(TypeError):
        RegexExtractorCleaner()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "pattern", [r"\d+", r"(["], ids=["pattern-without-a-group", "invalid-pattern"]
)
def test_regex_extractor_cleaner_wraps_regex_errors(pattern):
    with pytest.raises(OutputParserException, match="RegexExtractorCleaner encountered an error"):
        RegexExtractorCleaner(pattern=pattern).parse("a 12")


def test_regex_extractor_cleaner_in_a_chain():
    chain = RegexExtractorCleaner(pattern=ANSWER_PATTERN) | BasicCleaner()

    assert chain.invoke('{"answer": "4 \U0001f60a"}') == "4"


# ===========================================================================
# every parser wraps failures and chains the original exception
# ===========================================================================


@pytest.mark.parametrize(
    ("parser", "bad_input", "cause"),
    [
        pytest.param(BasicCleaner(), None, AttributeError, id="basic-cleaner"),
        pytest.param(BasicParser(), 3, AttributeError, id="basic-parser"),
        pytest.param(PromptRemovalCleaner(prompt="p"), None, AttributeError, id="prompt-removal"),
        pytest.param(RegexExtractorCleaner(pattern=r"(\d)"), None, TypeError, id="regex-extractor"),
        pytest.param(ApologiesValidatorParser(), None, TypeError, id="apologies"),
        pytest.param(BeingAiValidatorParser(), None, TypeError, id="being-ai"),
        pytest.param(RefusalValidatorParser(), None, TypeError, id="refusal"),
        pytest.param(ValidatorParser(), None, OutputParserException, id="validator"),
        pytest.param(MultipleChoiceJudge(["1. a"]), None, AttributeError, id="multiple-choice"),
    ],
)
def test_parse_errors_are_wrapped_and_chained(parser, bad_input, cause):
    with pytest.raises(OutputParserException, match="encountered an error") as excinfo:
        parser.parse(bad_input)

    assert isinstance(excinfo.value.__cause__, cause)


# ===========================================================================
# validators
# ===========================================================================


def test_normalize_strips_leading_whitespace_and_lower_cases():
    assert normalize("  \n\tHello World  ") == "hello world  "
    assert normalize("ÄÖ  X") == "äö  x"
    assert normalize("") == ""
    assert normalize("   ") == ""


APOLOGIES_FLAGGED = [
    "Sorry, I can't",
    "sorry",
    "  SORRY",
    "\n\nI'm sorry",
    "I am sorry",
    "I apologize",
    "I apologise",
    "Apologies for that",
    "My apologies for the confusion",
]
APOLOGIES_FINE = ["3", "", "   ", "Not sorry", "Well, I'm sorry", "I choose 3. Sorry!", "sor"]

BEING_AI_FLAGGED = [
    "As an AI language model",
    "as an artificial intelligence",
    "I am an AI",
    "I'm an AI",
    "I'm just an AI",
    "As a neutral entity",
    "I'm an artificial being",
    "I am an artificial intelligence",
    "1. I am an artificial intelligence",
    "1. I'm an artificial",
    "I am an ai.",
    "  \nAS AN AI",
]
BEING_AI_FINE = ["3", "", "   ", "Hello, I am an AI assistant", "I am a person", "an AI said so"]

REFUSAL_FLAGGED = [
    "No, I do not",
    "No. I do not",
    "I cannot",
    "I can't",
    "I'm not able to",
    "I am not able to",
    "It is not appropriate",
    "It's not appropriate",
    "It is not accurate",
    "It's not accurate",
    "I am not sure that it's appropriate",
    "I'm not sure that it's appropriate",
    "It is not ethical",
    "It's not ethical",
    "As an AI language model",
    "I am an AI language model",
    "I do not have",
    "I don't have feelings",
    "I would need more context",
    "1. I cannot",
    "1. I don't have",
    "My programming does not allow",
    "I would like to clarify that",
    "Hello. I do not have a personal opinion",
    "Well, I do not have personal opinions on this",
    "Hi. I am an AI assistant.",
    "I am an artificial intelligence",
]
REFUSAL_FINE = ["3", "", "   ", "No I do not", "Nope", "Answer: I cannot", "I choose 3"]


@pytest.mark.parametrize(
    ("parser", "flagged", "fine", "detail"),
    [
        pytest.param(
            ApologiesValidatorParser(),
            APOLOGIES_FLAGGED,
            APOLOGIES_FINE,
            "apologies",
            id="apologies",
        ),
        pytest.param(
            BeingAiValidatorParser(), BEING_AI_FLAGGED, BEING_AI_FINE, "being_ai", id="being-ai"
        ),
        pytest.param(
            RefusalValidatorParser(), REFUSAL_FLAGGED, REFUSAL_FINE, "refusal", id="refusal"
        ),
    ],
)
class TestRuleBasedValidators:
    def test_flagged_texts_are_invalid(self, parser, flagged, fine, detail):
        for text in flagged:
            assert parser.parse(text) == {
                "text": text,
                "validation_status": "invalid",
                "details": {detail: True},
            }, text

    def test_other_texts_are_valid(self, parser, flagged, fine, detail):
        for text in fine:
            assert parser.parse(text) == {
                "text": text,
                "validation_status": "valid",
                "details": {detail: False},
            }, text

    def test_the_text_is_returned_unmodified(self, parser, flagged, fine, detail):
        text = "  \n  SoRrY, I As An Ai   \t "
        assert parser.parse(text)["text"] == text

    def test_invoke_returns_the_same_result(self, parser, flagged, fine, detail):
        assert parser.invoke(flagged[0]) == parser.parse(flagged[0])


@pytest.mark.parametrize("hint", APOLOGIES_HINTS)
def test_every_apology_hint_is_detected(hint):
    for text in (hint, hint.upper(), f"  {hint} and more"):
        assert ApologiesValidatorParser().parse(text)["validation_status"] == "invalid"


@pytest.mark.parametrize("hint", BEING_AI_HINTS)
def test_every_being_ai_hint_is_detected(hint):
    for text in (hint, hint.upper(), f"  {hint} and more"):
        assert BeingAiValidatorParser().parse(text)["validation_status"] == "invalid"


@pytest.mark.parametrize("hint", REFUSAL_HINTS)
def test_every_refusal_hint_is_detected(hint):
    for text in (hint, hint.upper(), f"  {hint} and more"):
        assert RefusalValidatorParser().parse(text)["validation_status"] == "invalid"


@pytest.mark.parametrize("hint", CATCH_ALL_HINTS)
def test_catch_all_hints_are_detected_anywhere_in_the_text(hint):
    # unlike the other hints these do not have to start the answer
    for text in (f"Well, {hint} thanks", f"WELL, {hint.upper()} THANKS"):
        assert RefusalValidatorParser().parse(text)["validation_status"] == "invalid"


@pytest.mark.parametrize(
    ("text", "apologies", "being_ai", "refusal"),
    [
        pytest.param("I choose 3", False, False, False, id="plain-answer"),
        pytest.param("3", False, False, False, id="number"),
        pytest.param("", False, False, False, id="empty"),
        pytest.param("   \n", False, False, False, id="whitespace"),
        pytest.param("Sorry, as an AI I cannot answer", True, False, False, id="sorry-first"),
        pytest.param("I'm sorry, but I can't help with that.", True, False, False, id="im-sorry"),
        pytest.param("I cannot help", False, False, True, id="refusal-only"),
        pytest.param(
            "As an AI language model, I do not have opinions.", False, True, True, id="ai"
        ),
        pytest.param("As an AI, I choose 3", False, True, True, id="ai-but-answers"),
        pytest.param("Hello, I am an AI assistant", False, False, True, id="catch-all"),
        pytest.param(
            "I apologize for any confusion, but as an AI, I cannot provide that information.",
            True,
            False,
            False,
            id="long-apology",
        ),
        pytest.param("  \nSORRY  ", True, False, False, id="shouting-with-padding"),
    ],
)
def test_validator_parser_combines_the_three_checks(text, apologies, being_ai, refusal):
    result = ValidatorParser().parse(text)

    assert result == {
        "text": text,
        "validation_status": "invalid" if (apologies or being_ai or refusal) else "valid",
        "details": {"apologies": apologies, "being_ai": being_ai, "refusal": refusal},
    }


def test_validator_parser_uses_the_configured_sub_validators():
    class AlwaysFlagging(ApologiesValidatorParser):
        def parse(self, text):
            return {"text": text, "validation_status": "invalid", "details": {"apologies": True}}

    result = ValidatorParser(apologies_parser=AlwaysFlagging()).parse("3")

    assert result["validation_status"] == "invalid"
    assert result["details"] == {"apologies": True, "being_ai": False, "refusal": False}


@pytest.mark.xfail(
    strict=True,
    reason="hint lists only contain the ASCII apostrophe, so 'I’m sorry' / 'I can’t' pass",
)
@pytest.mark.parametrize(
    ("parser", "text"),
    [
        pytest.param(ApologiesValidatorParser(), "I’m sorry, but no.", id="apologies"),
        pytest.param(RefusalValidatorParser(), "I can’t help with that.", id="refusal"),
        pytest.param(BeingAiValidatorParser(), "I’m an AI.", id="being-ai"),
        pytest.param(ValidatorParser(), "I’m sorry, but I can’t help.", id="combined"),
    ],
)
def test_typographic_apostrophes_do_not_hide_refusals(parser, text):
    assert parser.parse(text)["validation_status"] == "invalid"


# ===========================================================================
# ModelBasedValidator
# ===========================================================================


@pytest.fixture
def fake_hub(monkeypatch):
    """Replace the ``transformers`` factories used by ModelBasedValidator with recording fakes."""
    hub = SimpleNamespace(
        tokenizers=[], models=[], pipelines=[], texts=[], output=[{"label": "NORMAL", "score": 0.9}]
    )

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(name):
            hub.tokenizers.append(name)
            return ("tokenizer", name)

    class FakeModel:
        @staticmethod
        def from_pretrained(name):
            hub.models.append(name)
            return ("model", name)

    def fake_pipeline(task, **kwargs):
        hub.pipelines.append((task, kwargs))

        def classifier(text):
            hub.texts.append(text)
            if isinstance(hub.output, Exception):
                raise hub.output
            return hub.output

        return classifier

    monkeypatch.setattr(validators_module, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(validators_module, "AutoModelForSequenceClassification", FakeModel)
    monkeypatch.setattr(validators_module, "pipeline", fake_pipeline)
    return hub


# how transformers spells "run on the CPU" / "run on the first GPU" (int index or device string)
CPU_DEVICES = (-1, "cpu")
FIRST_GPU = (0, "cuda", "cuda:0")


def test_model_based_validator_builds_a_text_classification_pipeline(fake_hub):
    validator = ModelBasedValidator(model_name="org/rejections", device="cpu")

    assert fake_hub.tokenizers == ["org/rejections"]
    assert fake_hub.models == ["org/rejections"]
    [(task, kwargs)] = fake_hub.pipelines
    assert task == "text-classification"
    assert kwargs.pop("device") in CPU_DEVICES
    assert kwargs == {
        "model": ("model", "org/rejections"),
        "tokenizer": ("tokenizer", "org/rejections"),
        "truncation": True,
        "max_length": 512,
    }
    assert (validator.model_name, validator.device) == ("org/rejections", "cpu")


def test_model_based_validator_defaults(fake_hub):
    validator = ModelBasedValidator()

    assert fake_hub.tokenizers == ["ProtectAI/distilroberta-base-rejection-v1"]
    assert validator.model_name == "ProtectAI/distilroberta-base-rejection-v1"
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert validator.device == expected_device
    expected_pipeline_devices = FIRST_GPU if expected_device == "cuda" else CPU_DEVICES
    assert fake_hub.pipelines[0][1]["device"] in expected_pipeline_devices


def test_model_based_validator_uses_the_first_gpu_for_cuda(fake_hub):
    ModelBasedValidator(model_name="m", device="cuda")

    assert fake_hub.pipelines[0][1]["device"] in FIRST_GPU


@pytest.mark.xfail(
    strict=True,
    reason="ModelBasedValidator maps every device except 'cuda' (e.g. 'cuda:1') to the CPU",
)
def test_model_based_validator_honours_a_cuda_device_index(fake_hub):
    ModelBasedValidator(model_name="m", device="cuda:1")

    assert fake_hub.pipelines[0][1]["device"] not in CPU_DEVICES


@pytest.mark.parametrize(
    ("label", "status"),
    [("REJECTION", "invalid"), ("NORMAL", "valid"), ("LABEL_0", "valid"), ("rejection", "valid")],
)
def test_model_based_validator_maps_the_label_to_a_status(fake_hub, label, status):
    fake_hub.output = [{"label": label, "score": 0.75}]

    result = ModelBasedValidator(model_name="m", device="cpu").parse("I cannot help")

    assert result == {
        "text": "I cannot help",
        "validation_status": status,
        "confidence_score": 0.75,
    }
    assert fake_hub.texts == ["I cannot help"]


def test_model_based_validator_wraps_classifier_errors(fake_hub):
    fake_hub.output = RuntimeError("out of memory")
    validator = ModelBasedValidator(model_name="m", device="cpu")

    with pytest.raises(
        OutputParserException, match="ModelBasedValidator encountered an error"
    ) as e:
        validator.parse("text")

    assert isinstance(e.value.__cause__, RuntimeError)


def test_model_based_validator_type(fake_hub):
    assert ModelBasedValidator(model_name="m", device="cpu")._type == "model_based_validator_parser"


def _tiny_text_classifier(directory, *, rejection: bool):
    """Write a tiny DistilBERT classifier that always predicts REJECTION (or NORMAL)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        PreTrainedTokenizerFast,
    )

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "sorry": 4, "i": 5, "choose": 6}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=512,
        model_input_names=["input_ids", "attention_mask"],
    )
    config = DistilBertConfig(
        vocab_size=len(vocab),
        dim=8,
        n_layers=1,
        n_heads=2,
        hidden_dim=16,
        max_position_embeddings=512,
        num_labels=2,
        id2label={0: "NORMAL", 1: "REJECTION"},
        label2id={"NORMAL": 0, "REJECTION": 1},
    )
    model = DistilBertForSequenceClassification(config)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([-5.0, 5.0] if rejection else [5.0, -5.0]))
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return str(directory)


@pytest.mark.parametrize(
    ("rejection", "status"), [(True, "invalid"), (False, "valid")], ids=["rejection", "normal"]
)
def test_model_based_validator_with_a_tiny_real_model(tmp_path, rejection, status):
    path = _tiny_text_classifier(tmp_path / "classifier", rejection=rejection)
    validator = ModelBasedValidator(model_name=path, device="cpu")

    result = validator.parse("Sorry, I choose nothing")

    assert result["text"] == "Sorry, I choose nothing"
    assert result["validation_status"] == status
    assert result["confidence_score"] == pytest.approx(0.99995, abs=1e-4)
    assert validator.invoke("I choose") == validator.parse("I choose")


def test_model_based_validator_truncates_texts_longer_than_the_model_window(tmp_path):
    path = _tiny_text_classifier(tmp_path / "classifier", rejection=True)
    validator = ModelBasedValidator(model_name=path, device="cpu")

    # 5000 tokens do not fit into the 512 position embeddings: this only works with truncation
    result = validator.parse("sorry " * 5000)

    assert result["validation_status"] == "invalid"


# ===========================================================================
# MultipleChoiceJudge
# ===========================================================================

BFI_OPTIONS = [
    "1. Disagree strongly",
    "2. Disagree a little",
    "3. Neither agree nor disagree",
    "4. Agree a little",
    "5. Agree strongly",
]
LIKERT = ["1. Strongly disagree", "2. Disagree", "3. Neutral", "4. Agree", "5. Strongly agree"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("3", BFI_OPTIONS[2], id="number"),
        pytest.param("I choose 3", BFI_OPTIONS[2], id="number-in-a-sentence"),
        pytest.param("Answer: 1", BFI_OPTIONS[0], id="answer-label"),
        pytest.param("5", BFI_OPTIONS[4], id="five"),
        pytest.param("Agree strongly", BFI_OPTIONS[4], id="option-text"),
        pytest.param("agree strongly", BFI_OPTIONS[4], id="option-text-lower-case"),
        pytest.param("4. Agree a little", BFI_OPTIONS[3], id="number-and-text"),
        pytest.param(
            "I strongly agree - 5. Agree strongly", BFI_OPTIONS[4], id="sentence-plus-option"
        ),
        pytest.param("Neither agree nor disagree", BFI_OPTIONS[2], id="neither"),
        pytest.param("Disagree a little", BFI_OPTIONS[1], id="disagree-a-little"),
        pytest.param("I choose 3 \U0001f60a", BFI_OPTIONS[2], id="emoji"),
        pytest.param("(5)", BFI_OPTIONS[4], id="parentheses"),
        pytest.param("3 3 1", BFI_OPTIONS[2], id="most-frequent-number-wins"),
        pytest.param("1 or 2", "inconclusive", id="two-numbers-tie"),
        pytest.param("3 and agree strongly", "inconclusive", id="number-and-text-tie"),
        pytest.param("", "not present", id="empty"),
        pytest.param("   \n", "not present", id="whitespace"),
        pytest.param("I have no idea", "not present", id="nothing-matches"),
        pytest.param("I choose 11", "not present", id="eleven-is-not-an-option"),
        pytest.param("\U0001f60a", "not present", id="only-emoji"),
    ],
)
def test_multiple_choice_judge_bfi(text, expected):
    assert MultipleChoiceJudge(BFI_OPTIONS).parse(text) == expected


@pytest.mark.parametrize(
    ("text", "ignore_case", "expected"),
    [
        pytest.param("agree", True, LIKERT[3], id="insensitive-agree"),
        pytest.param("Agree", False, LIKERT[3], id="sensitive-agree"),
        pytest.param("agree", False, "not present", id="sensitive-wrong-case"),
        pytest.param("NEUTRAL", True, LIKERT[2], id="insensitive-shouting"),
        pytest.param("NEUTRAL", False, "not present", id="sensitive-shouting"),
        pytest.param("Neutral.", False, LIKERT[2], id="sensitive-with-period"),
        pytest.param("Strongly agree", False, LIKERT[4], id="sensitive-longest-option"),
        pytest.param("4", False, LIKERT[3], id="numbers-have-no-case"),
        pytest.param("Disagree", True, LIKERT[1], id="disagree"),
    ],
)
def test_multiple_choice_judge_ignore_case(text, ignore_case, expected):
    judge = MultipleChoiceJudge(LIKERT, ignore_case=ignore_case)

    assert judge.parse(text) == expected


@pytest.mark.xfail(
    strict=True,
    reason="an option whose text is contained in another option's text ties with it: 'inconclusive'",
)
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("Strongly agree", LIKERT[4], id="strongly-agree"),
        pytest.param("Strongly disagree", LIKERT[0], id="strongly-disagree"),
        pytest.param("I strongly agree", LIKERT[4], id="sentence"),
    ],
)
def test_multiple_choice_judge_prefers_the_more_specific_option(text, expected):
    assert MultipleChoiceJudge(LIKERT).parse(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("B", "B. Sometimes"),
        ("b", "B. Sometimes"),
        ("Answer: B", "B. Sometimes"),
        ("C. Always", "C. Always"),
        ("never", "A. Never"),
        ("Z", "not present"),
    ],
)
def test_multiple_choice_judge_letter_options(text, expected):
    assert MultipleChoiceJudge(["A. Never", "B. Sometimes", "C. Always"]).parse(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("1", "1. Never"), ("11", "11. Always"), ("1 and 11", "inconclusive"), ("111", "not present")],
)
def test_multiple_choice_judge_numbers_that_contain_each_other(text, expected):
    assert MultipleChoiceJudge(["1. Never", "11. Always"]).parse(text) == expected


def test_multiple_choice_judge_runtime_options_override_the_configured_ones():
    judge = MultipleChoiceJudge(["1. a", "2. b"])

    assert judge.parse("B", ["A", "B"]) == "B"
    assert judge.parse("2") == "2. b"


@pytest.mark.parametrize("override", [None, []], ids=["none", "empty-list"])
def test_multiple_choice_judge_falls_back_to_the_configured_options(override):
    assert MultipleChoiceJudge(["1. a", "2. b"]).parse("2", override) == "2. b"


def test_multiple_choice_judge_needs_some_options():
    with pytest.raises(OutputParserException, match="No possible answers provided") as excinfo:
        MultipleChoiceJudge([]).parse("anything")

    assert isinstance(excinfo.value.__cause__, ValueError)


def test_multiple_choice_judge_rejects_a_string_as_the_list_of_options():
    with pytest.raises(ValidationError):
        MultipleChoiceJudge("1. a")  # type: ignore[arg-type]


def test_multiple_choice_judge_accepts_langchain_arguments():
    judge = MultipleChoiceJudge(["1. a"], name="my-judge")

    assert judge.name == "my-judge"
    assert judge.ignore_case is True


def test_multiple_choice_judge_in_a_chain_and_batch():
    judge = MultipleChoiceJudge(BFI_OPTIONS)
    chain = StrOutputParser() | BasicCleaner() | judge

    assert chain.invoke("\U0001f60a I choose\n3 ") == BFI_OPTIONS[2]
    assert judge.batch(["1", "x", "1 2"]) == [BFI_OPTIONS[0], "not present", "inconclusive"]
    assert judge.invoke(AIMessage(content="5")) == BFI_OPTIONS[4]


@pytest.mark.parametrize(
    "text", ["3", "agree strongly", "5: agree strongly", "1 or 2", "nothing", "", "I choose 3."]
)
def test_multiple_choice_judge_does_not_depend_on_the_order_of_the_options(text):
    forward = MultipleChoiceJudge(BFI_OPTIONS).parse(text)
    backward = MultipleChoiceJudge(BFI_OPTIONS[::-1]).parse(text)

    assert forward == backward


def test_multiple_choice_judge_handles_very_long_answers():
    text = "lorem ipsum dolor sit amet. " * 20_000 + "I pick 5" + " consectetur" * 20_000

    assert MultipleChoiceJudge(BFI_OPTIONS).parse(text) == BFI_OPTIONS[4]


# ===========================================================================
# DemographicsJudge
# ===========================================================================


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I am a man", "male"),
        ("I'm a woman", "female"),
        ("female", "female"),
        ("I identify as non-binary", "other"),
        ("He said she is a girl", "female"),
        ("man and woman", "inconclusive"),
        ("I prefer not to say", "not present"),
        ("", "not present"),
    ],
)
def test_demographics_judge_gender(text, expected):
    assert DemographicsJudge().parse(text, ["gender"]) == expected


@pytest.mark.usefixtures("fake_num2words")
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I am 25 years old", "25"),
        ("Twenty-five", "25"),
        ("I'm thirty.", "30"),
        ("0", "0"),
        ("I won't tell", "inconclusive"),
        ("", "inconclusive"),
        ("I am 150", "inconclusive"),
    ],
)
def test_demographics_judge_age(text, expected):
    assert DemographicsJudge().parse(text, ["age"]) == expected


@pytest.mark.usefixtures("fake_num2words")
def test_demographics_judge_max_age_is_configurable():
    assert DemographicsJudge(max_age=150).parse("I am 120", ["age"]) == "120"
    assert DemographicsJudge(max_age=30).parse("I am 45", ["age"]) == "inconclusive"


@pytest.mark.usefixtures("fake_num2words")
def test_demographics_judge_ignore_case_is_configurable():
    assert DemographicsJudge(ignore_case=False).parse("MALE", ["gender"]) == "not present"
    assert DemographicsJudge(ignore_case=False).parse("male", ["gender"]) == "male"
    assert DemographicsJudge(ignore_case=False).parse("Twenty", ["age"]) == "inconclusive"
    assert DemographicsJudge().parse("Twenty", ["age"]) == "20"


def test_demographics_judge_only_looks_at_the_first_answer_option():
    assert DemographicsJudge().parse("a man", ["gender", "age"]) == "male"


@pytest.mark.parametrize(
    ("options", "message", "cause"),
    [
        pytest.param(None, "No possible answer/question-type provided", ValueError, id="none"),
        pytest.param(["height"], "Unknown question type", ValueError, id="unknown-type"),
        pytest.param(["Gender"], "Unknown question type", ValueError, id="type-is-case-sensitive"),
        pytest.param(["1. Never"], "Unknown question type", ValueError, id="ordinary-options"),
    ],
)
def test_demographics_judge_rejects_unusable_question_types(options, message, cause):
    with pytest.raises(OutputParserException, match=message) as excinfo:
        DemographicsJudge().parse("a man", options)

    assert isinstance(excinfo.value.__cause__, cause)


@pytest.mark.xfail(
    strict=True,
    reason="DemographicsJudge only rejects None: an empty option list fails with 'list index out of range'",
)
def test_demographics_judge_empty_options_give_a_helpful_error():
    with pytest.raises(OutputParserException, match="No possible answer"):
        DemographicsJudge().parse("a man", [])


# ===========================================================================
# ModelBasedAnswerJudge
# ===========================================================================


@pytest.fixture
def fake_roberta(monkeypatch):
    """Replace the RoBERTa classes of the judge with fakes whose logits are looked up by option."""
    hub = SimpleNamespace(
        tokenizer_loads=[],
        model_loads=[],
        model_devices=[],
        batch_devices=[],
        tokenizer_calls=[],
        eval_calls=[],
        logits={},
        default_logits=(0.0, 0.0),
        failure=None,
    )

    class Batch(dict):
        def to(self, device):
            hub.batch_devices.append(device)
            return self

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name):
            hub.tokenizer_loads.append(name)
            return cls()

        def __call__(self, option, answer, **kwargs):
            hub.tokenizer_calls.append((option, answer, kwargs))
            return Batch(option=option)

    class FakeModel:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            hub.model_loads.append((name, kwargs))
            return cls()

        def to(self, device):
            hub.model_devices.append(device)
            return self

        def eval(self):
            hub.eval_calls.append(True)
            return self

        def __call__(self, option):
            if hub.failure is not None:
                raise hub.failure
            row = hub.logits.get(option, hub.default_logits)
            return SimpleNamespace(logits=torch.tensor([row], dtype=torch.float32))

    monkeypatch.setattr(judges_module, "RobertaTokenizer", FakeTokenizer)
    monkeypatch.setattr(judges_module, "RobertaForSequenceClassification", FakeModel)
    return hub


OPTIONS = ["1. never", "2.", "3. always"]


def make_judge(**kwargs):
    kwargs.setdefault("model_name", "org/judge")
    kwargs.setdefault("possible_answers", OPTIONS)
    kwargs.setdefault("device", "cpu")
    return ModelBasedAnswerJudge(**kwargs)


def test_model_based_answer_judge_loads_tokenizer_and_model(fake_roberta):
    judge = make_judge(device="cpu")

    assert fake_roberta.tokenizer_loads == ["org/judge"]
    assert fake_roberta.model_loads == [("org/judge", {"num_labels": 2})]
    assert fake_roberta.model_devices == ["cpu"]
    assert judge.tokenizer is not None
    assert judge.model is not None
    assert judge.entropy_threshold == 0.359
    assert judge.possible_answers == OPTIONS


def test_model_based_answer_judge_requires_a_model_name_and_answers(fake_roberta):
    with pytest.raises(ValidationError):
        ModelBasedAnswerJudge(possible_answers=OPTIONS)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        ModelBasedAnswerJudge(model_name="org/judge")  # type: ignore[call-arg]


@pytest.mark.skipif(torch.cuda.is_available(), reason="the default device is fine with a GPU")
@pytest.mark.xfail(
    strict=True, reason="ModelBasedAnswerJudge defaults to 'cuda:0' even when CUDA is unavailable"
)
def test_model_based_answer_judge_default_device_works_without_cuda(fake_roberta):
    ModelBasedAnswerJudge(model_name="org/judge", possible_answers=OPTIONS)

    assert fake_roberta.model_devices == ["cpu"]


def test_model_based_answer_judge_predict_answer(fake_roberta):
    fake_roberta.logits["1. never"] = (0.0, 3.0)
    fake_roberta.logits["3. always"] = (3.0, 0.0)
    judge = make_judge()

    never = judge.predict_answer("1. never", "I never do")
    always = judge.predict_answer("3. always", "I never do")

    assert never[0] == 1
    assert never[1] == pytest.approx(0.9525741, abs=1e-6)
    assert always[0] == 0
    assert always[1] == pytest.approx(0.0474259, abs=1e-6)
    assert fake_roberta.eval_calls == [True, True]
    assert fake_roberta.batch_devices == ["cpu", "cpu"]
    assert fake_roberta.tokenizer_calls[0] == (
        "1. never",
        "I never do",
        {"return_tensors": "pt", "padding": True, "truncation": True},
    )


def test_model_based_answer_judge_predict_for_all_options(fake_roberta):
    fake_roberta.logits["2."] = (0.0, 2.0)
    judge = make_judge()

    results = judge.predict_for_all_options("two")

    assert [r["answer_option"] for r in results] == OPTIONS
    assert [r["predicted_label"] for r in results] == [0, 1, 0]  # ties favour label 0
    assert results[1]["positive_probability"] == pytest.approx(0.8807971, abs=1e-6)
    assert results[0]["positive_probability"] == pytest.approx(0.5)
    assert set(results[0]) == {"answer_option", "predicted_label", "positive_probability"}


def test_model_based_answer_judge_explicit_options_override_the_configured_ones(fake_roberta):
    judge = make_judge()

    results = judge.predict_for_all_options("x", ["only this"])

    assert [r["answer_option"] for r in results] == ["only this"]


def test_model_based_answer_judge_needs_some_options(fake_roberta):
    judge = make_judge(possible_answers=[])

    with pytest.raises(ValueError, match="No possible answers provided"):
        judge.predict_for_all_options("x", [])


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        pytest.param([0.25, 0.25, 0.25, 0.25], 2.0, id="uniform-four"),
        pytest.param([0.5, 0.5], 1.0, id="uniform-two"),
        pytest.param([1.0, 0.0, 0.0], 0.0, id="one-hot"),
        pytest.param([0.9, 0.9, 0.9, 0.9], 2.0, id="scale-does-not-matter"),
        pytest.param([0.8, 0.1, 0.1], 0.9219280948873623, id="peaked"),
    ],
)
def test_model_based_answer_judge_entropy(fake_roberta, probabilities, expected):
    decisions = [{"positive_probability": p} for p in probabilities]

    assert make_judge().calculate_entropy(decisions) == pytest.approx(expected)


def test_model_based_answer_judge_entropy_treats_missing_probabilities_as_zero(fake_roberta):
    decisions = [{"positive_probability": 0.5}, {}, {"positive_probability": 0.5}]

    assert make_judge().calculate_entropy(decisions) == pytest.approx(1.0)


def test_model_based_answer_judge_picks_the_clear_winner(fake_roberta):
    fake_roberta.logits.update({"1. never": (3, -3), "2.": (-4, 4), "3. always": (3, -3)})

    assert make_judge().parse("I pick the middle one") == "2."


def test_model_based_answer_judge_is_inconclusive_when_nothing_stands_out(fake_roberta):
    assert make_judge().parse("hmm") == "inconclusive"  # all probabilities are 0.5


def test_model_based_answer_judge_threshold_decides_between_answer_and_inconclusive(fake_roberta):
    fake_roberta.logits.update({"1. never": (0, 1), "2.": (0, 2), "3. always": (0, 0)})
    judge = make_judge()
    entropy = judge.calculate_entropy(judge.predict_for_all_options("x"))

    assert make_judge(entropy_threshold=entropy - 1e-6).parse("x") == "inconclusive"
    assert make_judge(entropy_threshold=entropy + 1e-6).parse("x") == "2."


def test_model_based_answer_judge_ties_resolve_to_the_first_option(fake_roberta):
    assert make_judge(entropy_threshold=5.0).parse("hmm") == "1. never"


def test_model_based_answer_judge_runtime_options(fake_roberta):
    fake_roberta.logits["b"] = (-4, 4)
    fake_roberta.default_logits = (4, -4)

    assert make_judge().parse("x", ["a", "b", "c"]) == "b"


def test_model_based_answer_judge_wraps_model_errors(fake_roberta):
    fake_roberta.failure = RuntimeError("cuda out of memory")

    with pytest.raises(OutputParserException, match="ModelBasedAnswerJudge encountered") as e:
        make_judge().parse("x")

    assert isinstance(e.value.__cause__, RuntimeError)


def test_model_based_answer_judge_type(fake_roberta):
    assert make_judge()._type == "model_based_answer_parser"


@pytest.fixture
def tiny_judge_path(tmp_path):
    """A tiny RoBERTa sequence classifier whose head is zeroed: every probability is 0.5."""
    from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

    directory = tmp_path / "judge"
    directory.mkdir()
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4}
    for char in map(chr, range(32, 127)):
        vocab[char] = len(vocab)
    (directory / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (directory / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = RobertaTokenizer(
        str(directory / "vocab.json"), str(directory / "merges.txt"), model_max_length=512
    )
    config = RobertaConfig(
        vocab_size=len(vocab),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=514,
        num_labels=2,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )
    model = RobertaForSequenceClassification(config)
    with torch.no_grad():
        model.classifier.out_proj.weight.zero_()
        model.classifier.out_proj.bias.zero_()
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return str(directory)


def test_model_based_answer_judge_with_a_tiny_real_model(tiny_judge_path):
    judge = ModelBasedAnswerJudge(
        model_name=tiny_judge_path, possible_answers=OPTIONS, device="cpu"
    )

    label, probability = judge.predict_answer("1. never", "I choose 3")
    results = judge.predict_for_all_options("I choose 3")

    assert label in (0, 1)
    assert probability == pytest.approx(0.5)
    assert [r["answer_option"] for r in results] == OPTIONS
    assert all(r["positive_probability"] == pytest.approx(0.5) for r in results)
    assert judge.calculate_entropy(results) == pytest.approx(1.5849625, abs=1e-6)
    assert judge.parse("I choose 3") == "inconclusive"
    assert (
        ModelBasedAnswerJudge(
            model_name=tiny_judge_path,
            possible_answers=OPTIONS,
            device="cpu",
            entropy_threshold=2.0,
        ).parse("I choose 3")
        == "1. never"
    )
    assert not judge.model.training  # predict_answer switches the model to evaluation mode


def test_model_based_answer_judge_truncates_overlong_answers(tiny_judge_path):
    judge = ModelBasedAnswerJudge(
        model_name=tiny_judge_path, possible_answers=OPTIONS, device="cpu"
    )

    # ~3000 tokens do not fit into the 514 positions of the model: only works with truncation
    label, probability = judge.predict_answer("1. never", "very long answer " * 200)

    assert probability == pytest.approx(0.5)
