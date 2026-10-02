"""Naive reference implementations (oracles) for the property-based tests of rupsycho.

``test_properties_parsers``, ``test_properties_questionnaire`` and ``test_properties_run`` check
the library the way ``implicit-word-network`` checks its vectorised engine against a literal
implementation of the definitions: every function in this module re-implements one *documented*
behaviour as plainly as possible and shares no code with ``src/rupsycho``.

Two habits keep the oracles independent of the code they judge:

* explicit loops, ``str`` methods and hand-written scanners instead of the regular expressions
  the library uses, and
* building the expected output from the *structure* a test generated (an option has a number and
  a phrase, a prompt has a system part and a question) instead of parsing a rendered string back.

When library and oracle disagree, one of the two has a bug; the disagreement is the finding.
"""

from __future__ import annotations

import hashlib
import re
import string
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# =============================================================================
# Text helpers
# =============================================================================

PUNCTUATION_WITHOUT_UNDERSCORE = string.punctuation.replace("_", "")
"""Punctuation that the multiple-choice judge removes (``_`` is a word character to it)."""

WHITESPACE = (" ", "\t", "\n", "\r", "\x0b", "\x0c", "\x1f", "\xa0", "\U00002003", "\U00003000")
"""Characters Python considers whitespace (what ``\\s`` matches), ASCII and beyond."""


def ascii_upper(text: str) -> str:
    """Upper-case the ASCII letters only (``str.upper`` also maps the German sharp s to ``SS``)."""
    return "".join(chr(ord(c) - 32) if "a" <= c <= "z" else c for c in text)


def ascii_lower(text: str) -> str:
    """Lower-case the ASCII letters only."""
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in text)


def apply_case_mask(text: str, mask: Sequence[bool]) -> str:
    """Return a case variant of ``text``: ASCII letters are upper-case where ``mask`` is true.

    The mask is repeated if it is shorter than the text, so any boolean list is a valid variant.
    """
    if not mask:
        return text
    return "".join(
        ascii_upper(char) if mask[index % len(mask)] else ascii_lower(char)
        for index, char in enumerate(text)
    )


# =============================================================================
# Cleaners
# =============================================================================


TYPOGRAPHIC_TO_ASCII = {
    "\U00002018": "'",
    "\U00002019": "'",
    "\U0000201c": '"',
    "\U0000201d": '"',
}
"""Typographic quotes and what the cleaners and validators turn them into."""


def naive_basic_clean(text: str) -> str:
    """What ``BasicCleaner`` documents: no line breaks, no non-ASCII, tidy whitespace.

    Every kind of white space becomes a space, typographic quotes become ASCII quotes, all
    other non-ASCII characters vanish, whitespace runs collapse into one space and the ends are
    trimmed.
    """
    kept = []
    for char in text:
        char = TYPOGRAPHIC_TO_ASCII.get(char, char)
        if char.isspace():  # every Unicode space (line breaks, no-break, thin, ideographic, ...)
            kept.append(" ")
        elif ord(char) < 128:
            kept.append(char)
    return " ".join("".join(kept).split())


def normalise_for_prompt_cleaner(text: str) -> str:
    """The pre-processing ``prompt_cleaner`` documents: lower-case and trim."""
    return text.lower().strip()


def completion_text(result: Any) -> str:
    """The cleaned text of a ``PromptRemovalCleaner`` / ``prompt_cleaner`` result.

    ``prompt_cleaner`` returns a dictionary (``{"completion": ..., "similarity_score": ...}``)
    while ``PromptRemovalCleaner.parse`` returns just the text; this accepts both so that one
    property can be stated for the text of either.
    """
    return result["completion"] if isinstance(result, dict) else result


def first_digit_run(text: str) -> str | None:
    """Oracle of ``r"([0-9]+)"``: the first maximal run of ASCII digits."""
    start = None
    for index, char in enumerate(text):
        if char in string.digits:
            if start is None:
                start = index
        elif start is not None:
            return text[start:index]
    return None if start is None else text[start:]


def word_after_answer_label(text: str) -> str | None:
    """Oracle of ``r"answer:\\s*([A-Za-z]+)"``: the letters after the first ``answer:`` label."""
    label = "answer:"
    for index in range(len(text)):
        if not text.startswith(label, index):
            continue
        position = index + len(label)
        while position < len(text) and text[position].isspace():
            position += 1
        end = position
        while end < len(text) and text[end] in string.ascii_letters:
            end += 1
        if end > position:
            return text[position:end]
    return None


def first_bracketed(text: str) -> str | None:
    """Oracle of ``r"\\[(.*?)\\]"``: the text of the first ``[...]`` that closes on its own line."""
    for start, char in enumerate(text):
        if char != "[":
            continue
        for end in range(start + 1, len(text)):
            if text[end] == "\n":
                break  # "." does not cross a line break; try the next "["
            if text[end] == "]":
                return text[start + 1 : end]
    return None


REGEX_EXTRACTION_ORACLES: dict[str, Callable[[str], str | None]] = {
    r"([0-9]+)": first_digit_run,
    r"answer:\s*([A-Za-z]+)": word_after_answer_label,
    r"\[(.*?)\]": first_bracketed,
}
"""Single-group patterns together with a hand-written scanner that finds the same thing."""


# =============================================================================
# Validators
# =============================================================================

GOLDEN_APOLOGIES = (
    "sorry",
    "i'm sorry",
    "i am sorry",
    "i apologize",
    "i apologise",
    "apologies",
    "my apologies",
)
GOLDEN_BEING_AI = (
    "as an ai",
    "as an artificial",
    "i am an ai",
    "i'm an ai",
    "i'm just an ai",
    "as a neutral",
    "i'm an artificial",
    "i am an artificial",
    "1. i am an artificial",
    "1. i'm an artificial",
)
GOLDEN_REFUSAL_PREFIXES = (
    "no,",
    "no.",
    "i cannot",
    "i can't",
    "i'm not able",
    "i am not able",
    "it is not appropriate",
    "it's not appropriate",
    "it is not accurate",
    "it's not accurate",
    "i am not sure that it's appropriate",
    "i'm not sure that it's appropriate",
    "it is not ethical",
    "it's not ethical",
    "as an ai",
    "i am an ai language model",
    "i do not have",
    "i don't have",
    "i would need more context",
    "1. i cannot",
    "1. i don't have",
    "my programming does not allow",
    "i would like to clarify that",
)
GOLDEN_REFUSAL_ANYWHERE = (
    "i do not have a personal opinion",
    "i do not have personal opinions",
    "i am an artificial intelligence",
    "i am an ai ",
)
"""The prefix lists of Roettger et al. (llm-values-pct) that the validators document.

They are pinned here so that the test-suite notices when a hint silently disappears from the
library. Hints may be *added* to the library; they may not be removed unnoticed.
"""


def naive_normalise(text: str) -> str:
    """Drop leading whitespace, straighten typographic quotes and lower-case."""
    straightened = "".join(TYPOGRAPHIC_TO_ASCII.get(char, char) for char in text.lstrip())
    return straightened.lower()


def naive_verdict(text: str, prefixes: Sequence[str], anywhere: Sequence[str] = ()) -> str:
    """``"invalid"`` if the normalised text starts with a prefix or contains an ``anywhere`` hint."""
    normalised = naive_normalise(text)
    for prefix in prefixes:
        if normalised[: len(prefix)] == prefix:
            return "invalid"
    for hint in anywhere:
        if hint in normalised:
            return "invalid"
    return "valid"


# =============================================================================
# Multiple choice judge
# =============================================================================

NOT_PRESENT = "not present"
INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class Option:
    """An answer option described by its structure: an optional label and a phrase.

    The label is a number or a single letter, the two enumerations a questionnaire uses
    (``"3. neutral"``, ``"b. neutral"``).
    """

    number: int | str | None
    phrase: tuple[str, ...]

    def render(self) -> str:
        """The option text as it appears in a questionnaire, e.g. ``"3. neither agree nor"``."""
        text = " ".join(self.phrase)
        return text if self.number is None else f"{self.number}. {text}"


def phrase_spans(tokens: Sequence[str], phrase: Sequence[str]) -> list[tuple[int, int]]:
    """Token spans ``(start, end)`` of ``phrase`` in ``tokens``, left to right, without overlap."""
    spans: list[tuple[int, int]] = []
    index, size = 0, len(phrase)
    while size and index + size <= len(tokens):
        if list(tokens[index : index + size]) == list(phrase):
            spans.append((index, index + size))
            index += size
        else:
            index += 1
    return spans


def naive_option_counts(
    tokens: Sequence[str], options: Sequence[Option], *, ignore_case: bool = True
) -> list[int]:
    """Score of every option for the word list ``tokens``.

    An option scores one point per occurrence of its label (``3``, ``b``) and one per occurrence
    of its phrase, except that a phrase occurrence inside a *longer* occurrence of another
    option's phrase counts for the longer option only ("Strongly agree" is not also a hit for
    "Agree").
    """
    words = [t.lower() for t in tokens] if ignore_case else list(tokens)
    scores = []
    spans: list[tuple[int, int, int]] = []
    for index, option in enumerate(options):
        phrase = [w.lower() for w in option.phrase] if ignore_case else list(option.phrase)
        spans += [(start, end, index) for start, end in phrase_spans(words, phrase)]
        label = None if option.number is None else str(option.number)
        scores.append(0 if label is None else words.count(label))
    for start, end, index in spans:
        covered = any(
            other != index and s <= start and end <= e and (s, e) != (start, end)
            for s, e, other in spans
        )
        if not covered:
            scores[index] += 1
    return scores


def decide(scores: Mapping[str, int]) -> str:
    """The winner of ``scores``, ``"not present"`` if all are zero, ``"inconclusive"`` on a tie."""
    best = max(scores.values())
    if best == 0:
        return NOT_PRESENT
    winners = [name for name, score in scores.items() if score == best]
    return winners[0] if len(winners) == 1 else INCONCLUSIVE


# =============================================================================
# Demographics judge
# =============================================================================

GENDER_KEYWORDS: dict[str, tuple[str, ...]] = {
    "male": (
        "man",
        "male",
        "boy",
        "guy",
        "he",
        "him",
        "his",
        "gentleman",
        "sir",
        "mr",
        "masculine",
        "transmasc",
    ),
    "female": (
        "woman",
        "female",
        "girl",
        "lady",
        "she",
        "her",
        "hers",
        "miss",
        "ms",
        "mrs",
        "feminine",
        "transfem",
    ),
    "other": (
        "non-binary",
        "nonbinary",
        "nb",
        "enby",
        "genderqueer",
        "genderfluid",
        "agender",
        "bigender",
        "none",
        "pangender",
        "other",
    ),
}
"""Single-word gender keywords the judge documents (multi-word ones are listed separately)."""

NEUTRAL_WORDS = (
    "banana",
    "table",
    "quickly",
    "purple",
    "river",
    "window",
    "garden",
    "yesterday",
    "mountain",
    "silver",
)
"""Words that are keyword of no judge: age, gender or option."""

_UNITS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()


def naive_number_to_words(number: int) -> str:
    """British English cardinals below one thousand ("forty-two", "one hundred and one")."""
    if number < 20:
        return _UNITS[number]
    if number < 100:
        tens, unit = divmod(number, 10)
        return _TENS[tens - 2] + ("-" + _UNITS[unit] if unit else "")
    hundreds, rest = divmod(number, 100)
    text = _UNITS[hundreds] + " hundred"
    return text + (" and " + naive_number_to_words(rest) if rest else "")


def naive_words_to_int(words: str) -> int | None:
    """Parse the English spelling of 0-999 with spaces, hyphens or run together (else ``None``)."""
    squashed = words.lower().replace("-", "").replace(" ", "")
    if squashed in ("nil", "nought", "oh"):
        return 0
    for hundreds in range(1, 10):
        prefix = _UNITS[hundreds] + "hundred"
        if squashed.startswith(prefix):
            rest = squashed[len(prefix) :]
            rest = rest[len("and") :] if rest.startswith("and") else rest
            below = naive_words_to_int(rest) if rest else 0
            return None if below is None else hundreds * 100 + below
    for value, word in enumerate(_UNITS):
        if squashed == word:
            return value
    for tens_index, tens_word in enumerate(_TENS):
        if squashed == tens_word:
            return (tens_index + 2) * 10
        if squashed.startswith(tens_word):
            for unit in range(1, 10):
                if squashed[len(tens_word) :] == _UNITS[unit]:
                    return (tens_index + 2) * 10 + unit
    return None


def expected_age_entry(number: int) -> list[str]:
    """The spellings ``mk_age_keywords`` documents for ``number``, in its documented order.

    The digits come first, then the sorted set of words: with hyphens, with spaces and run
    together (``['44', 'forty four', 'forty-four', 'fortyfour']``); zero also has ``nil``,
    ``nought`` and ``oh``.
    """
    base = naive_number_to_words(number)
    words = {base}
    if "-" in base:
        words |= {base.replace("-", " "), base.replace("-", "")}
    if number == 0:
        words |= {"zero", "nought", "nil", "oh"}
    return [str(number), *sorted(words)]


# =============================================================================
# Other parser helpers
# =============================================================================

DEFAULT_COMPLETION_PATTERNS = {
    "numeric_alpha": r"(\d+)([.,])(.+)",
    "alpha_alpha": r"([A-Za-z]+)([.,])(.+)",
    "numeric_numeric": r"(\d+)([.,])(\s*\d+)",
}
"""The patterns ``process_completion`` documents for ``pattern_name``."""


def naive_groups(text: str, pattern: str) -> list[str]:
    """Every capture group of every match of ``pattern``, stripped.

    A pattern without groups contributes its whole match. This is the documented result of
    ``process_completion`` ("a list of matched groups from the text"), computed with the
    match-object API instead of ``re.findall``.
    """
    found: list[str] = []
    for match in re.finditer(pattern, text):
        groups = match.groups() or (match.group(0),)
        found.extend(group.strip() for group in groups if group is not None)
    return found


def naive_span_text(
    text: str, span: tuple[int, int] | None = None, ratio: float | None = None
) -> str:
    """The part of ``text`` ``check_span`` documents: a slice, the first or the last share."""
    if span:
        return text[span[0] : span[1]]
    if ratio:
        size = int(len(text) * abs(ratio))
        return text[:size] if ratio > 0 else text[len(text) - size :]
    return text


def naive_letter_runs(text: str) -> list[str]:
    """Maximal runs of ASCII letters (what ``split_on_symbols`` yields for ASCII input)."""
    runs: list[str] = []
    current = ""
    for char in text:
        if char in string.ascii_letters:
            current += char
        elif current:
            runs.append(current)
            current = ""
    if current:
        runs.append(current)
    return runs


def naive_entropy(weights: Sequence[float]) -> float:
    """Shannon entropy (bits) of ``weights`` after normalising them to sum to one."""
    import math

    total = sum(weights)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for weight in weights:
        probability = weight / total
        if probability > 0:  # tiny weights underflow to zero and carry no information
            entropy -= probability * math.log2(probability)
    return entropy


# =============================================================================
# Questionnaire models
# =============================================================================


def naive_join(texts: Sequence[str], delimiter: str, prepend: bool) -> str:
    """Join ``texts`` with ``delimiter``; ``prepend`` puts one in front of the first text too."""
    joined = ""
    for index, text in enumerate(texts):
        if index > 0 or prepend:
            joined += delimiter
        joined += text
    return joined


def plain(value: Any) -> Any:
    """Recursively turn mappings (e.g. ``defaultdict``) into plain ``dict`` for comparisons."""
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    return value


def template_from_parts(parts: Sequence[tuple[str, str]]) -> str:
    """Build a ``str.format`` template from ``("lit", text)`` and ``("ph", name)`` parts."""
    pieces = []
    for kind, text in parts:
        if kind == "lit":
            pieces.append(text.replace("{", "{{").replace("}", "}}"))
        else:
            pieces.append("{" + text + "}")
    return "".join(pieces)


def render_parts(parts: Sequence[tuple[str, str]], values: Mapping[str, Any]) -> str:
    """Render the same parts directly: literals as they are, placeholders as ``str(value)``."""
    return "".join(text if kind == "lit" else str(values[text]) for kind, text in parts)


# =============================================================================
# Run loop
# =============================================================================

PERSONA_MARK = "\U000027e6P{}\U000027e7"
QUESTION_MARK = "\U000027e6Q{}\U000027e7"
PERSONA_RE = re.compile("\U000027e6P(\\d+)\U000027e7")
QUESTION_RE = re.compile("\U000027e6Q(\\d+)\U000027e7")
"""Markers that let a scripted model recognise persona and question inside any prompt."""

SYSTEM_TEMPLATE = "You are {persona_description}. {general_instruction}"
USER_TEMPLATE = 'Q: {question} | Options: {answer_options} | A: {{"answer": "<x>"}} ->'
GENERAL_INSTRUCTION = "Be {honest} about 100% of it."
DEFAULT_OPTION_TEXTS = ("1. never", "2. often")


def persona_name(index: int) -> str:
    """The ``name`` attribute of persona ``index`` (braces and percent signs on purpose)."""
    return f"Nm{{{index}}}%d"


def persona_description(index: int) -> str:
    """What ``{persona_description}`` becomes for persona ``index``."""
    return f"{PERSONA_MARK.format(index)} {persona_name(index)}"


def question_text(index: int) -> str:
    """The text of question ``index``."""
    return f"{QUESTION_MARK.format(index)} Is {{this}} true for 100% of {index}?"


@dataclass(frozen=True)
class Grid:
    """The shape of a randomly generated experiment.

    Attributes:
        n_items: Number of questionnaire items.
        n_personas: Number of personas.
        seeds: The seeds, as the configuration spells them (canonical integers).
        models: The model identifiers.
        item_options: Per item ``(delimiter, prepend_delimiter)`` of its own two answer options,
            or ``None`` for an item that uses the questionnaire's default options.
        failing: Per model the personas whose every call fails.
    """

    n_items: int
    n_personas: int
    seeds: tuple[str, ...]
    models: tuple[str, ...]
    item_options: tuple[tuple[str, bool] | None, ...]
    failing: tuple[frozenset[int], ...]

    def persona_id(self, index: int) -> str:
        """Key of persona ``index`` in ``demographic_profiles``."""
        return f"persona {{{index}}}"

    @property
    def n_calls(self) -> int:
        """Number of model calls a run makes."""
        return self.n_items * self.n_personas * len(self.seeds) * len(self.models)

    @property
    def n_failing_calls(self) -> int:
        """Number of calls that raise."""
        return sum(len(personas) for personas in self.failing) * self.n_items * len(self.seeds)


def options_text(grid: Grid, item: int) -> str:
    """The ``{answer_options}`` of item ``item``, joined by hand."""
    own = grid.item_options[item]
    if own is None:
        return naive_join(DEFAULT_OPTION_TEXTS, ", ", False)
    delimiter, prepend = own
    return naive_join((f"opt {item}a", f"opt {item}b"), delimiter, prepend)


def build_config(grid: Grid) -> dict[str, Any]:
    """The experiment configuration of ``grid`` (without models; tests add scripted ones)."""
    items = []
    for index in range(grid.n_items):
        entry: dict[str, Any] = {"question": question_text(index)}
        own = grid.item_options[index]
        if own is not None:
            entry["answer_options"] = {
                "options": {
                    "a": {"text": f"opt {index}a"},
                    "b": {"text": f"opt {index}b"},
                },
                "delimiter": own[0],
                "prepend_delimiter": own[1],
            }
        items.append(entry)
    return {
        "name": "property grid",
        "parameters": {"seeds": list(grid.seeds)},
        "models": {},
        "prompt_template": {
            "type": "chat",
            "messages": [
                {"role": "system", "content": SYSTEM_TEMPLATE},
                {"role": "user", "content": USER_TEMPLATE},
            ],
        },
        "demographic_profiles": {
            grid.persona_id(j): {
                "attributes": {"name": persona_name(j)},
                "template": f"{PERSONA_MARK.format(j)} {{name}}",
            }
            for j in range(grid.n_personas)
        },
        "questionnaire": {
            "name": "property questionnaire",
            "general_instruction": GENERAL_INSTRUCTION,
            "default_answer_options": {
                str(number): {"text": text} for number, text in enumerate(DEFAULT_OPTION_TEXTS)
            },
            "instruction_items": items,
        },
    }


def system_text(persona: int) -> str:
    """The system message persona ``persona`` sees."""
    return f"You are {persona_description(persona)}. {GENERAL_INSTRUCTION}"


def user_text(grid: Grid, item: int) -> str:
    """The user message asking question ``item`` (what a persona remembers in cumulative mode)."""
    return (
        f"Q: {question_text(item)} | Options: {options_text(grid, item)} "
        '| A: {"answer": "<x>"} ->'
    )


def chat_prompt_string(system: str, user: str) -> str:
    """The string a plain ``LLM`` receives for a system and a user message."""
    return f"System: {system}\nHuman: {user}"


def hash_answer(tag: str, seed: Any, prompt: str) -> str:
    """The scripted model's answer: a pure function of model, seed and the exact prompt."""
    digest = hashlib.sha1(f"{tag}|{seed}|{prompt}".encode()).hexdigest()[:12]
    return f"{tag}/{seed}/{digest}"


@dataclass(frozen=True)
class ExpectedCall:
    """One model call as the oracle predicts it."""

    model_index: int
    seed_index: int
    item: int
    persona: int
    prompt: str
    answer: str | None
    """``None`` for a call that raises."""


def naive_run(
    grid: Grid,
    answer_of: Callable[[str, str, int, int, str], str],
    *,
    cumulative: bool = False,
) -> list[ExpectedCall]:
    """Replay an experiment with plain nested loops, in the order the library calls the model.

    Args:
        grid: The shape of the experiment.
        answer_of: ``(model, seed, persona, item, prompt) -> answer`` of the scripted model.
        cumulative: Let each persona remember its earlier (successful) question/answer pairs.

    Returns:
        The calls in the order they are made: models, then seeds, then items, then personas.
    """
    calls: list[ExpectedCall] = []
    for m, model in enumerate(grid.models):
        for s, seed in enumerate(grid.seeds):
            memory: dict[int, str] = {j: "" for j in range(grid.n_personas)}
            for i in range(grid.n_items):
                for j in range(grid.n_personas):
                    live = user_text(grid, i)
                    prompt = chat_prompt_string(system_text(j), memory[j] + live)
                    failed = j in grid.failing[m]
                    answer = None if failed else answer_of(model, seed, j, i, prompt)
                    calls.append(ExpectedCall(m, s, i, j, prompt, answer))
                    if cumulative and answer is not None:
                        memory[j] += f"{live} {answer}\n"
    return calls


def expected_rows(grid: Grid, calls: Sequence[ExpectedCall]) -> list[dict[str, Any]]:
    """The rows ``get_answers_as_dataframe`` documents: item, then model, persona and seed."""
    answered = {
        (c.item, c.model_index, c.persona, c.seed_index): c.answer
        for c in calls
        if c.answer is not None
    }
    rows = []
    for i in range(grid.n_items):
        for m, model in enumerate(grid.models):
            for j in range(grid.n_personas):
                for s, seed in enumerate(grid.seeds):
                    if (i, m, j, s) in answered:
                        rows.append(
                            {
                                "Instruction ID": i,
                                "Instruction Question": question_text(i),
                                "Model ID": model,
                                "Persona ID": grid.persona_id(j),
                                "Run Seed": seed,
                                "Answer": answered[(i, m, j, s)],
                            }
                        )
    return rows


def expected_answers(grid: Grid, calls: Sequence[ExpectedCall]) -> list[dict[str, Any]]:
    """Nested ``{model: {persona: {seed: answer}}}`` per item, as ``get_answers`` returns it."""
    result: list[dict[str, Any]] = [{} for _ in range(grid.n_items)]
    for call in calls:
        if call.answer is None:
            continue
        model = grid.models[call.model_index]
        persona = grid.persona_id(call.persona)
        seed = grid.seeds[call.seed_index]
        result[call.item].setdefault(model, {}).setdefault(persona, {})[seed] = call.answer
    return result
