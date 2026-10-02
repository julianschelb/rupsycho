"""Synthetic experiments for the benchmarks: any grid size, no network, no model downloads.

An experiment asks ``n_items`` questions to ``n_personas`` personas for ``n_seeds`` seeds, so one
``run()`` makes ``n_items * n_personas * n_seeds`` model calls per model. ``synthetic_config``
builds such a configuration with ``"models": {}``; a scripted fake model (``ScriptedLLM`` or
``ScriptedChatModel``, optionally sleeping to imitate network latency) is attached with
``experiment.add_model`` by ``build_experiment``.

The fake models answer deterministically: the answer is a function of the seed and the full
prompt text. Two runs that produce identical answers therefore asked the same questions, in
the same prompt wording, with the same seed.

The module also hosts the small helpers that all benchmark scripts share (machine description,
table formatting, extraction of the pre-restructure source tree). It only imports
``langchain_core`` at import time (``rupsycho`` is imported lazily), so the legacy benchmark can
import it with a different ``rupsycho`` tree on ``PYTHONPATH``.

Example:
    Run ``python benchmarks/synthetic.py --items 3 --personas 2 --show-prompt`` to inspect a
    small synthetic experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import tarfile
import time
import zlib
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.language_models.llms import LLM

if TYPE_CHECKING:
    from rupsycho import ExperimentDocument

__all__ = [
    "LEGACY_COMMIT",
    "ModelKind",
    "RecordingCallback",
    "ScriptedChatModel",
    "ScriptedLLM",
    "answers_digest",
    "answers_table",
    "build_experiment",
    "extract_legacy_tree",
    "format_table",
    "git_revision",
    "grid_size",
    "library_versions",
    "machine_description",
    "make_model",
    "parse_grid",
    "synthetic_config",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_COMMIT = "94f99d7"
"""The last commit before the repository restructuring (the baseline of the legacy benchmarks)."""

ModelKind = Literal["llm", "chat"]
"""``"llm"``: text-completion model (``BaseLLM``, like Ollama or a local pipeline);
``"chat"``: chat model (``BaseChatModel``, like OpenAI, DeepSeek or Gemini)."""

GENERAL_INSTRUCTION = (
    "Here are a number of characteristics that may or may not apply to you. For example, do you "
    "agree that you are someone who likes to spend time with others? Please return the number "
    "corresponding to the answer options to indicate the extent to which you agree or disagree "
    "with that statement."
)
"""Same instruction as the bundled BFI example, so that prompts have a realistic length."""

SYSTEM_PROMPT = (
    "Objective: Act like you are {persona_description}, a survey participant answering a "
    "questionnaire.\n{general_instruction}\nInstructions: Choose from the list of answer options "
    "to answer the question. Answer the question using only the provided answer options. If none "
    "of the options are correct, choose the option that is closest to being correct. The solution "
    'must be provided in this format: {{answer: "answer option"}}'
)
USER_PROMPT = "Question: {question}\nAnswer Options: {answer_options}\nAnswer:"
"""Chat prompt of the bundled BFI example (identical placeholders and wording)."""

_STEMS = (
    "Is talkative",
    "Tends to find fault with others",
    "Does a thorough job",
    "Is depressed, blue",
    "Is original, comes up with new ideas",
    "Is reserved",
    "Is helpful and unselfish with others",
    "Can be somewhat careless",
    "Is relaxed, handles stress well",
    "Is curious about many different things",
)
_TRAITS = ("optimistic", "conservative", "curious", "reserved", "outgoing")


class ScriptedLLM(LLM):
    """Deterministic fake text-completion model with optional latency.

    The answer is ``{answer: "<k>"}`` with ``k`` derived from the seed and the prompt, so it
    is reproducible and differs between prompts. The seed is read from the ``seed`` field
    (rupsycho sets it on a copy of the model, like it does for OpenAI or Ollama) or from a
    ``seed`` keyword argument (what the pre-restructure release passed via ``model.bind``).

    Attributes:
        latency: Seconds each call sleeps (``time.sleep`` releases the GIL, like network I/O).
        seed: Seed of the current run; ``None`` if the model was not seeded.
        n_options: Number of answer options to choose from.
        fail_on: Regular expression; calls whose prompt matches it (``re.search`` with
            ``DOTALL``) raise ``RuntimeError``. Use it to make chosen prompts fail.

    Example:
        ```python
        model = ScriptedLLM(latency=0.02)
        model.invoke("Question?")  # '{answer: "5"}' after 20 ms
        ```
    """

    latency: float = 0.0
    seed: int | None = None
    n_options: int = 5
    fail_on: str | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _call(
        self, prompt: str, stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any
    ) -> str:
        return _answer(self, prompt, kwargs.get("seed"))


class ScriptedChatModel(SimpleChatModel):
    """Deterministic fake chat model with optional latency (see :class:`ScriptedLLM`).

    Attributes:
        latency: Seconds each call sleeps.
        seed: Seed of the current run; ``None`` if the model was not seeded.
        n_options: Number of answer options to choose from.
        fail_on: Regular expression; calls whose prompt matches it raise ``RuntimeError``.
    """

    latency: float = 0.0
    seed: int | None = None
    n_options: int = 5
    fail_on: str | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted-chat"

    def _call(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        prompt = "\n".join(str(message.content) for message in messages)
        return _answer(self, prompt, kwargs.get("seed"))


def _answer(model: ScriptedLLM | ScriptedChatModel, prompt: str, seed: Any) -> str:
    """Shared behaviour of the scripted models: sleep, maybe fail, answer deterministically."""
    if model.latency:
        time.sleep(model.latency)
    if model.fail_on and re.search(model.fail_on, prompt, re.DOTALL):
        raise RuntimeError(f"cannot answer: {model.fail_on}")
    seed = model.seed if seed is None else seed
    choice = 1 + zlib.crc32(f"{seed}|{prompt}".encode()) % model.n_options
    return f'{{answer: "{choice}"}}'


def make_model(
    kind: ModelKind = "llm",
    *,
    latency: float = 0.0,
    n_options: int = 5,
    fail_on: str | None = None,
) -> ScriptedLLM | ScriptedChatModel:
    """Create a scripted fake model.

    Args:
        kind: ``"llm"`` for a text-completion model, ``"chat"`` for a chat model.
        latency: Seconds each call sleeps.
        n_options: Number of answer options to choose from.
        fail_on: Regular expression; calls whose prompt matches it raise ``RuntimeError``.

    Returns:
        The model, ready for ``experiment.add_model``.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    if kind not in ("llm", "chat"):
        raise ValueError(f"kind must be 'llm' or 'chat', got {kind!r}")
    cls = ScriptedLLM if kind == "llm" else ScriptedChatModel
    return cls(latency=latency, n_options=n_options, fail_on=fail_on)


def grid_size(n_items: int, n_personas: int, n_seeds: int = 1, n_models: int = 1) -> int:
    """Number of model calls of one ``run()``: items x personas x seeds x models."""
    return n_items * n_personas * n_seeds * n_models


def synthetic_config(
    n_items: int = 20,
    n_personas: int = 5,
    n_seeds: int = 1,
    n_options: int = 5,
    *,
    name: str | None = None,
) -> dict[str, Any]:
    """Build an experiment configuration of arbitrary size.

    The configuration uses the chat prompt, the instruction text and the answer-option layout
    of the bundled BFI example, so that prompts have a realistic length (about 850 characters).
    The ``models`` section is empty: add a model with ``experiment.add_model``.

    Args:
        n_items: Number of questionnaire items.
        n_personas: Number of personas (demographic profiles).
        n_seeds: Number of seeds; one pass over the grid is made per seed.
        n_options: Number of default answer options per item.
        name: Experiment name; generated from the sizes if omitted.

    Returns:
        A configuration dictionary for ``rupsycho.experiment_from_dict``.

    Raises:
        ValueError: If any size is smaller than one.

    Example:
        ```python
        config = synthetic_config(n_items=10, n_personas=4, n_seeds=2)
        assert len(config["questionnaire"]["instruction_items"]) == 10
        ```
    """
    for label, value in (
        ("n_items", n_items),
        ("n_personas", n_personas),
        ("n_seeds", n_seeds),
        ("n_options", n_options),
    ):
        if value < 1:
            raise ValueError(f"{label} must be at least 1, got {value}")

    personas = {
        f"Persona {index + 1}": {
            "attributes": {
                "age": 18 + index % 60,
                "title": "Ms" if index % 2 == 0 else "Mr",
                "name": f"Person{index + 1}",
            },
            "template": "{title} {name} is {age} years old and has a "
            + _TRAITS[index % len(_TRAITS)]
            + " personality.",
        }
        for index in range(n_personas)
    }
    items = [
        {
            "question": f"{_STEMS[index % len(_STEMS)]} (item {index + 1})",
            "reversed": index % 7 == 3,
            "attributes": {"dimension": str(1 + index % 5)},
        }
        for index in range(n_items)
    ]
    options = {
        str(k): {"text": f"{k}. Option {k}", "ignored_for_scale": False, "weight": k}
        for k in range(1, n_options + 1)
    }
    return {
        "name": name or f"Synthetic {n_items}x{n_personas}x{n_seeds}",
        "description": "Synthetic experiment for benchmarks.",
        "parameters": {"seeds": [str(seed) for seed in range(1, n_seeds + 1)]},
        "models": {},
        "prompt_template": {
            "type": "chat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
        },
        "demographic_profiles": personas,
        "questionnaire": {
            "name": "Synthetic questionnaire",
            "general_instruction": GENERAL_INSTRUCTION,
            "attributes": {"dimension": {str(d): f"Dimension {d}" for d in range(1, 6)}},
            "default_answer_options": options,
            "instruction_items": items,
        },
    }


def build_experiment(
    n_items: int = 20,
    n_personas: int = 5,
    n_seeds: int = 1,
    n_options: int = 5,
    *,
    kind: ModelKind = "llm",
    latency: float = 0.0,
    fail_on: str | None = None,
    model_id: str = "fake",
) -> ExperimentDocument:
    """Create a synthetic experiment with one scripted fake model attached.

    Args:
        n_items: Number of questionnaire items.
        n_personas: Number of personas.
        n_seeds: Number of seeds.
        n_options: Number of answer options.
        kind: ``"llm"`` or ``"chat"`` fake model.
        latency: Seconds each model call sleeps.
        fail_on: Regular expression; calls whose prompt matches it fail.
        model_id: Identifier of the model in the experiment.

    Returns:
        A ready-to-run experiment (``n_items * n_personas * n_seeds`` calls).

    Example:
        ```python
        experiment = build_experiment(n_items=8, n_personas=4)
        summary = experiment.run(show_progress=False)
        ```
    """
    import rupsycho as rup

    experiment = rup.experiment_from_dict(synthetic_config(n_items, n_personas, n_seeds, n_options))
    experiment.add_model(
        make_model(kind, latency=latency, n_options=n_options, fail_on=fail_on),
        identifier=model_id,
    )
    return experiment


# ------------------- Verifying that two runs did the same work -------------------


class RecordingCallback:
    """Callback that records ``(item_id, model, persona, seed, answer)`` in callback order.

    It only relies on the positional arguments the run loop passes, which are the same in the
    current and in the pre-restructure release, so it works with both.
    """

    def __init__(self) -> None:
        self.rows: list[tuple[int, str, str, str, Any]] = []

    def save_answer(
        self,
        experiment: Any,
        item_id: int,
        item: Any,
        model_id: str,
        persona_id: str,
        seed: Any,
        *rest: Any,
    ) -> None:
        # ``rest`` is ``(elapsed, answer)``; only the answer is of interest
        self.rows.append((item_id, model_id, persona_id, str(seed), rest[-1]))


def answers_table(experiment: Any) -> list[tuple[int, str, str, str, Any]]:
    """All answers stored on the questionnaire items as sorted ``(item, model, persona, seed, answer)``."""
    rows = []
    for item_id, item in enumerate(experiment.questionnaire.instruction_items):
        for model_id, personas in item.answers.items():
            for persona_id, seeds in personas.items():
                for seed, answer in seeds.items():
                    rows.append((item_id, str(model_id), str(persona_id), str(seed), answer))
    return sorted(rows)


def answers_digest(rows: Sequence[Any]) -> str:
    """Short SHA-256 fingerprint of a list of answer rows (order matters).

    Args:
        rows: For instance the output of :func:`answers_table` or ``RecordingCallback.rows``.

    Returns:
        The first 16 hex digits of the hash.
    """
    return hashlib.sha256(json.dumps(list(rows), default=str).encode()).hexdigest()[:16]


# ------------------- Helpers shared by the benchmark scripts -------------------


def parse_grid(text: str) -> tuple[int, int, int]:
    """Parse ``"ITEMSxPERSONASxSEEDS"`` (the seeds part is optional) into three integers.

    Args:
        text: For instance ``"50x100x3"`` or ``"20x5"`` (one seed).

    Returns:
        ``(n_items, n_personas, n_seeds)``.

    Raises:
        ValueError: If the text is not of that form.
    """
    parts = text.lower().split("x")
    if len(parts) not in (2, 3) or not all(part.isdigit() and int(part) > 0 for part in parts):
        raise ValueError(f"grid must look like ITEMSxPERSONAS[xSEEDS], got {text!r}")
    numbers = [int(part) for part in parts]
    return numbers[0], numbers[1], numbers[2] if len(numbers) == 3 else 1


def format_table(
    headers: Sequence[str], rows: Sequence[Sequence[Any]], align: str | None = None
) -> str:
    """Render a GitHub-flavoured Markdown table with padded columns (readable as plain text).

    Args:
        headers: Column titles.
        rows: Table rows; every cell is converted with ``str``.
        align: One letter per column, ``"l"`` or ``"r"``. Defaults to a left-aligned first
            column and right-aligned others.

    Returns:
        The table, without a trailing newline.
    """
    align = align or "l" + "r" * (len(headers) - 1)
    cells = [[str(cell) for cell in row] for row in rows]
    widths = [
        max([len(header)] + [len(row[index]) for row in cells])
        for index, header in enumerate(headers)
    ]

    def line(values: Sequence[str]) -> str:
        padded = [
            value.ljust(width) if side == "l" else value.rjust(width)
            for value, width, side in zip(values, widths, align)  # type: ignore[arg-type]
        ]
        return "| " + " | ".join(padded) + " |"

    rule = "|" + "|".join(
        (":" + "-" * (width + 1)) if side == "l" else ("-" * (width + 1) + ":")
        for width, side in zip(widths, align)
    )
    return "\n".join([line(headers), rule + "|", *(line(row) for row in cells)])


def machine_description() -> str:
    """Describe the machine: CPU model, core count, operating system and Python version."""
    cpu = ""
    try:
        if platform.system() == "Darwin":
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        elif platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except (OSError, subprocess.SubprocessError):
        pass
    cpu = cpu or platform.processor() or platform.machine()
    return (
        f"{cpu}, {os.cpu_count()} logical cores, {platform.platform(terse=True)}, "
        f"Python {platform.python_version()}"
    )


def library_versions(names: Sequence[str] = ("langchain-core", "pydantic", "tqdm")) -> str:
    """Installed versions of the libraries that dominate the run loop, e.g. ``"pydantic 2.13"``."""
    found = []
    for name in names:
        try:
            found.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            found.append(f"{name} (not installed)")
    return ", ".join(found)


def git_revision(repo: str | os.PathLike[str] = REPO_ROOT) -> str:
    """Short commit hash of the checkout, plus a note if tracked files under ``src/`` changed.

    Returns:
        For example ``"e6ec7b3"`` or ``"e6ec7b3 + uncommitted changes in src/"``; ``"unknown"``
        if git is unavailable or ``repo`` is no git checkout.
    """
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        changed = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no", "--", "src"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return commit + (" + uncommitted changes in src/" if changed else "")


def extract_legacy_tree(
    dest: str | os.PathLike[str],
    commit: str = LEGACY_COMMIT,
    repo: str | os.PathLike[str] = REPO_ROOT,
) -> Path:
    """Extract ``src/`` of an old commit with ``git archive`` so that it can be imported.

    Args:
        dest: Existing directory to extract into.
        commit: The commit to extract (default: the last one before the restructuring).
        repo: The git repository containing the commit.

    Returns:
        The extracted ``src`` directory; put it on ``PYTHONPATH`` to import the old package.

    Raises:
        RuntimeError: If git is missing or does not know the commit (for instance in a shallow
            clone).
    """
    try:
        archive = subprocess.run(
            ["git", "-C", str(repo), "archive", commit, "src"], capture_output=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", b"") or str(error).encode()
        raise RuntimeError(
            f"cannot extract commit {commit} from {repo} with 'git archive': "
            f"{detail.decode(errors='replace').strip()}. Fetch the full history or pass an "
            "already extracted tree with --legacy-src."
        ) from error
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:  # pragma: no cover - Python without the extraction filters
            tar.extractall(dest)
    return Path(dest) / "src"


# ------------------- Command line -------------------


def main(argv: Sequence[str] | None = None) -> None:
    """Describe a synthetic experiment and run it once with the zero-latency fake model."""
    parser = argparse.ArgumentParser(
        description="Build a synthetic experiment, run it once and print what was generated."
    )
    parser.add_argument("--items", type=int, default=4, help="questionnaire items (default: 4)")
    parser.add_argument("--personas", type=int, default=2, help="personas (default: 2)")
    parser.add_argument("--seeds", type=int, default=1, help="seeds (default: 1)")
    parser.add_argument("--options", type=int, default=5, help="answer options (default: 5)")
    parser.add_argument("--kind", choices=["llm", "chat"], default="llm", help="fake model type")
    parser.add_argument("--show-prompt", action="store_true", help="print one assembled prompt")
    parser.add_argument("--dump", metavar="FILE", help="write the configuration as JSON to FILE")
    args = parser.parse_args(argv)

    config = synthetic_config(args.items, args.personas, args.seeds, args.options)
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        print(f"configuration written to {args.dump}")

    experiment = build_experiment(
        args.items, args.personas, args.seeds, args.options, kind=args.kind
    )
    if args.show_prompt:
        experiment.print_assembled_prompt(0, 0)
    summary = experiment.run(show_progress=False)
    rows = answers_table(experiment)
    print(
        f"{args.items} items x {args.personas} personas x {args.seeds} seeds = "
        f"{grid_size(args.items, args.personas, args.seeds)} calls: {summary}; "
        f"{len(rows)} answers, digest {answers_digest(rows)}"
    )


if __name__ == "__main__":
    main()
