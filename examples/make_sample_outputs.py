"""Regenerate the sample result files in ``examples/data/output/``.

The post-processing example (``Postprocessing.ipynb``) needs result files such as the ones
``CSVCallback`` and ``JSONLCallback`` write. This script creates them **offline**: it runs the
demo configuration ``examples/data/bfi_demo_config.json`` through the real run loop and the
real callbacks, but answers every question with a scripted stand-in model (``FakeListLLM`` with
hand-written replies), so the model id of the rows is ``scripted``. The replies imitate the
typical problems of free-text answers: the requested ``{answer: ...}`` format, bare numbers,
chatty sentences, emoji and line breaks, a refusal, and answers that name no option or two
options at once.

Re-run it whenever the callbacks change their output format (``tests/test_examples.py`` fails
when the files in the repository are out of date)::

    python examples/make_sample_outputs.py
    python examples/make_sample_outputs.py --output-dir /tmp/sample-outputs

Nothing is downloaded and no key is needed.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Sequence
from pathlib import Path

from langchain_core.language_models.fake import FakeListLLM

import rupsycho as rup
from rupsycho.callbacks import CSVCallback, JSONLCallback

EXAMPLES_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXAMPLES_DIR / "data" / "bfi_demo_config.json"
DEFAULT_OUTPUT_DIR = EXAMPLES_DIR / "data" / "output"
STEM = "bfi_demo_config_output"
MODEL_ID = "scripted"

# One reply per model call, in run order: question 0 (Optimistic, Conservative), question 1, ...
REPLIES = [
    '{answer: "4. Agree a little"}',  # the format the prompt asks for
    "3",  # a bare option number
    'I would say "2. Disagree a little".',  # a sentence around the option
    "As an AI, I do not have personal opinions on this.",  # a refusal
    "5. Agree strongly\n\nI always make sure to do a thorough job. \N{SMILING FACE WITH SMILING EYES}",
    "It depends on the situation.",  # names no option at all
    "Disagree strongly.",  # the option text without its number
    "Somewhere between 2 and 3.",  # names two options at once
]


def generate(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Run the demo configuration with the scripted model and write the sample files.

    Existing sample files are replaced (the callbacks would append to them otherwise).

    Args:
        output_dir: Directory for ``bfi_demo_config_output.csv`` and ``.jsonl``.

    Returns:
        The paths of the written files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{STEM}.csv"
    jsonl_path = output_dir / f"{STEM}.jsonl"
    for path in (csv_path, jsonl_path):
        path.unlink(missing_ok=True)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["models"] = {}  # the scripted model replaces the Hugging Face model of the demo
    experiment = rup.experiment_from_dict(config)
    experiment.add_model(FakeListLLM(responses=REPLIES), identifier=MODEL_ID)

    with warnings.catch_warnings():
        # FakeListLLM has no seed parameter; the replies do not depend on it anyway
        warnings.filterwarnings("ignore", message="Models of type FakeListLLM")
        summary = experiment.run(
            callbacks=[CSVCallback(csv_path), JSONLCallback(jsonl_path)], show_progress=False
        )
    if summary.n_failed:
        raise RuntimeError(f"the scripted run failed: {summary.errors}")
    return [csv_path, jsonl_path]


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="where to write the files (default: examples/data/output)",
    )
    args = parser.parse_args(argv)
    for path in generate(args.output_dir):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
