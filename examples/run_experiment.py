"""Run an experiment from a JSON configuration and stream the answers into a CSV file.

**This script DOWNLOADS a language model** (unless it is already in your Hugging Face cache) and
runs it on your machine. The bundled Big Five example uses ``Qwen/Qwen2.5-0.5B-Instruct`` (about
1 GB; its eight generations take minutes at most on a laptop CPU). Needs the ``huggingface``
extra::

    pip install "rupsycho[huggingface] @ git+https://github.com/julianschelb/rupsycho.git"

Usage::

    python examples/run_experiment.py                          # bundled example, its own model
    python examples/run_experiment.py my_config.json           # your configuration
    python examples/run_experiment.py --model Qwen/Qwen2.5-1.5B-Instruct   # another model
    python examples/run_experiment.py --model ./my-local-model --device cpu
    python examples/run_experiment.py --dry-run                # check everything, load no model

Without ``--model`` the models of the configuration are used, whatever their type (local
Hugging Face, OpenAI, Ollama, ...). With ``--model NAME`` they are replaced by one local
Hugging Face model, given by a Hub name or a local directory. ``--dry-run`` validates the
configuration, shows the size of the experiment and the assembled prompt, and exits before any
model is loaded; use it to check a configuration offline.

The answers are appended to ``<name>_answers.csv`` as they are generated, so a crash does not
lose them. The script refuses to write into an existing file unless ``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import rupsycho as rup
from rupsycho.callbacks import CSVCallback


def local_model_section(name: str, device: str, max_new_tokens: int) -> dict[str, Any]:
    """The configuration section of a local Hugging Face model."""
    return {
        "type": "local_huggingface",
        "name_or_path": name,
        "device_map": device,
        "task": "text-generation",
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "return_full_text": False,
        },
    }


def build_experiment(
    config_path: Path | None,
    *,
    model: str | None = None,
    device: str = "cpu",
    max_new_tokens: int = 64,
    seeds: Sequence[str] | None = None,
) -> rup.ExperimentDocument:
    """Load the configuration, apply the command-line overrides and validate it.

    Args:
        config_path: JSON configuration, or ``None`` for the bundled Big Five example.
        model: Replace the configured models by this local Hugging Face model.
        device: ``device_map`` of that model (``cpu``, ``cuda``, ``mps`` or ``auto``).
        max_new_tokens: Generation limit for that model.
        seeds: Replace the seeds of the configuration.

    Returns:
        The experiment. No model is loaded yet; that happens when the run reaches it.
    """
    if config_path is None:
        config = rup.load_example_config("bfi")
    else:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    if model:
        name = Path(model).name if Path(model).is_dir() else model
        config["models"] = {name: local_model_section(model, device, max_new_tokens)}
    parameters = config.setdefault("parameters", {})
    if seeds:
        parameters["seeds"] = list(seeds)
    # Load models when the run reaches them, never while validating (also for --dry-run)
    parameters["lazy_load_models"] = True
    return rup.experiment_from_dict(config)


def describe(experiment: rup.ExperimentDocument) -> None:
    """Print what is going to run."""
    items = experiment.questionnaire.get_number_of_questions() if experiment.questionnaire else 0
    seeds = experiment.parameters.seeds or ["(random)"]
    calls = math.prod(
        (items, len(experiment.demographic_profiles), len(experiment.models), len(seeds))
    )
    print(f"experiment: {experiment.name}")
    for identifier, model in experiment.models.items():
        target = getattr(model, "name_or_path", None) or getattr(model, "model", "")
        print(f"model     : {identifier} ({model.type}: {target})")
    print(f"seeds     : {', '.join(map(str, seeds))}")
    print(
        f"calls     : {items} questions x {len(experiment.demographic_profiles)} personas "
        f"x {len(experiment.models)} models x {len(seeds)} seeds = {calls}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="JSON configuration (default: bundled Big Five example)",
    )
    parser.add_argument(
        "--model", help="run this local Hugging Face model instead of the configured ones"
    )
    parser.add_argument("--device", default="cpu", help="device_map of --model (default: cpu)")
    parser.add_argument(
        "--max-new-tokens", type=int, default=64, help="generation limit of --model"
    )
    parser.add_argument("--seeds", nargs="+", help="replace the seeds of the configuration")
    parser.add_argument(
        "-o", "--output", type=Path, help="CSV file for the answers (default: <name>_answers.csv)"
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    parser.add_argument(
        "--max-concurrency", type=int, default=1, help="parallel calls for API models (default: 1)"
    )
    parser.add_argument(
        "--on-error", choices=("warn", "raise", "ignore"), default="warn", help="failed model calls"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and preview, load no model"
    )
    args = parser.parse_args(argv)

    if args.config is not None and not args.config.is_file():
        parser.error(f"{args.config}: no such file")
    stem = args.config.stem if args.config else "bfi"
    output = args.output or Path(f"{stem}_answers.csv")
    if output.exists() and not args.overwrite and not args.dry_run:
        parser.error(f"{output} already exists; use --overwrite or choose another --output")

    try:
        experiment = build_experiment(
            args.config,
            model=args.model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            seeds=args.seeds,
        )
    except (ValueError, TypeError, AttributeError) as error:  # bad JSON or a bad configuration
        parser.error(f"invalid configuration: {error}")

    describe(experiment)
    if args.dry_run:
        experiment.print_assembled_prompt(item_idx=0, persona_idx=0)
        print("\nDry run: no model was loaded and nothing was written.")
        return 0

    output.unlink(missing_ok=True)  # the callback appends, so start from an empty file
    print("\nRunning. Models that are not cached yet are downloaded first; this can take a while.")
    summary = experiment.run(
        callbacks=[CSVCallback(output)],
        max_concurrency=args.max_concurrency,
        on_error=args.on_error,
    )
    print(f"\n{summary}; answers streamed to {output}")
    print(experiment.get_answers_as_dataframe().head(10).to_string(index=False))
    return 3 if summary.n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
