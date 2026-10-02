"""Check that identical seeds reproduce identical answers, and see what the seeding does.

Runs offline: nothing is downloaded and no key is needed. Every check prints ``identical`` or
``different`` next to what it expects, and the exit code is 0 only if all checks hold::

    python examples/reproducibility_check.py
    python examples/reproducibility_check.py --skip-local   # without the Hugging Face part

1. **A model with a ``seed`` field** (a scripted stand-in that behaves like an API model called
   with a seed): the same seeds give the same answers, other seeds give other answers.
2. **A model without seed support** (the situation with Google Gemini): ``rupsycho.seeding``
   warns, and two runs with the same seeds differ.
3. **A tiny local Hugging Face model** with random weights, built on the fly (needs the
   ``huggingface`` extra, otherwise this part is skipped). ``rupsycho`` seeds local pipelines with
   ``transformers.set_seed`` right before every call. Without that, two sampled generations from
   the same prompt differ.

The stand-in models here are *not* language models; they only make the seeding observable
without downloads.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models.llms import LLM

import rupsycho as rup
from rupsycho.seeding import seed_model, supports_seeding

SEEDS = ["1", "2"]
OTHER_SEEDS = ["3", "4"]


class ScriptedLLM(LLM):
    """STAND-IN for a model behind an API that accepts a seed.

    The reply is an option number drawn from the seed and the prompt: the same seed and prompt
    give the same reply. Without a seed every call is a fresh sample.
    """

    seed: int | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        seed = self.seed
        rng = random.SystemRandom() if seed is None else random.Random(f"{seed}|{prompt}")
        return str(rng.randint(1, 5))


class CoinFlipLLM(LLM):
    """STAND-IN for a back-end without seed support: it draws from the global random state."""

    @property
    def _llm_type(self) -> str:
        return "coin-flip"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        return str(random.randint(1, 5))


def new_config(seeds: Sequence[str]) -> dict[str, Any]:
    """The bundled Big Five example without its model, with the given seeds."""
    config = rup.load_example_config("bfi")
    config["models"] = {}
    config["parameters"]["seeds"] = list(seeds)
    return config


def answers(experiment: rup.ExperimentDocument) -> list[str]:
    """Run the experiment and return all answers in a fixed order."""
    experiment.run(show_progress=False)
    return [str(answer) for answer in experiment.get_answers_as_dataframe()["Answer"]]


def answers_of_model(model: LLM, seeds: Sequence[str]) -> list[str]:
    """Answers of the bundled example when ``model`` is added in code."""
    experiment = rup.experiment_from_dict(new_config(seeds))
    experiment.add_model(model, identifier="model")
    return answers(experiment)


class Report:
    """Print the outcome of each check and remember how many of them did not hold."""

    def __init__(self) -> None:
        self.failed = 0

    def expect(
        self,
        label: str,
        observed: bool,
        expected: bool,
        words: tuple[str, str] = ("identical", "different"),
    ) -> None:
        """Print ``label`` with the observed and the expected outcome (``words[0]`` if true)."""
        ok = observed == expected
        self.failed += not ok
        flag = "" if ok else "  <-- FAILED"
        print(f"  {label:<36} {words[not observed]:<9} (expected {words[not expected]}){flag}")


def check_seeded_api_model(report: Report) -> None:
    print("1. Model with a seed field (stand-in for an API model)")
    model = ScriptedLLM()
    seeded = seed_model(model, 42)
    print(f"  supports_seeding: {supports_seeding(model)}; seed_model(model, 42).seed: ", end="")
    print(getattr(seeded, "seed", None))
    first, again = answers_of_model(model, SEEDS), answers_of_model(model, SEEDS)
    other = answers_of_model(model, OTHER_SEEDS)
    report.expect("same seeds, run twice", first == again, expected=True)
    report.expect("different seeds", first == other, expected=False)


def check_unseedable_model(report: Report) -> None:
    print("2. Model without seed support (stand-in for Google Gemini)")
    model = CoinFlipLLM()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = answers_of_model(model, SEEDS)
    again = answers_of_model(model, SEEDS)
    warned = any("cannot be seeded" in str(warning.message) for warning in caught)
    print(f"  supports_seeding: {supports_seeding(model)}")
    report.expect("rupsycho warns", warned, expected=True, words=("yes", "no"))
    report.expect("same seeds, run twice", first == again, expected=False)


def have(*modules: str) -> bool:
    return all(importlib.util.find_spec(module) is not None for module in modules)


def build_tiny_model(directory: Path) -> Path:
    """Create a randomly initialised one-layer GPT-2 with a character tokenizer in ``directory``.

    It produces gibberish, but it goes through the real Transformers pipeline and sampling code.
    """
    import string

    import torch
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    specials = ["[PAD]", "[UNK]", "[EOS]"]
    characters = string.ascii_letters + string.digits + string.punctuation + " \n"
    vocab = {token: index for index, token in enumerate(specials + list(characters))}

    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}assistant:"
    )

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=4096,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=vocab["[EOS]"],
        eos_token_id=vocab["[EOS]"],
        pad_token_id=vocab["[PAD]"],
    )
    GPT2LMHeadModel(config).save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory


def check_local_model(report: Report) -> None:
    print("3. Tiny local Hugging Face model (random weights, built on the fly)")
    if not have("torch", "transformers", "tokenizers", "accelerate", "langchain_huggingface"):
        print("  skipped: needs the huggingface extra, pip install 'rupsycho[huggingface]'")
        return

    from transformers.utils import logging as transformers_logging

    from rupsycho.models.model import LocalHuggingFaceModelConfig

    transformers_logging.set_verbosity_error()  # no "Device set to use cpu" for every load

    with tempfile.TemporaryDirectory(prefix="rupsycho-tiny-") as tmp:
        model_config = {
            "type": "local_huggingface",
            "name_or_path": str(build_tiny_model(Path(tmp))),
            "device_map": "cpu",
            "task": "text-generation",
            "parameters": {
                "max_new_tokens": 12,
                "do_sample": True,
                "temperature": 1.5,
                "top_k": 0,
                "return_full_text": False,
            },
        }

        def run(seeds: Sequence[str]) -> list[str]:
            config = new_config(seeds)
            config["models"] = {"tiny": model_config}
            return answers(rup.experiment_from_dict(config))

        first, again, other = run(SEEDS), run(SEEDS), run(OTHER_SEEDS)
        report.expect("same seeds, run twice", first == again, expected=True)
        report.expect("different seeds", first == other, expected=False)

        # What the seeding contributes: sample the same prompt directly, with and without it
        model = LocalHuggingFaceModelConfig(**model_config).load_model()  # type: ignore[arg-type]
        prompt = "Question: Does a thorough job\nAnswer:"
        unseeded = {model.invoke(prompt).content for _ in range(3)}
        seeded = {seed_model(model, 42).invoke(prompt).content for _ in range(3)}
        report.expect("3 samples, no seed applied", len(unseeded) == 1, expected=False)
        report.expect("3 samples, seed_model(model, 42)", len(seeded) == 1, expected=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checks; return 0 if all of them hold."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--skip-local", action="store_true", help="skip the Hugging Face model (part 3)"
    )
    args = parser.parse_args(argv)

    report = Report()
    check_seeded_api_model(report)
    check_unseedable_model(report)
    if not args.skip_local:
        check_local_model(report)

    print("All checks passed." if not report.failed else f"{report.failed} check(s) failed.")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
