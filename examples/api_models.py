"""Configure models served over an API: OpenAI, Ollama, Google Gemini and DeepSeek.

The script documents how the four back-ends are configured and which environment variables
they read. By default it **sends nothing over the network**: it validates the configuration
snippets with rupsycho (no model is created, so no key is needed), prints them, and reports
which extras and keys are present on this machine::

    python examples/api_models.py

To run the bundled Big Five example against one provider (needs the extra, the key where one is
required and network access; hosted models cost a few cents for this small example)::

    python examples/api_models.py --run openai --max-concurrency 8

Configuration style
-------------------
A model is one entry of the ``models`` section of the experiment configuration::

    "models": {"gpt-4o-mini": {"type": "openai", "name_or_path": "gpt-4o-mini", "parameters": {}}}

Code style
----------
Or add any LangChain model to the experiment in code, which is handy for models that have no
configuration type::

    experiment.add_model(ChatOpenAI(model="gpt-4o-mini"), identifier="gpt-4o-mini")

Keys
----
Keep API keys in the **environment**, not in the configuration: configurations are meant to be
shared with a paper or a repository. A model section without ``api_key`` reads the key from the
variable of its provider (see the table printed by this script). A section may carry an
``api_key`` too; rupsycho masks it in ``repr`` and in exports, but the configuration file you
wrote would still contain it in plain text.

Hosted models spend their time waiting, so ``experiment.run(max_concurrency=8)`` runs several
calls at once. OpenAI, DeepSeek and Ollama accept the experiment seeds (``seed`` parameter),
Gemini cannot be seeded and rupsycho warns about it. Hugging Face Inference Endpoints
(``"type": "remote_huggingface"``) work the same way and read ``HUGGINGFACEHUB_API_TOKEN``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import rupsycho as rup


@dataclass(frozen=True)
class Provider:
    """How one back-end is configured."""

    name: str
    identifier: str  # id of the model in the results
    extra: str  # the rupsycho extra that installs the LangChain integration
    package: str  # the module that must be importable
    env_var: str | None  # environment variable that holds the key (None: no key needed)
    seeds: str  # how the experiment seeds reach the model
    model: dict[str, Any]  # the section for the "models" key of the configuration
    code: str  # the same model added in code


PROVIDERS: dict[str, Provider] = {
    provider.name: provider
    for provider in (
        Provider(
            name="openai",
            identifier="gpt-4o-mini",
            extra="openai",
            package="langchain_openai",
            env_var="OPENAI_API_KEY",
            seeds="seed parameter",
            # For OpenAI-compatible servers (vLLM, LM Studio, ...) add "base_url": "http://.../v1"
            model={
                "type": "openai",
                "name_or_path": "gpt-4o-mini",
                "parameters": {"temperature": 0.7, "max_tokens": 64},
            },
            code=(
                "from langchain_openai import ChatOpenAI\n"
                "experiment.add_model(ChatOpenAI(model='gpt-4o-mini', temperature=0.7),"
                " identifier='gpt-4o-mini')"
            ),
        ),
        Provider(
            name="ollama",
            identifier="llama3.2:3b",
            extra="ollama",
            package="langchain_ollama",
            env_var=None,  # a local server needs no key
            seeds="seed parameter",
            model={
                "type": "ollama",
                "model": "llama3.2:3b",  # fetch it first: ollama pull llama3.2:3b
                # OLLAMA_BASE_URL is read by this script only; write the address into your own
                # configuration or read the variable in your own code
                "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                "parameters": {"temperature": 0.7, "num_predict": 64},
            },
            code=(
                "from langchain_ollama import OllamaLLM\n"
                "experiment.add_model(OllamaLLM(model='llama3.2:3b', temperature=0.7),"
                " identifier='llama3.2:3b')"
            ),
        ),
        Provider(
            name="google",
            identifier="gemini-2.0-flash",
            extra="google",
            package="langchain_google_genai",
            env_var="GOOGLE_API_KEY",
            seeds="not supported (rupsycho warns)",
            model={
                "type": "google",
                "name_or_path": "gemini-2.0-flash",
                "parameters": {"temperature": 0.7, "max_output_tokens": 64},
            },
            code=(
                "from langchain_google_genai import ChatGoogleGenerativeAI\n"
                "experiment.add_model(ChatGoogleGenerativeAI(model='gemini-2.0-flash'),"
                " identifier='gemini-2.0-flash')"
            ),
        ),
        Provider(
            name="deepseek",
            identifier="deepseek-chat",
            extra="deepseek",
            package="langchain_deepseek",
            env_var="DEEPSEEK_API_KEY",
            seeds="seed parameter",
            model={
                "type": "deepseek",
                "name_or_path": "deepseek-chat",
                "parameters": {"temperature": 0.7, "max_tokens": 64},
            },
            code=(
                "from langchain_deepseek import ChatDeepSeek\n"
                "experiment.add_model(ChatDeepSeek(model='deepseek-chat'),"
                " identifier='deepseek-chat')"
            ),
        ),
    )
}


def installed(provider: Provider) -> bool:
    """Whether the LangChain integration of ``provider`` can be imported."""
    return importlib.util.find_spec(provider.package) is not None


def key_is_set(provider: Provider) -> bool | None:
    """Whether the key variable is set (``None`` if the provider needs no key)."""
    return None if provider.env_var is None else bool(os.environ.get(provider.env_var))


def configuration_with(provider: Provider) -> dict[str, Any]:
    """The bundled Big Five example whose only model is the one of ``provider``."""
    config = rup.load_example_config("bfi")
    config["models"] = {provider.identifier: json.loads(json.dumps(provider.model))}
    return config


def describe() -> int:
    """Validate the snippets and report what is available. Never touches the network."""
    print("Models served over an API (nothing is sent over the network)\n")
    print(f"{'provider':<9} {'extra':<9} {'installed':<10} {'key':<27} seeds")
    for provider in PROVIDERS.values():
        # Validating builds the configuration objects only. Models are created when the run
        # reaches them, so this works without the extra and without a key.
        rup.experiment_from_dict(configuration_with(provider))
        key = key_is_set(provider)
        key_text = (
            "not needed" if key is None else f"{provider.env_var}: {'set' if key else 'not set'}"
        )
        print(
            f"{provider.name:<9} {provider.extra:<9} {'yes' if installed(provider) else 'no':<10} "
            f"{key_text:<27} {provider.seeds}"
        )

    for provider in PROVIDERS.values():
        print(f"\n--- {provider.name}: pip install 'rupsycho[{provider.extra}]'")
        print('"models": ' + json.dumps({provider.identifier: provider.model}, indent=2))
        print("# or, in code:")
        print(provider.code)
    print("\nAll configuration snippets are valid. Use --run PROVIDER to try one (needs network).")
    return 0


def run(name: str, max_concurrency: int) -> int:
    """Run the bundled example against one provider. Needs network access."""
    provider = PROVIDERS[name]
    problems = []
    if not installed(provider):
        problems.append(f"the extra is missing: pip install 'rupsycho[{provider.extra}]'")
    if key_is_set(provider) is False:
        problems.append(f"set the environment variable {provider.env_var}")
    if problems:
        print(f"Cannot run {name}: " + "; ".join(problems))
        return 2

    experiment = rup.experiment_from_dict(configuration_with(provider))
    summary = experiment.run(max_concurrency=max_concurrency)
    print(summary)
    print(experiment.get_answers_as_dataframe()[["Instruction ID", "Persona ID", "Answer"]])
    return 3 if summary.n_failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Describe the providers, or run the example against one."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run", choices=sorted(PROVIDERS), help="run the bundled example against this provider"
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=4, help="parallel calls for --run (default: 4)"
    )
    args = parser.parse_args(argv)
    return run(args.run, args.max_concurrency) if args.run else describe()


if __name__ == "__main__":
    raise SystemExit(main())
