# cli.py
"""Command-line interface of R.U.Psycho: the ``rupsycho`` command.

Sub-commands: ``run``, ``validate``, ``prompt``, ``examples``, ``postprocess`` and
``configurator``. Results go to stdout, diagnostics to stderr. Exit codes: 0 success, 1 error,
2 usage error, 3 the run finished but some model calls failed. Set ``RUPSYCHO_DEBUG=1`` to get
tracebacks instead of one-line error messages. The docstring of every ``_cmd_*`` function is
the ``--help`` text of its sub-command.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import glob
import importlib.util
import inspect
import json
import math
import os
import re
import sys
import textwrap
import warnings
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rupsycho import __version__

if TYPE_CHECKING:
    from rupsycho.experiment import ExperimentDocument
    from rupsycho.mixins.experiment_processing import RunSummary

__all__ = ["CLIError", "build_parser", "main"]

EXIT_OK, EXIT_ERROR, EXIT_FAILED_CALLS, EXIT_INTERRUPTED = 0, 1, 3, 130


class CLIError(Exception):
    """A problem the user can fix, reported as ``rupsycho: error: <message>`` with exit code 1."""


# ------------------------------------------------ diagnostics


def _say(message: str, level: str | None = None) -> None:
    """Write a diagnostic line to stderr (stdout is reserved for data)."""
    print(f"rupsycho: {level + ': ' if level else ''}{message}", file=sys.stderr)


def _show_warning(message: Warning | str, *_: Any, **__: Any) -> None:
    _say(str(message), "warning")


@contextlib.contextmanager
def _diagnostics() -> Iterator[None]:
    """Show library warnings as compact ``rupsycho: warning: ...`` lines on stderr."""
    with warnings.catch_warnings():
        warnings.showwarning = _show_warning
        # The run summary already reports failed calls
        warnings.filterwarnings("ignore", r"\d+ of \d+ model calls failed", RuntimeWarning)
        yield


# ------------------------------------------------ loading and checking configurations


def _summarise(exc: BaseException) -> str:
    """Condense an error raised while building an experiment into one line."""
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        first, *rest = exc.errors()
        where = ".".join(str(part) for part in first["loc"])
        if exc.title != "ExperimentDocument":
            where = ".".join(filter(None, [exc.title, where]))
        text = f"{where}: {first['msg']}" if where else first["msg"]
        return text + (f" (and {len(rest)} more error{'s' * (len(rest) > 1)})" if rest else "")
    text = (str(exc).strip().splitlines() or [""])[0]
    return text if isinstance(exc, ValueError) else f"{type(exc).__name__}: {text}"


def _read_config(path: str) -> dict[str, Any]:
    """Read a JSON configuration file."""
    if not Path(path).is_file():
        raise CLIError(f"{path}: no such file")
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise CLIError(f"{path}: cannot read the file ({exc.strerror})") from exc
    except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError
        raise CLIError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(config, dict):
        raise CLIError(f"{path}: the top level must be a JSON object")
    return config


def _items(experiment: ExperimentDocument) -> list[Any]:
    questionnaire = experiment.questionnaire
    return (questionnaire.instruction_items if questionnaire else None) or []


def _seeds(experiment: ExperimentDocument) -> list[Any]:
    from rupsycho.mixins.experiment_processing import DEFAULT_SEED

    return experiment.parameters.seeds or [DEFAULT_SEED]


def _describe(experiment: ExperimentDocument) -> str:
    """Size of the experiment grid: ``4 items x 2 personas x 1 models x 1 seeds = 8 calls``."""
    items, personas = len(_items(experiment)), len(experiment.demographic_profiles)
    models, seeds = len(experiment.models), len(_seeds(experiment))
    calls = math.prod((items, personas, models, seeds))
    return f"{items} items x {personas} personas x {models} models x {seeds} seeds = {calls} calls"


def _with_hints(keys: Sequence[str], known: Sequence[str]) -> str:
    """Quote ``keys``, each with the closest known key as a suggestion if there is one."""
    hints = (difflib.get_close_matches(key, known, 1) for key in keys)
    return ", ".join(
        f"'{key}'" + (f" (did you mean '{hint[0]}'?)" if hint else "")
        for key, hint in zip(keys, hints)
    )


def _load_experiment(
    path: str,
    *,
    models: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    need_models: bool = True,
) -> ExperimentDocument:
    """Load and check a configuration without loading any model; apply command-line overrides.

    Models are always loaded lazily, one at a time as the run reaches them, whatever
    ``parameters.lazy_load_models`` says. Validation and ``--dry-run`` thus never download
    anything and ``--model`` skips the models that are not wanted.
    """
    from rupsycho import experiment_from_dict
    from rupsycho.experiment import ExperimentDocument

    config = _read_config(path)
    known = list(ExperimentDocument.model_fields)
    if unknown := [key for key in config if key not in known]:
        # Ignoring the key would run the experiment with defaults (e.g. a small default model)
        raise CLIError(f"{path}: unknown top-level key(s): {_with_hints(unknown, known)}")
    parameters = config.get("parameters")
    if parameters is None or isinstance(parameters, dict):
        config["parameters"] = {**(parameters or {}), "lazy_load_models": True}
    try:
        experiment = experiment_from_dict(config)
    except Exception as exc:
        raise CLIError(f"{path}: {_summarise(exc)}") from exc

    missing = {
        "no questionnaire": experiment.questionnaire is None,
        "the questionnaire has no instruction_items": experiment.questionnaire is not None
        and not _items(experiment),
        "no demographic_profiles": not experiment.demographic_profiles,
        "no models": need_models and not experiment.models,
        "the prompt_template could not be loaded": experiment.runnable_prompt is None,
    }
    if problems := [text for text, present in missing.items() if present]:
        raise CLIError(f"{path}: {'; '.join(problems)}")

    if models:
        if absent := [name for name in models if not experiment.has_model(name)]:
            available = ", ".join(experiment.list_models())
            raise CLIError(f"unknown model id(s): {', '.join(absent)} (available: {available})")
        for name in experiment.list_models():
            if name not in models:
                experiment.remove_model(name)
    if seeds:
        experiment.parameters.seeds = [str(seed) for seed in dict.fromkeys(seeds)]
    return experiment


def _assemble_prompt(experiment: ExperimentDocument, item: int, persona: str) -> tuple[str, str]:
    """Return ``(heading, prompt)`` for an item index and a persona index or id."""
    items, ids = _items(experiment), list(experiment.demographic_profiles)
    if not 0 <= item < len(items):
        raise CLIError(f"--item {item} is out of range: {len(items)} items (0-{len(items) - 1})")
    if persona in ids:
        index = ids.index(persona)
    elif persona.isdecimal() and int(persona) < len(ids):
        index = int(persona)
    else:
        raise CLIError(
            f"--persona {persona!r} is out of range: {len(ids)} personas "
            f"(0-{len(ids) - 1}): {', '.join(ids)}"
        )
    heading = f"item {item}, persona {index} '{ids[index]}'"
    try:
        profile = experiment.demographic_profiles[ids[index]]
        inputs = experiment._create_input_dict(profile, items[item])
        return heading, experiment.runnable_prompt.format(**inputs)
    except Exception as exc:
        raise CLIError(f"cannot assemble the prompt of {heading}: {exc!r}") from exc


def _check_prompts(experiment: ExperimentDocument, path: str) -> None:
    """Make sure that every persona and every item yields a prompt, before any model is loaded."""
    personas, items = len(experiment.demographic_profiles), len(_items(experiment))
    try:
        for item, persona in [(0, p) for p in range(personas)] + [(i, 0) for i in range(1, items)]:
            _assemble_prompt(experiment, item, str(persona))
    except CLIError as exc:
        raise CLIError(f"{path}: {exc}") from None


def _check_output(value: str, allowed: tuple[str, ...], args: argparse.Namespace) -> Path:
    """Check an output file: supported extension, no accidental overwrite."""
    path = Path(value)
    if path.suffix.lower() not in allowed:
        args._parser.error(f"unsupported output format {path.suffix!r}; use {', '.join(allowed)}")
    if path.exists() and not args.force:
        raise CLIError(f"{path} already exists; use --force to overwrite it")
    return path


# ------------------------------------------------ commands


def _open_output(path: Path | None) -> list[Any]:
    """Prepare the output file and return the callbacks that stream answers into it."""
    if path is None:
        return []
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        return []  # written once the run has finished
    from rupsycho.callbacks import CSVCallback, JSONLCallback

    path.unlink(missing_ok=True)  # the callbacks append; overwriting was checked before
    return [(CSVCallback if path.suffix.lower() == ".csv" else JSONLCallback)(str(path))]


def _write_results(experiment: ExperimentDocument, path: Path | None) -> None:
    if path is None:
        print(experiment.get_answers_as_dataframe().to_string(index=False))
    elif path.suffix.lower() == ".json":
        experiment.export_to_file(path)


def _print_dry_run(experiment: ExperimentDocument, output: Path | None) -> None:
    heading, prompt = _assemble_prompt(experiment, 0, "0")
    print(
        "Dry run: no model was loaded and nothing was written.\n"
        f"experiment: {experiment.name or 'unnamed'}\n"
        f"models:     {', '.join(experiment.models)}\n"
        f"seeds:      {', '.join(map(str, _seeds(experiment)))}\n"
        f"calls:      {_describe(experiment)}\n"
        f"output:     {output or 'stdout (table)'}\n\n"
        f"Assembled prompt ({heading}):\n{prompt}"
    )


def _report(summary: RunSummary, output: Path | None, quiet: bool) -> None:
    line = str(summary)
    if summary.n_failed:
        line += f" (first error: {textwrap.shorten(summary.errors[0], 300)})"
    if output:
        line += f"; answers in {output}"
    if summary.n_failed or not quiet:
        _say(line)


def _cmd_run(args: argparse.Namespace) -> int:
    """Run an experiment and collect the answers.

    examples:
      rupsycho run config.json -o results.csv               stream the answers into a CSV file
      rupsycho run config.json --seeds 1 2 3 -o runs.jsonl  three repetitions as JSON lines
      rupsycho run config.json --model small --dry-run      preview the work, load no model
      rupsycho run config.json                              print the answers as a table

    .csv and .jsonl files grow while the run is going, .json is written at the end.
    exit codes: 0 success, 1 error, 2 usage error, 3 some model calls failed
    """
    output = _check_output(args.output, (".csv", ".jsonl", ".json"), args) if args.output else None
    experiment = _load_experiment(args.config, models=args.model, seeds=args.seeds)
    _check_prompts(experiment, args.config)
    if "seeds" not in experiment.parameters.model_fields_set and not args.quiet:
        _say(
            f"the configuration defines no seeds; using the random seed {_seeds(experiment)[0]} "
            "(pass --seeds to make the run reproducible)",
            "note",
        )
    if args.dry_run:
        _print_dry_run(experiment, output)
        return EXIT_OK

    summary = experiment.run(
        _open_output(output),
        args.cumulative,
        max_concurrency=args.max_concurrency,
        on_error=args.on_error,
        show_progress=not args.quiet,
    )
    _write_results(experiment, output)
    _report(summary, output, args.quiet)
    return EXIT_FAILED_CALLS if summary.n_failed else EXIT_OK


def _cmd_validate(args: argparse.Namespace) -> int:
    """Check configuration files without loading models.

    Prints one OK or ERROR line per file and exits with 1 if any file is invalid.

    examples:
      rupsycho validate config.json
      rupsycho validate configs/*.json
    """
    invalid = 0
    for path in args.configs:
        try:
            experiment = _load_experiment(path)
            _check_prompts(experiment, path)
            print(f"OK  {path}: {experiment.name or 'unnamed'} - {_describe(experiment)}")
        except CLIError as exc:
            invalid += 1
            print(f"ERROR {exc}")
    return EXIT_ERROR if invalid else EXIT_OK


def _cmd_prompt(args: argparse.Namespace) -> int:
    """Print the fully assembled prompt of one item and persona.

    examples:
      rupsycho prompt config.json
      rupsycho prompt config.json --item 3 --persona "Conservative Persona" > prompt.txt
    """
    experiment = _load_experiment(args.config, need_models=False)
    heading, prompt = _assemble_prompt(experiment, args.item, args.persona)
    _say(f"assembled prompt of {heading}")
    print(prompt)
    return EXIT_OK


def _example_text(name: str) -> str:
    from rupsycho.datasets import load_example_config

    try:
        config = load_example_config(name)
    except KeyError as exc:
        raise CLIError(str(exc.args[0])) from None
    return json.dumps(config, indent=4, ensure_ascii=False) + "\n"


def _cmd_examples_list(args: argparse.Namespace) -> int:
    """List the bundled examples."""
    from rupsycho.datasets import list_examples

    print("\n".join(list_examples()))
    return EXIT_OK


def _cmd_examples_show(args: argparse.Namespace) -> int:
    """Print an example configuration as JSON."""
    print(_example_text(args.name), end="")
    return EXIT_OK


def _cmd_examples_copy(args: argparse.Namespace) -> int:
    """Copy an example configuration to a file."""
    text = _example_text(args.name)
    dest = Path(args.dest)
    if dest.is_dir():
        dest = dest / f"{args.name}.json"
    if dest.exists() and not args.force:
        raise CLIError(f"{dest} already exists; use --force to overwrite it")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    _say(f"wrote {dest}")
    return EXIT_OK


def _make_cleaner(args: argparse.Namespace) -> Any:
    """The cleaner chosen by ``--cleaner`` / ``--pattern``; bad combinations are usage errors."""
    from rupsycho.parsers.cleaners import BasicCleaner, RegexExtractorCleaner

    kind = args.cleaner or ("regex" if args.pattern else "basic")
    if kind == "basic":
        if args.pattern:
            args._parser.error("--pattern only applies to --cleaner regex")
        return BasicCleaner()
    if not args.pattern:
        args._parser.error("--cleaner regex needs --pattern")
    try:
        groups = re.compile(args.pattern).groups
    except re.error as exc:
        args._parser.error(f"invalid --pattern: {exc}")
    if not groups:
        args._parser.error('--pattern needs a capturing group, e.g. \'answer:\\s*"([^"]*)"\'')
    return RegexExtractorCleaner(args.pattern)


def _cmd_postprocess(args: argparse.Namespace) -> int:
    """Clean, validate and score the answers of a run.

    Applies the rule-based defaults: a cleaner (basic or regex), the combined validator and the
    multiple-choice judge. Adds the columns cleaned_answer, validation_status, valid and decision.

    examples:
      rupsycho postprocess config.json results.csv -o scored.csv
      rupsycho postprocess config.json 'runs/*.csv' -o scored.csv --pattern 'answer:\\s*"([^"]*)"'
    """
    output = _check_output(args.output, (".csv",), args)
    cleaner = _make_cleaner(args)
    matches = {pattern: glob.glob(pattern, recursive=True) for pattern in args.results}
    if missing := [pattern for pattern, files in matches.items() if not files]:
        raise CLIError(f"no results file matches: {', '.join(missing)}")
    if output.resolve() in {Path(file).resolve() for files in matches.values() for file in files}:
        raise CLIError(f"{output} is also a results file; choose another output file")
    experiment = _load_experiment(args.config, need_models=False)

    from rupsycho.parsers.judges import MultipleChoiceJudge
    from rupsycho.parsers.validators import ValidatorParser
    from rupsycho.postprocessing import PostprocessingPipeline

    options = experiment.questionnaire.default_answer_options if experiment.questionnaire else None
    judge = MultipleChoiceJudge(possible_answers=options.get_options_as_list() if options else [])
    output.parent.mkdir(parents=True, exist_ok=True)
    pipeline = PostprocessingPipeline(
        args.config,
        args.results,
        cleaner,
        ValidatorParser(),
        judge,
        output_path=output,
        show_progress=not args.quiet,
    )
    processed = pipeline.run()
    if not args.quiet:
        _say(f"wrote {output} ({len(processed)} rows)")
    return EXIT_OK


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _cmd_configurator(args: argparse.Namespace) -> int:
    """Launch the Streamlit configurator app.

    Arguments after -- are passed on to streamlit, e.g. rupsycho configurator -- --server.port 8502
    Needs the extra: pip install 'rupsycho[configurator]'
    """
    if not _has_module("streamlit"):
        raise CLIError(
            "the configurator needs Streamlit; install it with: pip install 'rupsycho[configurator]'"
        )
    from rupsycho_configurator.launcher import main as launch

    options = args.streamlit_args[1:] if args.streamlit_args[:1] == ["--"] else args.streamlit_args
    argv, sys.argv = sys.argv, ["rup-configurator", *options]  # the launcher reads sys.argv
    try:
        result = launch()
    finally:
        sys.argv = argv
    return result if isinstance(result, int) else EXIT_OK


# ------------------------------------------------ argument parser


def _bounded_int(minimum: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid integer: {text!r}") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {value}")
        return value

    return parse


def _command(
    commands: Any, name: str, handler: Callable[[argparse.Namespace], int] | None, doc: str = ""
) -> argparse.ArgumentParser:
    """Add a sub-command whose help text is the docstring of its handler."""
    summary, _, epilog = inspect.cleandoc(doc or (handler and handler.__doc__) or name).partition(
        "\n\n"
    )
    parser: argparse.ArgumentParser = commands.add_parser(
        name,
        help=summary.rstrip("."),
        description=summary,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.set_defaults(_parser=parser, **({"handler": handler} if handler else {}))
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser of the ``rupsycho`` command.

    Returns:
        The parser; each sub-command stores its handler in ``args.handler``.

    Example:
        ```python
        args = build_parser().parse_args(["run", "config.json", "--seeds", "1", "2"])
        ```
    """
    epilog = """\
    examples:
      rupsycho examples copy bfi bfi.json    start from the bundled Big Five example
      rupsycho validate bfi.json             check the configuration, load no model
      rupsycho run bfi.json --dry-run        preview the prompt and the number of calls
      rupsycho run bfi.json -o results.csv   run it and stream the answers to a file
      rupsycho postprocess bfi.json results.csv -o scored.csv

    exit codes: 0 success, 1 error, 2 usage error, 3 some model calls failed
    set RUPSYCHO_DEBUG=1 to see tracebacks"""
    parser = argparse.ArgumentParser(
        prog="rupsycho",
        description="R.U.Psycho: robust, unified and reproducible psychometric testing of "
        "language models.",
        epilog=textwrap.dedent(epilog),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    add = _command(commands, "run", _cmd_run).add_argument
    add("config", metavar="CONFIG", help="experiment configuration (JSON file)")
    add("-o", "--output", metavar="FILE", help="write the answers to a .csv, .jsonl or .json file")
    add("-f", "--force", action="store_true", help="overwrite FILE if it exists")
    add("--model", nargs="+", action="extend", metavar="ID", help="run only these model ids")
    add("--seeds", nargs="+", type=_bounded_int(0), metavar="S", help="override parameters.seeds")
    add("--cumulative", action="store_true", help="let each persona remember its earlier answers")
    add("--max-concurrency", type=_bounded_int(1), default=1, metavar="N", help="calls at once")
    add("--on-error", choices=("warn", "raise", "ignore"), default="warn", help="if a call fails")
    add("--dry-run", action="store_true", help="load no model; show the call count and a prompt")
    add("-q", "--quiet", action="store_true", help="hide the progress bar and the summary")

    add = _command(commands, "validate", _cmd_validate).add_argument
    add("configs", nargs="+", metavar="CONFIG", help="configuration files (JSON)")

    add = _command(commands, "prompt", _cmd_prompt).add_argument
    add("config", metavar="CONFIG", help="experiment configuration (JSON file)")
    add("--item", type=int, default=0, metavar="I", help="item index (default: 0)")
    add("--persona", default="0", metavar="P", help="persona index or id (default: 0)")

    group = """\
    List, show and copy the bundled example configurations.

    examples:
      rupsycho examples list
      rupsycho examples show bfi
      rupsycho examples copy bfi my-experiment.json"""
    actions = _command(commands, "examples", None, group).add_subparsers(
        dest="action", metavar="ACTION", required=True
    )
    _command(actions, "list", _cmd_examples_list)
    _command(actions, "show", _cmd_examples_show).add_argument(
        "name", metavar="NAME", help="name of the example"
    )
    add = _command(actions, "copy", _cmd_examples_copy).add_argument
    add("name", metavar="NAME", help="name of the example")
    add("dest", metavar="DEST", help="target file or directory")
    add("-f", "--force", action="store_true", help="overwrite DEST if it exists")

    add = _command(commands, "postprocess", _cmd_postprocess).add_argument
    add("config", metavar="CONFIG", help="the configuration the results come from")
    add("results", nargs="+", metavar="RESULTS", help="CSV files of 'run -o' (globs allowed)")
    add("-o", "--output", required=True, metavar="OUT", help="processed CSV file")
    add("--cleaner", choices=("basic", "regex"), help="basic (default) or regex")
    add("--pattern", metavar="REGEX", help="regex with one capturing group; implies regex")
    add("-f", "--force", action="store_true", help="overwrite OUT if it exists")
    add("-q", "--quiet", action="store_true", help="hide the progress bar and the summary")

    add = _command(commands, "configurator", _cmd_configurator).add_argument
    add("streamlit_args", nargs=argparse.REMAINDER, metavar="ARGS", help="options for streamlit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``rupsycho`` command.

    Args:
        argv: Command-line arguments without the program name (default: ``sys.argv[1:]``).

    Returns:
        The exit code: 0 success, 1 error, 3 the run finished but some model calls failed,
        130 interrupted. Usage errors and ``--help`` / ``--version`` raise ``SystemExit`` like
        any argparse program.

    Example:
        ```python
        from rupsycho.cli import main

        exit_code = main(["validate", "config.json"])
        ```
    """
    args = build_parser().parse_args(argv)
    try:
        with _diagnostics():
            return int(args.handler(args))
    except KeyboardInterrupt:
        _say("interrupted", "error")
        return EXIT_INTERRUPTED
    except Exception as exc:
        if os.environ.get("RUPSYCHO_DEBUG"):
            raise
        if isinstance(exc, (CLIError, ValueError, OSError, ImportError)):
            _say(str(exc) or type(exc).__name__, "error")
        else:
            _say(f"{type(exc).__name__}: {exc}", "error")
            _say("set RUPSYCHO_DEBUG=1 to see the full traceback")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
