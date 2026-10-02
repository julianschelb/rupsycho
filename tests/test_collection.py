"""Tests for ``ExperimentCollection`` and for loading experiments (``rupsycho``, ``reader``)."""

import asyncio
import copy
import glob
import json
import logging
import sys
import types
import warnings
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

import rupsycho as rup
import rupsycho.reader as reader
from rupsycho.callbacks import Callback
from rupsycho.experiment import ExperimentDocument
from rupsycho.experiment_collection import ExperimentCollection
from rupsycho.mixins.experiment_processing import RunSummary
from rupsycho.reader import ExperimentLoader

from .conftest import CONFIG_PATH, DATA_DIR

LOGGER = "rupsycho.reader"


def named(config_dict, name, **changes):
    config = copy.deepcopy(config_dict)
    config["name"] = name
    config.update(changes)
    return config


def write_config(path, config):
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.fixture
def config_dir(tmp_path, config_dict):
    """A directory with valid configs a.json, b.json, sub/c.json and two broken files."""
    write_config(tmp_path / "a.json", named(config_dict, "A"))
    write_config(tmp_path / "b.json", named(config_dict, "B"))
    (tmp_path / "sub").mkdir()
    write_config(tmp_path / "sub" / "c.json", named(config_dict, "C"))
    write_config(
        tmp_path / "invalid.json", {"parameters": {}, "questionnaire": "not a questionnaire"}
    )
    (tmp_path / "garbage.json").write_text("{this is not json", encoding="utf-8")
    return tmp_path


def names(experiments):
    return sorted(experiment.name for experiment in experiments)


def warnings_of_the_reader(caplog):
    """The messages logged by the reader (one line per skipped experiment)."""
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelno >= 30]


class BrokenLLM(FakeListLLM):
    """A fake LLM whose every call fails."""

    def _call(self, *args, **kwargs):
        raise RuntimeError("boom")


class Recorder(Callback):
    """Remembers which experiment produced which answer."""

    def __init__(self):
        self.seen = []

    def save_answer(self, experiment, item_id, item, model_id, profile_id, seed, time, answer):
        self.seen.append((experiment.name, answer))


# ------------------------------------------------------------------------------------------
# ExperimentCollection
# ------------------------------------------------------------------------------------------


@pytest.fixture
def two_experiments(config_dict):
    first = rup.experiment_from_dict(named(config_dict, "A"))
    second = rup.experiment_from_dict(named(config_dict, "B", description=None))
    return first, second


def test_collection_holds_the_given_experiments(two_experiments):
    first, second = two_experiments

    collection = ExperimentCollection(experiments=[first, second])

    assert len(collection) == 2
    assert collection.experiments[0] is first
    assert collection.experiments[1] is second
    assert collection.type == "ExperimentCollection"
    assert collection.metadata == {}


def test_collection_accepts_experiments_positionally_and_metadata(two_experiments):
    collection = ExperimentCollection(list(two_experiments), metadata={"source": "Lab"})

    assert len(collection) == 2
    assert collection.metadata == {"source": "Lab"}


def test_collection_string_lists_experiments_and_metadata(two_experiments):
    collection = ExperimentCollection(list(two_experiments), metadata={"source": "Lab"})

    assert str(collection) == (
        "experiments=["
        "name=A, description=Description: Testing Generative Models for BFI questionnaire "
        "using Rupsycho., metadata={}, "
        "name=B, description=None, metadata={}"
        "], metadata={'source': 'Lab'}"
    )


def test_empty_collection():
    collection = ExperimentCollection([])

    assert len(collection) == 0
    assert str(collection) == "experiments=[], metadata={}"
    assert collection.run_all(show_progress=False) == []  # nothing to do


def test_collection_requires_valid_experiments():
    with pytest.raises(ValidationError):
        ExperimentCollection(experiments=["not an experiment"])
    with pytest.raises(TypeError):
        ExperimentCollection()


def test_collection_is_langchain_serialisable_in_its_own_namespace():
    assert ExperimentCollection.is_lc_serializable() is True
    assert ExperimentCollection.get_lc_namespace() == [
        "langchain",
        "schema",
        "experiment_collection",
    ]


def test_run_all_runs_every_experiment(two_experiments):
    first, second = two_experiments
    first.add_model(FakeListLLM(responses=["from-A"]), identifier="m")
    second.add_model(FakeListLLM(responses=["from-B"]), identifier="m")
    collection = ExperimentCollection([first, second])

    collection.run_all(show_progress=False)

    assert set(first.get_answers_as_dataframe()["Answer"]) == {"from-A"}
    assert set(second.get_answers_as_dataframe()["Answer"]) == {"from-B"}
    assert len(first.get_answers_as_dataframe()) == len(second.get_answers_as_dataframe()) == 8


def test_run_all_returns_one_summary_per_experiment(two_experiments):
    first, second = two_experiments
    first.add_model(FakeListLLM(responses=["x"]), identifier="m")
    second.add_model(FakeListLLM(responses=["x"]), identifier="m")
    second.add_persona(first.get_persona("Optimistic Persona"), identifier="third")

    summaries = ExperimentCollection([first, second]).run_all(show_progress=False)

    assert all(isinstance(summary, RunSummary) for summary in summaries)
    assert [(s.n_calls, s.n_failed) for s in summaries] == [(8, 0), (12, 0)]


def test_run_all_runs_the_experiments_in_order(two_experiments):
    order = []

    class OrderedLLM(FakeListLLM):
        def _call(self, prompt, stop=None, run_manager=None, **kwargs):
            order.append(self.responses[0])
            return super()._call(prompt, stop=stop, run_manager=run_manager, **kwargs)

    first, second = two_experiments
    first.add_model(OrderedLLM(responses=["A"]), identifier="m")
    second.add_model(OrderedLLM(responses=["B"]), identifier="m")

    ExperimentCollection([first, second]).run_all(show_progress=False)

    assert order == ["A"] * 8 + ["B"] * 8


def test_run_all_passes_its_keyword_arguments_to_every_run(two_experiments):
    first, second = two_experiments
    first.add_model(FakeListLLM(responses=["from-A"]), identifier="m")
    second.add_model(FakeListLLM(responses=["from-B"]), identifier="m")
    recorder = Recorder()

    ExperimentCollection([first, second]).run_all(
        callbacks=[recorder], max_concurrency=2, on_error="ignore", show_progress=False
    )

    assert recorder.seen == [("A", "from-A")] * 8 + [("B", "from-B")] * 8


class PromptLog(FakeListLLM):
    """A fake LLM that keeps the prompts it received."""

    prompts: list = []

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        self.prompts.append(prompt)
        return super()._call(prompt, stop=stop, run_manager=run_manager, **kwargs)


def test_run_all_can_run_the_experiments_cumulatively(two_experiments):
    plain, cumulative = (
        PromptLog(responses=["ans"], prompts=[]),
        PromptLog(responses=["ans"], prompts=[]),
    )
    first, second = two_experiments
    first.add_model(plain, identifier="m")
    second.add_model(cumulative, identifier="m")

    ExperimentCollection([first]).run_all(show_progress=False)
    ExperimentCollection([second]).run_all(cumulative=True, show_progress=False)

    assert not any("Answer: ans" in prompt for prompt in plain.prompts)  # no memory
    assert any("Answer: ans" in prompt for prompt in cumulative.prompts)  # earlier answers are in


def test_run_all_applies_the_error_policy_to_every_experiment(two_experiments):
    broken, working = two_experiments
    broken.add_model(BrokenLLM(responses=["x"]), identifier="m")
    working.add_model(FakeListLLM(responses=["ok"]), identifier="m")

    with pytest.warns(RuntimeWarning, match="8 of 8 model calls failed"):
        summaries = ExperimentCollection([broken, working]).run_all(show_progress=False)

    assert [(s.n_calls, s.n_failed) for s in summaries] == [(8, 8), (8, 0)]
    assert len(working.get_answers_as_dataframe()) == 8  # the failure did not stop the collection


def test_run_all_stops_at_the_first_failure_with_the_raise_policy(two_experiments):
    broken, working = two_experiments
    broken.add_model(BrokenLLM(responses=["x"]), identifier="m")
    working.add_model(FakeListLLM(responses=["ok"]), identifier="m")

    with pytest.raises(RuntimeError, match="boom"):
        ExperimentCollection([broken, working]).run_all(on_error="raise", show_progress=False)

    assert len(working.get_answers_as_dataframe()) == 0


def test_run_all_propagates_a_missing_model(two_experiments):
    first, second = two_experiments
    second.add_model(FakeListLLM(responses=["B"]), identifier="m")

    with pytest.raises(ValueError, match="No models have been set in runnable_models."):
        ExperimentCollection([first, second]).run_all(show_progress=False)

    assert len(second.get_answers_as_dataframe()) == 0  # the failure stopped the collection


def test_experiments_in_a_collection_are_independent(config_dict):
    one, two = (rup.experiment_from_dict(copy.deepcopy(config_dict)) for _ in range(2))
    one.add_model(FakeListLLM(responses=["x"]), identifier="m")

    one.run(show_progress=False)

    assert len(one.get_answers_as_dataframe()) == 8
    assert len(two.get_answers_as_dataframe()) == 0
    assert two.get_answers() == [{}, {}, {}, {}]


def test_package_exposes_the_loading_api():
    assert rup.ExperimentDocument is ExperimentDocument
    assert rup.ExperimentCollection is ExperimentCollection
    assert rup.ExperimentLoader is ExperimentLoader
    assert rup.RunSummary is RunSummary
    for name in (
        "experiment_from_dict",
        "experiment_from_file",
        "experiments_from_dicts",
        "experiments_from_files",
        "example_experiment_bfi",
    ):
        assert callable(getattr(rup, name))


# ------------------------------------------------------------------------------------------
# experiment_from_dict / experiments_from_dicts
# ------------------------------------------------------------------------------------------


def test_experiment_from_dict_builds_a_document(config_dict):
    experiment = rup.experiment_from_dict(copy.deepcopy(config_dict))

    assert isinstance(experiment, ExperimentDocument)
    assert experiment.name == "Generative Models for Big Five Inventory"
    assert experiment.description == config_dict["description"]
    assert experiment.parameters.seeds == ["7"]
    assert list(experiment.demographic_profiles) == ["Optimistic Persona", "Conservative Persona"]
    assert experiment.questionnaire.get_number_of_questions() == 4
    assert experiment.list_models() == []


def test_unknown_top_level_keys_are_ignored_without_a_warning(config_dict):
    """docs/configuration.md: unknown keys are ignored, a misspelt key only shows up later."""
    config = copy.deepcopy(config_dict)
    config["no_such_key"] = {"anything": 1}

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        experiment = rup.experiment_from_dict(config)

    assert not hasattr(experiment, "no_such_key")
    assert experiment.name == config_dict["name"]


def test_experiment_from_dict_does_not_modify_its_input(config_dict):
    config = copy.deepcopy(config_dict)
    experiment = rup.experiment_from_dict(config)
    experiment.add_model(FakeListLLM(responses=["3"]), identifier="m")
    experiment.run(show_progress=False)

    assert config == config_dict


def test_experiments_from_the_same_dict_are_independent(config_dict):
    first, second = rup.experiments_from_dicts([config_dict, config_dict])
    first.add_model(FakeListLLM(responses=["3"]), identifier="m")

    first.run(show_progress=False)

    assert len(first.get_answers_as_dataframe()) == 8
    assert len(second.get_answers_as_dataframe()) == 0


def test_a_configuration_without_parameters_is_valid(config_dict):
    config = {k: v for k, v in config_dict.items() if k != "parameters"}

    experiment = rup.experiment_from_dict(config)

    assert len(experiment.parameters.seeds) == 1  # a seed is drawn for this experiment
    assert experiment.parameters.lazy_load_models is True


def test_experiment_from_dict_raises_a_value_error_that_names_the_problem(capsys):
    config = {"parameters": {}, "questionnaire": "not a questionnaire"}

    with pytest.raises(ValueError, match="'questionnaire' must be an object"):
        rup.experiment_from_dict(config)

    assert capsys.readouterr().out == ""  # the error is raised, not printed


def test_experiment_from_dict_raises_pydantic_validation_errors(config_dict):
    config = {**copy.deepcopy(config_dict), "questionnaire": {"name": 5}}

    with pytest.raises(ValidationError) as error:
        rup.experiment_from_dict(config)

    assert isinstance(error.value, ValueError)
    assert "general_instruction" in str(error.value) and "Field required" in str(error.value)


def test_experiment_from_dict_rejects_an_unknown_model_type():
    with pytest.raises(ValueError, match="Unknown or missing model type for key: x"):
        rup.experiment_from_dict({"parameters": {}, "models": {"x": {"type": "no-such-type"}}})


def test_experiment_from_dict_rejects_seeds_that_are_no_integers(config_dict):
    config = {**copy.deepcopy(config_dict), "parameters": {"seeds": ["seven"]}}

    with pytest.raises(ValidationError, match="seeds must be integers"):
        rup.experiment_from_dict(config)


def test_experiments_from_dicts_returns_a_generator_that_builds_lazily(config_dict, monkeypatch):
    built = []
    original = reader._experiment_from_json

    def spy(json_data):
        built.append(json_data["name"])
        return original(json_data)

    monkeypatch.setattr(reader, "_experiment_from_json", spy)

    experiments = rup.experiments_from_dicts([named(config_dict, "A"), named(config_dict, "B")])

    assert isinstance(experiments, types.GeneratorType)
    assert built == []
    assert next(experiments).name == "A"
    assert built == ["A"]
    assert next(experiments).name == "B"
    assert built == ["A", "B"]
    assert list(experiments) == []


def test_experiments_from_dicts_skips_invalid_entries_and_logs_them(config_dict, caplog, capsys):
    configs = [
        named(config_dict, "first"),
        {"parameters": {}, "questionnaire": "broken"},
        {"parameters": {}, "models": {"x": {"type": "no-such-type"}}},
        {"questionnaire": {"name": 5}},
        named(config_dict, "last"),
    ]

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        experiments = list(rup.experiments_from_dicts(configs))

    assert [e.name for e in experiments] == ["first", "last"]
    first, second, third = warnings_of_the_reader(caplog)
    assert first.startswith("Cannot read dictionary #1: 'questionnaire' must be an object")
    assert second == "Cannot read dictionary #2: Unknown or missing model type for key: x"
    assert third.startswith("Invalid experiment dictionary #3: ")  # pydantic's ValidationError
    assert capsys.readouterr().out == ""


def test_experiments_from_dicts_skips_entries_that_are_no_dictionaries(config_dict, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        experiments = list(rup.experiments_from_dicts([[], named(config_dict, "ok"), None]))

    assert [e.name for e in experiments] == ["ok"]
    assert len(warnings_of_the_reader(caplog)) == 2


def test_experiments_from_dicts_in_strict_mode_raises_at_the_first_invalid_entry(config_dict):
    configs = [named(config_dict, "first"), {"questionnaire": "broken"}, named(config_dict, "last")]

    experiments = rup.experiments_from_dicts(configs, strict=True)

    assert next(experiments).name == "first"
    with pytest.raises(ValueError, match="'questionnaire' must be an object"):
        next(experiments)


def test_experiments_from_dicts_of_nothing_is_empty():
    assert list(rup.experiments_from_dicts([])) == []


# ------------------------------------------------------------------------------------------
# experiment_from_file / experiments_from_files
# ------------------------------------------------------------------------------------------


def test_experiment_from_file_loads_the_configuration():
    experiment = rup.experiment_from_file(str(CONFIG_PATH))

    assert experiment.name == "Generative Models for Big Five Inventory"
    assert list(experiment.models) == ["Qwen/Qwen2.5-0.5B-Instruct"]  # lazily loaded, not loaded
    assert experiment.questionnaire.get_number_of_questions() == 4


def test_experiment_from_file_accepts_a_path_object():
    assert rup.experiment_from_file(CONFIG_PATH).name == "Generative Models for Big Five Inventory"


def test_experiment_from_file_reads_utf8(tmp_path, config_dict):
    path = tmp_path / "umlauts.json"
    path.write_bytes(
        json.dumps(named(config_dict, "Größe – 日本語"), ensure_ascii=False).encode("utf-8")
    )

    assert rup.experiment_from_file(path).name == "Größe – 日本語"


def test_experiment_from_file_does_not_expand_patterns(config_dir):
    """A pattern is not a file: use experiments_from_files to load several."""
    with pytest.raises(FileNotFoundError, match=r"\[ab\]\.json"):
        rup.experiment_from_file(str(config_dir / "[ab].json"))


def test_experiment_from_file_of_an_invalid_file_raises_and_prints_nothing(config_dir, capsys):
    with pytest.raises(ValueError, match="'questionnaire' must be an object"):
        rup.experiment_from_file(str(config_dir / "invalid.json"))

    assert capsys.readouterr().out == ""


def test_experiment_from_file_of_invalid_field_values_raises_a_validation_error(tmp_path):
    path = write_config(tmp_path / "bad.json", {"questionnaire": {"name": 5}})

    with pytest.raises(ValidationError, match=r"validation errors? for Questionnaire"):
        rup.experiment_from_file(path)


def test_experiment_from_file_of_broken_json_raises_a_value_error(config_dir):
    with pytest.raises(ValueError, match="Expecting property name"):
        rup.experiment_from_file(str(config_dir / "garbage.json"))


def test_experiment_from_file_of_a_missing_file_raises_file_not_found(tmp_path):
    path = str(tmp_path / "missing.json")

    with pytest.raises(FileNotFoundError, match="missing.json"):
        rup.experiment_from_file(path)


def test_experiment_from_file_of_a_directory_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        rup.experiment_from_file(tmp_path)


@pytest.mark.parametrize("content", ["[]", "5", '"text"', "null"])
def test_experiment_from_file_of_json_that_is_no_object_raises_a_value_error(tmp_path, content):
    path = tmp_path / "not-an-object.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        rup.experiment_from_file(path)


def test_experiments_from_files_returns_a_collection(config_dir):
    collection = rup.experiments_from_files(str(config_dir / "*.json"))

    assert isinstance(collection, ExperimentCollection)
    assert names(collection.experiments) == ["A", "B"]
    assert len(collection) == 2


def test_experiments_from_files_with_a_single_file_path(config_dir):
    collection = rup.experiments_from_files(str(config_dir / "a.json"))

    assert names(collection.experiments) == ["A"]


def test_experiments_from_files_with_a_recursive_pattern(config_dir):
    collection = rup.experiments_from_files(str(config_dir / "**" / "*.json"))

    assert names(collection.experiments) == ["A", "B", "C"]


def test_experiments_from_files_with_a_sub_directory_pattern(config_dir):
    collection = rup.experiments_from_files(str(config_dir / "sub" / "*.json"))

    assert names(collection.experiments) == ["C"]


def test_experiments_from_files_with_a_character_range_pattern(config_dir):
    collection = rup.experiments_from_files(str(config_dir / "[b-c]*.json"))

    assert names(collection.experiments) == ["B"]


def test_experiments_from_files_orders_the_experiments_by_file_name(tmp_path, config_dict):
    for file_name in ("c", "a", "b"):
        write_config(tmp_path / f"{file_name}.json", named(config_dict, file_name.upper()))

    collection = rup.experiments_from_files(str(tmp_path / "*.json"))

    assert [e.name for e in collection.experiments] == ["A", "B", "C"]


def test_experiments_from_files_skips_invalid_files_and_logs_them(config_dir, caplog, capsys):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        collection = rup.experiments_from_files(str(config_dir / "*.json"))

    assert names(collection.experiments) == ["A", "B"]
    garbage, invalid = warnings_of_the_reader(caplog)  # in file-name order
    assert garbage.startswith(f"Cannot read {config_dir / 'garbage.json'}: Expecting property name")
    assert invalid.startswith(f"Cannot read {config_dir / 'invalid.json'}: 'questionnaire' must be")
    assert capsys.readouterr().out == ""


def test_experiments_from_files_with_only_invalid_files_gives_an_empty_collection(
    config_dir, caplog
):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        collection = rup.experiments_from_files(str(config_dir / "[ig]*.json"))

    assert len(collection) == 0
    assert len(warnings_of_the_reader(caplog)) == 2


def test_experiments_from_files_in_strict_mode_raises_at_the_first_invalid_file(config_dir):
    with pytest.raises(ValueError, match="Expecting property name"):  # garbage.json comes first
        rup.experiments_from_files(str(config_dir / "*.json"), strict=True)


@pytest.mark.parametrize("pattern", ["nothing-here-*.json", "missing.json", "sub/zzz/*.json", ""])
def test_experiments_from_files_without_a_match_raises_value_error(config_dir, pattern):
    path_pattern = str(config_dir / pattern) if pattern else pattern

    with pytest.raises(ValueError, match="No experiment files match") as error:
        rup.experiments_from_files(path_pattern)

    assert repr(path_pattern) in str(error.value)


def test_experiments_from_files_of_the_test_data_directory():
    collection = rup.experiments_from_files(str(DATA_DIR / "*.json"))

    assert len(collection) >= 1
    assert "Generative Models for Big Five Inventory" in names(collection.experiments)


def test_loaded_collection_runs_with_fake_models(config_dir):
    collection = rup.experiments_from_files(str(config_dir / "[ab].json"))
    for experiment in collection.experiments:
        experiment.add_model(FakeListLLM(responses=[experiment.name]), identifier="m")

    summaries = collection.run_all(show_progress=False)

    assert [s.n_failed for s in summaries] == [0, 0]
    for experiment in collection.experiments:
        assert set(experiment.get_answers_as_dataframe()["Answer"]) == {experiment.name}


# ------------------------------------------------------------------------------------------
# ExperimentLoader and the reader helpers
# ------------------------------------------------------------------------------------------


def test_loader_resolves_paths(config_dir):
    assert ExperimentLoader(str(config_dir / "a.json")).file_paths == [str(config_dir / "a.json")]
    assert ExperimentLoader(str(config_dir / "*.json")).file_paths == [
        str(config_dir / name) for name in ("a.json", "b.json", "garbage.json", "invalid.json")
    ]
    assert ExperimentLoader(str(config_dir / "**" / "c.json")).file_paths == [
        str(config_dir / "sub" / "c.json")
    ]
    assert ExperimentLoader(str(config_dir / "nothing*")).file_paths == []
    assert ExperimentLoader().file_paths is None
    assert ExperimentLoader("").file_paths is None
    assert ExperimentLoader(None).file_paths is None


def test_loader_returns_matching_files_in_a_deterministic_order(monkeypatch):
    monkeypatch.setattr(
        glob, "glob", lambda pattern, recursive=False: ["c.json", "a.json", "b.json"]
    )

    assert ExperimentLoader("*.json").file_paths == ["a.json", "b.json", "c.json"]


def test_loader_load_returns_a_list_of_documents(config_dir):
    documents = ExperimentLoader(str(config_dir / "[ab].json")).load()

    assert isinstance(documents, list)
    assert names(documents) == ["A", "B"]
    assert all(isinstance(document, ExperimentDocument) for document in documents)


def test_loader_lazy_load_is_lazy(config_dir, monkeypatch):
    opened = []
    original = reader._experiment_from_file

    def spy(path):
        opened.append(path)
        return original(path)

    monkeypatch.setattr(reader, "_experiment_from_file", spy)
    iterator = ExperimentLoader(str(config_dir / "[ab].json")).lazy_load()

    assert opened == []
    next(iterator)
    assert len(opened) == 1


def test_loader_lazy_load_prefers_the_pattern_given_to_the_call(config_dir):
    loader = ExperimentLoader(str(config_dir / "a.json"))

    documents = list(loader.lazy_load(str(config_dir / "b.json")))

    assert names(documents) == ["B"]


def test_loader_lazy_load_without_any_pattern_raises():
    with pytest.raises(ValueError, match="No file paths provided. Please provide a path pattern."):
        list(ExperimentLoader().lazy_load())
    with pytest.raises(ValueError, match="No file paths provided"):
        list(ExperimentLoader(str(DATA_DIR / "nothing-here*")).lazy_load())


def test_loader_logs_validation_errors_and_other_problems_differently(config_dir, caplog):
    write_config(config_dir / "wrong-types.json", {"questionnaire": {"name": 5}})

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        documents = list(ExperimentLoader().lazy_load(str(config_dir / "*.json")))

    assert names(documents) == ["A", "B"]
    messages = warnings_of_the_reader(caplog)
    assert [m.split(" ", 2)[0:2] for m in messages] == [
        ["Cannot", "read"],  # garbage.json: not JSON
        ["Cannot", "read"],  # invalid.json: a section of the wrong type
        ["Invalid", "experiment"],  # wrong-types.json: pydantic's ValidationError
    ]
    assert "garbage.json" in messages[0]
    assert "invalid.json" in messages[1]
    assert "wrong-types.json" in messages[2]


def test_loader_skips_files_that_are_not_utf8(tmp_path, caplog):
    path = tmp_path / "latin1.json"
    path.write_bytes('{"name": "Größe"}'.encode("latin-1"))

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert list(ExperimentLoader(str(path)).lazy_load()) == []

    assert "utf-8" in warnings_of_the_reader(caplog)[0]


@pytest.mark.parametrize("content", ["[]", "5"])
def test_loader_skips_json_that_is_not_an_object(tmp_path, caplog, content):
    path = tmp_path / "not-an-object.json"
    path.write_text(content, encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert list(ExperimentLoader(str(path)).lazy_load()) == []

    assert str(path) in warnings_of_the_reader(caplog)[0]


def test_loader_in_strict_mode_raises_instead_of_skipping(config_dir):
    with pytest.raises(ValueError, match="Expecting property name"):
        list(ExperimentLoader(str(config_dir / "*.json"), strict=True).lazy_load())
    with pytest.raises(ValueError, match="'questionnaire' must be an object"):
        list(ExperimentLoader(str(config_dir / "invalid.json"), strict=True).lazy_load())


def test_loader_lazy_load_from_dicts_keeps_valid_and_logs_invalid(config_dict, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        documents = list(
            ExperimentLoader().lazy_load_from_dicts(
                [named(config_dict, "ok"), {"parameters": {}, "questionnaire": {"name": 5}}]
            )
        )

    assert names(documents) == ["ok"]
    (message,) = warnings_of_the_reader(caplog)
    assert message.startswith("Invalid experiment dictionary #1: ")


def test_reader_helpers_build_documents(config_dict, tmp_path):
    from_json = reader._experiment_from_json(copy.deepcopy(config_dict))
    from_file = reader._experiment_from_file(str(CONFIG_PATH))

    assert from_json.name == from_file.name == config_dict["name"]
    assert isinstance(from_json, ExperimentDocument)


def test_reader_helper_for_a_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        reader._experiment_from_file("/no/such/file.json")


# ---- asynchronous loading ------------------------------------------------------------------------


def test_async_loading_from_files(config_dir):
    async def load():
        loader = ExperimentLoader(str(config_dir / "[ab].json"))
        return [document async for document in loader.alazy_load()]

    assert names(asyncio.run(load())) == ["A", "B"]


def test_async_loading_with_a_pattern_given_to_the_call(config_dir):
    async def load():
        return [d async for d in ExperimentLoader().alazy_load(str(config_dir / "a.json"))]

    assert names(asyncio.run(load())) == ["A"]


def test_async_aload_collects_the_documents(config_dir):
    documents = asyncio.run(ExperimentLoader(str(config_dir / "[ab].json")).aload())

    assert names(documents) == ["A", "B"]


def test_async_loading_without_a_pattern_raises():
    async def load():
        return [document async for document in ExperimentLoader().alazy_load()]

    with pytest.raises(ValueError, match="No file paths provided"):
        asyncio.run(load())


def test_async_loading_skips_invalid_files_and_logs_them(config_dir, caplog):
    async def load():
        loader = ExperimentLoader()
        return [d async for d in loader.alazy_load(str(config_dir / "*.json"))]

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        documents = asyncio.run(load())

    assert names(documents) == ["A", "B"]
    garbage, invalid = warnings_of_the_reader(caplog)
    assert "garbage.json" in garbage and "invalid.json" in invalid


def test_async_loading_in_strict_mode_raises(config_dir):
    async def load():
        loader = ExperimentLoader(strict=True)
        return [d async for d in loader.alazy_load(str(config_dir / "*.json"))]

    with pytest.raises(ValueError, match="Expecting property name"):
        asyncio.run(load())


def test_async_loading_from_dicts(config_dict, caplog):
    async def load():
        loader = ExperimentLoader()
        configs = [
            named(config_dict, "x"),
            {"parameters": {}, "questionnaire": {"name": 5}},
            named(config_dict, "y"),
        ]
        return [document async for document in loader.alazy_load_from_dicts(configs)]

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        documents = asyncio.run(load())

    assert names(documents) == ["x", "y"]
    (message,) = warnings_of_the_reader(caplog)
    assert message.startswith("Invalid experiment dictionary #1: ")


def test_async_loading_from_dicts_logs_other_errors(caplog):
    async def load():
        loader = ExperimentLoader()
        configs = [{"parameters": {}, "models": {"x": {"type": "no-such-type"}}}]
        return [document async for document in loader.alazy_load_from_dicts(configs)]

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert asyncio.run(load()) == []

    assert warnings_of_the_reader(caplog) == [
        "Cannot read dictionary #0: Unknown or missing model type for key: x"
    ]


def test_async_loading_from_dicts_in_strict_mode_raises():
    async def load():
        loader = ExperimentLoader(strict=True)
        configs = [{"parameters": {}, "models": {"x": {"type": "no-such-type"}}}]
        return [document async for document in loader.alazy_load_from_dicts(configs)]

    with pytest.raises(ValueError, match="Unknown or missing model type for key: x"):
        asyncio.run(load())


def test_async_helper_builds_a_document(config_dict):
    from_file = asyncio.run(reader._experiment_from_file_async(str(CONFIG_PATH)))

    assert from_file.name == config_dict["name"]


# ------------------------------------------------------------------------------------------
# load_example_experiment: the bundled configuration as a loader
# ------------------------------------------------------------------------------------------


def test_load_example_experiment_can_replace_the_seeds():
    bundled = rup.load_example_config("bfi")["parameters"]["seeds"]

    experiment = rup.load_example_experiment("bfi", models={}, seeds=[1, 2, 3])

    assert experiment.parameters.seeds == ["1", "2", "3"]  # integers are stored as strings
    assert rup.load_example_experiment("bfi", models={}).parameters.seeds == bundled


def test_load_example_experiment_runs_once_per_replaced_seed():
    experiment = rup.load_example_experiment("bfi", models={}, seeds=["3", 4])
    experiment.add_model(FakeListLLM(responses=["3"]), identifier="fake")

    summary = experiment.run(show_progress=False)

    df = experiment.get_answers_as_dataframe()
    assert set(df["Run Seed"]) == {"3", "4"} and summary.n_calls == len(df)


def test_load_example_experiment_rejects_seeds_that_are_no_integers():
    with pytest.raises(ValidationError, match="seeds"):
        rup.load_example_experiment("bfi", models={}, seeds=["many"])


def test_load_example_experiment_never_changes_the_bundled_configuration():
    before = rup.load_example_config("bfi")

    rup.load_example_experiment("bfi", models={}, seeds=[99])

    assert rup.load_example_config("bfi") == before


# ------------------------------------------------------------------------------------------
# example_experiment_bfi (deprecated; the Hugging Face back-end is faked)
# ------------------------------------------------------------------------------------------


@pytest.fixture
def faked_backends(monkeypatch):
    """Fake ``transformers``, ``langchain_huggingface`` and ``huggingface_hub``: no download."""
    calls, logins = [], []
    llm = FakeListLLM(responses=["3"])

    def fake_pipeline(*args, **kwargs):
        calls.append((args, kwargs))
        return "PIPELINE"

    def fake_wrapper(pipeline):
        assert pipeline == "PIPELINE"
        return llm

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=fake_pipeline))
    monkeypatch.setitem(
        sys.modules, "langchain_huggingface", SimpleNamespace(HuggingFacePipeline=fake_wrapper)
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(login=logins.append))
    return SimpleNamespace(calls=calls, logins=logins, llm=llm)


def bfi_example(**kwargs):
    with pytest.warns(DeprecationWarning, match="load_example_experiment"):
        return rup.example_experiment_bfi(**kwargs)


def test_example_experiment_is_deprecated(faked_backends):
    with pytest.warns(DeprecationWarning, match=r"load_example_experiment\('bfi', models=\{\}\)"):
        rup.example_experiment_bfi()


def test_example_experiment_creates_the_pipeline_with_the_given_settings(faked_backends):
    bfi_example(
        model_name="org/small-model",
        pipeline_type="text-generation",
        temperature=0.2,
        max_new_tokens=16,
    )

    assert faked_backends.calls == [
        (
            ("text-generation",),
            {"model": "org/small-model", "temperature": 0.2, "max_new_tokens": 16},
        )
    ]


def test_example_experiment_defaults(faked_backends):
    bfi_example()

    assert faked_backends.calls == [
        (
            ("text2text-generation",),
            {"model": "google/flan-t5-small", "temperature": 0.7, "max_new_tokens": 128},
        )
    ]


def test_example_experiment_content(faked_backends):
    experiment = bfi_example()
    bundled = rup.load_example_config("bfi")

    assert isinstance(experiment, ExperimentDocument)
    assert experiment.name == bundled["name"]
    assert experiment.parameters.seeds == bundled["parameters"]["seeds"]
    assert experiment.list_personas() == list(bundled["demographic_profiles"])
    assert experiment.questionnaire.get_number_of_questions() == len(
        bundled["questionnaire"]["instruction_items"]
    )
    assert experiment.get_model("hf_generative_model") is faked_backends.llm
    prompt = experiment.get_prompt()
    assert isinstance(prompt, ChatPromptTemplate)
    assert sorted(prompt.input_variables) == [
        "answer_options",
        "general_instruction",
        "persona_description",
        "question",
    ]


def test_example_experiment_contains_only_the_requested_model(faked_backends):
    """The bundled example's own model is replaced, never added next to the requested one."""
    experiment = bfi_example()

    assert experiment.list_models() == ["hf_generative_model"]


def test_example_experiment_runs_offline_with_the_faked_model(faked_backends):
    experiment = bfi_example()
    bundled = rup.load_example_config("bfi")
    expected = (
        len(bundled["questionnaire"]["instruction_items"])
        * len(bundled["demographic_profiles"])
        * len(bundled["parameters"]["seeds"])
    )

    summary = experiment.run(show_progress=False)

    df = experiment.get_answers_as_dataframe()
    assert summary.n_calls == len(df) == expected
    assert summary.n_failed == 0
    assert set(df["Answer"]) == {"3"}
    assert set(df["Model ID"]) == {"hf_generative_model"}
    assert set(df["Persona ID"]) == set(bundled["demographic_profiles"])


def test_example_experiment_logs_in_when_an_api_key_is_given(faked_backends):
    bfi_example(api_key="hf_test_token")
    bfi_example()

    assert faked_backends.logins == ["hf_test_token"]


def test_example_experiment_without_the_huggingface_extra_names_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)  # makes "import transformers" fail

    with pytest.warns(DeprecationWarning):
        with pytest.raises(ImportError, match=r"pip install 'rupsycho\[huggingface\]'"):
            rup.example_experiment_bfi()
