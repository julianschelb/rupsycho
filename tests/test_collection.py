"""Tests for ``ExperimentCollection`` and for loading experiments (``rupsycho``, ``reader``)."""

import asyncio
import copy
import glob
import json
import types
import warnings

import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

import rupsycho as rup
import rupsycho.reader as reader
from rupsycho.experiment import ExperimentDocument
from rupsycho.experiment_collection import ExperimentCollection
from rupsycho.reader import ExperimentLoader

from .conftest import CONFIG_PATH, DATA_DIR


@pytest.fixture(autouse=True)
def _quiet_progress_bars(monkeypatch):
    monkeypatch.setenv("TQDM_DISABLE", "1")


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
    collection.run_all()  # nothing to do


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

    collection.run_all()

    assert set(first.get_answers_as_dataframe()["Answer"]) == {"from-A"}
    assert set(second.get_answers_as_dataframe()["Answer"]) == {"from-B"}
    assert len(first.get_answers_as_dataframe()) == len(second.get_answers_as_dataframe()) == 8


def test_run_all_runs_the_experiments_in_order(two_experiments):
    order = []

    class OrderedLLM(FakeListLLM):
        def _call(self, prompt, stop=None, run_manager=None, **kwargs):
            order.append(self.responses[0])
            return super()._call(prompt, stop=stop, run_manager=run_manager, **kwargs)

    first, second = two_experiments
    first.add_model(OrderedLLM(responses=["A"]), identifier="m")
    second.add_model(OrderedLLM(responses=["B"]), identifier="m")

    ExperimentCollection([first, second]).run_all()

    assert order == ["A"] * 8 + ["B"] * 8


def test_run_all_propagates_a_missing_model(two_experiments):
    first, second = two_experiments
    second.add_model(FakeListLLM(responses=["B"]), identifier="m")

    with pytest.raises(ValueError, match="No models have been set in runnable_models."):
        ExperimentCollection([first, second]).run_all()

    assert len(second.get_answers_as_dataframe()) == 0  # the failure stopped the collection


def test_experiments_in_a_collection_are_independent(config_dict):
    one, two = (rup.experiment_from_dict(copy.deepcopy(config_dict)) for _ in range(2))
    one.add_model(FakeListLLM(responses=["x"]), identifier="m")

    one.run()

    assert len(one.get_answers_as_dataframe()) == 8
    assert len(two.get_answers_as_dataframe()) == 0
    assert two.get_answers() == [{}, {}, {}, {}]


def test_package_exposes_the_loading_api():
    assert rup.ExperimentDocument is ExperimentDocument
    assert rup.ExperimentCollection is ExperimentCollection
    assert rup.ExperimentLoader is ExperimentLoader
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
    experiment.run()

    assert config == config_dict


def test_experiments_from_the_same_dict_are_independent(config_dict):
    first, second = rup.experiments_from_dicts([config_dict, config_dict])
    first.add_model(FakeListLLM(responses=["3"]), identifier="m")

    first.run()

    assert len(first.get_answers_as_dataframe()) == 8
    assert len(second.get_answers_as_dataframe()) == 0


def test_experiment_from_dict_wraps_validation_errors(config_dict, capsys):
    config = {"parameters": {}, "questionnaire": "not a questionnaire"}

    with pytest.raises(RuntimeError) as error:
        rup.experiment_from_dict(config)

    # documented: no detail after the colon, the pydantic error is printed instead
    assert str(error.value) == "Failed to create experiment from dict: "
    assert isinstance(error.value.__cause__, StopIteration)
    out = capsys.readouterr().out
    assert out.startswith("Validation error in provided dictionary: 1 validation error")
    assert "questionnaire" in out


def test_experiment_from_dict_reports_other_errors_as_processing_errors(capsys):
    with pytest.raises(RuntimeError, match="Failed to create experiment from dict"):
        rup.experiment_from_dict({"parameters": {}, "models": {"x": {"type": "no-such-type"}}})

    assert capsys.readouterr().out == (
        "Error processing provided dictionary: Unknown or missing model type for key: x\n"
    )


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


def test_experiments_from_dicts_skips_invalid_entries(config_dict, capsys):
    configs = [
        named(config_dict, "first"),
        {"parameters": {}, "questionnaire": "broken"},
        {"parameters": {}, "models": {"x": {"type": "no-such-type"}}},
        named(config_dict, "last"),
    ]

    experiments = list(rup.experiments_from_dicts(configs))

    assert [e.name for e in experiments] == ["first", "last"]
    out = capsys.readouterr().out
    assert "Validation error in provided dictionary" in out
    assert "Error processing provided dictionary: Unknown or missing model type for key: x" in out


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


def test_experiment_from_file_with_a_glob_returns_one_experiment(config_dir):
    experiment = rup.experiment_from_file(str(config_dir / "[ab].json"))

    assert experiment.name in {"A", "B"}


def test_experiment_from_file_of_an_invalid_file_returns_none_and_prints(config_dir, capsys):
    assert rup.experiment_from_file(str(config_dir / "invalid.json")) is None

    assert capsys.readouterr().out.startswith(
        f"Validation error in file {config_dir / 'invalid.json'}: 1 validation error"
    )


def test_experiment_from_file_of_broken_json_returns_none_and_prints(config_dir, capsys):
    assert rup.experiment_from_file(str(config_dir / "garbage.json")) is None

    out = capsys.readouterr().out
    assert out.startswith(f"Error reading file {config_dir / 'garbage.json'}: ")
    assert "Expecting property name" in out


def test_experiment_from_file_of_a_missing_file_raises_runtime_error(tmp_path):
    path = str(tmp_path / "missing.json")

    with pytest.raises(RuntimeError) as error:
        rup.experiment_from_file(path)

    assert str(error.value).startswith(f"Failed to create experiment from file {path}: ")
    assert isinstance(error.value.__cause__, ValueError)  # raised "from e"


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


def test_experiments_from_files_skips_invalid_files_with_a_message(config_dir, capsys):
    collection = rup.experiments_from_files(str(config_dir / "*.json"))

    assert names(collection.experiments) == ["A", "B"]
    out = capsys.readouterr().out
    assert f"Validation error in file {config_dir / 'invalid.json'}: " in out
    assert f"Error reading file {config_dir / 'garbage.json'}: " in out
    assert out.count("\n") >= 2


def test_experiments_from_files_with_only_invalid_files_gives_an_empty_collection(
    config_dir, capsys
):
    collection = rup.experiments_from_files(str(config_dir / "[ig]*.json"))

    assert len(collection) == 0
    assert capsys.readouterr().out.count("Validation error in file") == 1


@pytest.mark.parametrize("pattern", ["nothing-here-*.json", "missing.json", "sub/zzz/*.json", ""])
def test_experiments_from_files_without_a_match_raises_runtime_error(config_dir, pattern):
    path_pattern = str(config_dir / pattern) if pattern else pattern

    with pytest.raises(RuntimeError) as error:
        rup.experiments_from_files(path_pattern)

    assert str(error.value) == (
        f"Failed to create experiments from files matching {path_pattern}: "
        "No file paths provided. Please provide a path pattern."
    )
    assert isinstance(error.value.__cause__, ValueError)  # raised "from e"


def test_experiments_from_files_of_the_test_data_directory():
    collection = rup.experiments_from_files(str(DATA_DIR / "*.json"))

    assert len(collection) >= 1
    assert "Generative Models for Big Five Inventory" in names(collection.experiments)


def test_loaded_collection_runs_with_fake_models(config_dir):
    collection = rup.experiments_from_files(str(config_dir / "[ab].json"))
    for experiment in collection.experiments:
        experiment.add_model(FakeListLLM(responses=[experiment.name]), identifier="m")

    collection.run_all()

    for experiment in collection.experiments:
        assert set(experiment.get_answers_as_dataframe()["Answer"]) == {experiment.name}


# ------------------------------------------------------------------------------------------
# ExperimentLoader and the reader helpers
# ------------------------------------------------------------------------------------------


def test_loader_resolves_paths(config_dir):
    assert ExperimentLoader(str(config_dir / "a.json")).file_paths == [str(config_dir / "a.json")]
    assert sorted(ExperimentLoader(str(config_dir / "*.json")).file_paths) == sorted(
        str(config_dir / name) for name in ("a.json", "b.json", "invalid.json", "garbage.json")
    )
    assert ExperimentLoader(str(config_dir / "**" / "c.json")).file_paths == [
        str(config_dir / "sub" / "c.json")
    ]
    assert ExperimentLoader(str(config_dir / "nothing*")).file_paths == []
    assert ExperimentLoader().file_paths is None
    assert ExperimentLoader("").file_paths is None
    assert ExperimentLoader(None).file_paths is None


@pytest.mark.xfail(
    strict=True,
    reason="glob.glob returns files in filesystem order, so experiments_from_files, run_all and "
    "experiment_from_file(pattern) (which takes the first match) depend on the machine",
)
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


def test_loader_reports_validation_and_other_errors_differently(config_dir, capsys):
    documents = list(ExperimentLoader().lazy_load(str(config_dir / "*.json")))

    assert names(documents) == ["A", "B"]
    out = capsys.readouterr().out
    assert "Validation error in file" in out and "invalid.json" in out
    assert "Error reading file" in out and "garbage.json" in out


def test_loader_skips_json_that_is_not_an_object(tmp_path, capsys):
    path = write_config(tmp_path / "list.json", [])

    assert list(ExperimentLoader(str(path)).lazy_load()) == []
    assert capsys.readouterr().out.startswith(f"Error reading file {path}: ")


def test_loader_lazy_load_from_dicts_keeps_valid_and_reports_invalid(config_dict, capsys):
    documents = list(
        ExperimentLoader().lazy_load_from_dicts(
            [named(config_dict, "ok"), {"parameters": {}, "questionnaire": 5}]
        )
    )

    assert names(documents) == ["ok"]
    assert capsys.readouterr().out.startswith("Validation error in provided dictionary: ")


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


def test_async_loading_skips_invalid_files_with_a_message(config_dir, capsys):
    async def load():
        loader = ExperimentLoader()
        return [d async for d in loader.alazy_load(str(config_dir / "*.json"))]

    assert names(asyncio.run(load())) == ["A", "B"]
    out = capsys.readouterr().out
    assert "Validation error in file" in out
    assert "Error reading file" in out


def test_async_loading_from_dicts(config_dict, capsys):
    async def load():
        loader = ExperimentLoader()
        configs = [
            named(config_dict, "x"),
            {"parameters": {}, "questionnaire": 5},
            named(config_dict, "y"),
        ]
        return [document async for document in loader.alazy_load_from_dicts(configs)]

    assert names(asyncio.run(load())) == ["x", "y"]
    assert "Validation error in provided dictionary" in capsys.readouterr().out


def test_async_loading_from_dicts_reports_other_errors(capsys):
    async def load():
        loader = ExperimentLoader()
        configs = [{"parameters": {}, "models": {"x": {"type": "no-such-type"}}}]
        return [document async for document in loader.alazy_load_from_dicts(configs)]

    assert asyncio.run(load()) == []
    assert capsys.readouterr().out == (
        "Error processing provided dictionary: Unknown or missing model type for key: x\n"
    )


def test_async_helpers_build_documents(config_dict):
    from_json = asyncio.run(reader._experiment_from_json_async(copy.deepcopy(config_dict)))
    from_file = asyncio.run(reader._experiment_from_file_async(str(CONFIG_PATH)))

    assert from_json.name == from_file.name == config_dict["name"]


# ------------------------------------------------------------------------------------------
# example_experiment_bfi (with the Hugging Face pipeline faked)
# ------------------------------------------------------------------------------------------


@pytest.fixture
def faked_pipeline(monkeypatch):
    """Replace the Transformers pipeline and its LangChain wrapper: nothing is downloaded."""
    calls = []
    llm = FakeListLLM(responses=["3"])

    def fake_pipeline(*args, **kwargs):
        calls.append((args, kwargs))
        return "PIPELINE"

    def fake_wrapper(pipeline):
        assert pipeline == "PIPELINE"
        return llm

    monkeypatch.setattr(rup, "pipeline", fake_pipeline)
    monkeypatch.setattr(rup, "HuggingFacePipeline", fake_wrapper)
    return types.SimpleNamespace(calls=calls, llm=llm)


def test_example_experiment_creates_the_pipeline_with_the_given_settings(faked_pipeline):
    rup.example_experiment_bfi(
        model_name="org/small-model",
        pipeline_type="text-generation",
        temperature=0.2,
        max_new_tokens=16,
    )

    assert faked_pipeline.calls == [
        (
            ("text-generation",),
            {"model": "org/small-model", "temperature": 0.2, "max_new_tokens": 16},
        )
    ]


def test_example_experiment_defaults(faked_pipeline):
    rup.example_experiment_bfi()

    assert faked_pipeline.calls == [
        (
            ("text2text-generation",),
            {"model": "google/flan-t5-small", "temperature": 0.7, "max_new_tokens": 128},
        )
    ]


def test_example_experiment_content(faked_pipeline):
    experiment = rup.example_experiment_bfi()

    assert isinstance(experiment, ExperimentDocument)
    assert experiment.name == "Generative Models for Big Five Inventory"
    assert experiment.parameters.seeds == ["7"]
    assert experiment.list_personas() == ["Profile 1"]
    assert str(experiment.get_persona("Profile 1")) == "Mr Grueber"
    assert experiment.questionnaire.get_number_of_questions() == 4
    assert experiment.get_model("hf_generative_model") is faked_pipeline.llm
    prompt = experiment.get_prompt()
    assert isinstance(prompt, ChatPromptTemplate)
    assert sorted(prompt.input_variables) == [
        "answer_options",
        "general_instruction",
        "persona_description",
        "question",
    ]
    assert isinstance(experiment.runnable_parser, StrOutputParser)


def test_example_experiment_runs_offline_with_the_faked_model(faked_pipeline):
    experiment = rup.example_experiment_bfi()
    experiment.remove_model("default_model")  # see the xfail test below

    experiment.run()

    df = experiment.get_answers_as_dataframe()
    assert len(df) == 4
    assert set(df["Answer"]) == {"3"}
    assert set(df["Model ID"]) == {"hf_generative_model"}
    assert set(df["Persona ID"]) == {"Profile 1"}


def test_example_experiment_logs_in_when_an_api_key_is_given(faked_pipeline, monkeypatch):
    import huggingface_hub

    logins = []
    monkeypatch.setattr(huggingface_hub, "login", lambda token: logins.append(token))

    rup.example_experiment_bfi(api_key="hf_test_token")
    rup.example_experiment_bfi()

    assert logins == ["hf_test_token"]


@pytest.mark.xfail(
    strict=True,
    reason="experiment_data has no 'models' key, so the 1.7B default model (default_model) is "
    "added next to the requested one and run() would download and run it as well",
)
def test_example_experiment_contains_only_the_requested_model(faked_pipeline):
    experiment = rup.example_experiment_bfi()

    assert experiment.list_models() == ["hf_generative_model"]
