"""Unit tests for ``rupsycho.utils``: JSON file helpers and the tqdm flavour switch."""

import io
import json
import sys
import types
from types import SimpleNamespace

import pytest
import tqdm as tqdm_package

import rupsycho.utils as utils
from rupsycho.parsers.parser_utils import json_saver as parser_utils_json_saver
from rupsycho.utils.files import json_loader
from rupsycho.utils.files import json_saver as files_json_saver
from rupsycho.utils.imports import import_tqdm

# ===========================================================================
# package surface
# ===========================================================================


def test_utils_package_reexports_the_helpers():
    assert set(utils.__all__) == {"import_tqdm", "json_loader", "json_saver"}
    assert utils.json_loader is json_loader
    assert utils.json_saver is files_json_saver
    assert utils.import_tqdm is import_tqdm


# ===========================================================================
# json_loader
# ===========================================================================


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="empty-object"),
        pytest.param({"a": 1, "b": [1, 2, {"c": None}], "d": True, "e": 1.5}, id="nested"),
        pytest.param({"umlaut": "Grüße", "emoji": "\U0001f60a", "cjk": "日本語"}, id="unicode"),
        pytest.param([1, 2, 3], id="top-level-list"),
    ],
)
def test_json_loader_reads_utf8_files(tmp_path, payload):
    path = tmp_path / "data.json"
    path.write_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    assert json_loader(str(path)) == payload


def test_json_loader_accepts_a_path_object(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('{"a": 1}', encoding="utf-8")

    assert json_loader(path) == {"a": 1}  # type: ignore[arg-type]


@pytest.mark.parametrize("content", ["", "   ", "{not json}", '{"a": 1,}', "[1, 2"])
def test_json_loader_rejects_malformed_json(tmp_path, content):
    path = tmp_path / "broken.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        json_loader(str(path))


def test_json_loader_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        json_loader(str(tmp_path / "missing.json"))


# ===========================================================================
# json_saver (the helper exists twice: utils.files and parsers.parser_utils)
# ===========================================================================

SAVERS = [
    pytest.param(files_json_saver, id="utils.files"),
    pytest.param(parser_utils_json_saver, id="parsers.parser_utils"),
]


@pytest.mark.parametrize("saver", SAVERS)
@pytest.mark.parametrize(
    ("name", "expected_file"),
    [
        pytest.param("results", "results.json", id="extension-is-appended"),
        pytest.param("results.json", "results.json", id="extension-is-not-doubled"),
        pytest.param("run.v2", "run.v2.json", id="dots-in-the-name"),
        pytest.param("with space", "with space.json", id="space-in-the-name"),
    ],
)
def test_json_saver_file_name(saver, tmp_path, name, expected_file):
    saver({"a": 1}, name, str(tmp_path))

    assert [p.name for p in tmp_path.iterdir()] == [expected_file]


@pytest.mark.parametrize("saver", SAVERS)
def test_json_saver_defaults_to_output_json_in_the_working_directory(saver, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert saver({"a": 1}) is None

    assert json.loads((tmp_path / "output.json").read_text(encoding="utf-8")) == {"a": 1}


@pytest.mark.parametrize("saver", SAVERS)
def test_json_saver_writes_indented_json(saver, tmp_path):
    data = {"a": 1, "b": [1, 2], "c": {"d": None}}
    saver(data, "out", str(tmp_path))

    assert (tmp_path / "out.json").read_text(encoding="utf-8") == json.dumps(data, indent=4)


@pytest.mark.parametrize("saver", SAVERS)
def test_json_saver_announces_the_full_path(saver, tmp_path, capsys):
    saver({"a": 1}, "out", str(tmp_path))

    assert capsys.readouterr().out == f"File saved successfully at: {tmp_path / 'out.json'}\n"


@pytest.mark.parametrize("saver", SAVERS)
def test_json_saver_round_trips_unicode(saver, tmp_path):
    data = {"umlaut": "Grüße", "emoji": "\U0001f60a", "cjk": "日本語", "quote": 'say "hi"\n'}
    saver(data, "unicode", str(tmp_path))

    assert json_loader(str(tmp_path / "unicode.json")) == data


@pytest.mark.parametrize("saver", SAVERS)
def test_json_saver_overwrites_an_existing_file(saver, tmp_path):
    saver({"old": True, "extra": [1, 2, 3]}, "out", str(tmp_path))
    saver({"new": True}, "out", str(tmp_path))

    assert json_loader(str(tmp_path / "out.json")) == {"new": True}


@pytest.mark.parametrize("saver", SAVERS)
def test_json_saver_does_not_create_missing_directories(saver, tmp_path):
    with pytest.raises(FileNotFoundError):
        saver({"a": 1}, "out", str(tmp_path / "missing"))


@pytest.mark.xfail(
    strict=True,
    reason="json_saver opens the target for writing before serialising, so a TypeError leaves a truncated file",
)
@pytest.mark.parametrize("saver", SAVERS)
def test_json_saver_keeps_the_existing_file_when_the_data_cannot_be_serialised(saver, tmp_path):
    saver({"precious": 1}, "out", str(tmp_path))

    with pytest.raises(TypeError):
        saver({"bad": object()}, "out", str(tmp_path))

    assert json_loader(str(tmp_path / "out.json")) == {"precious": 1}


# ===========================================================================
# import_tqdm
# ===========================================================================


def _fake_ipython(monkeypatch, instance):
    """Make ``from IPython import get_ipython`` return ``instance`` (IPython need not be installed)."""
    module = types.ModuleType("IPython")
    module.get_ipython = lambda: instance  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", module)


def test_import_tqdm_outside_of_ipython_returns_the_console_bar(monkeypatch):
    _fake_ipython(monkeypatch, None)

    assert import_tqdm() is tqdm_package.tqdm


def test_import_tqdm_in_a_terminal_ipython_returns_the_console_bar(monkeypatch):
    _fake_ipython(monkeypatch, SimpleNamespace(config={"TerminalInteractiveShell": {}}))

    assert import_tqdm() is tqdm_package.tqdm


def test_import_tqdm_in_a_jupyter_kernel_returns_the_notebook_bar(monkeypatch):
    from tqdm.notebook import tqdm as notebook_tqdm

    _fake_ipython(monkeypatch, SimpleNamespace(config={"IPKernelApp": {"connection_file": "x"}}))

    result = import_tqdm()

    assert result is notebook_tqdm
    assert result is not tqdm_package.tqdm
    assert issubclass(result, tqdm_package.std.tqdm)


def test_import_tqdm_without_ipython_falls_back_to_the_console_bar(monkeypatch):
    # a ``None`` entry makes ``from IPython import ...`` raise ImportError
    monkeypatch.setitem(sys.modules, "IPython", None)

    assert import_tqdm() is tqdm_package.tqdm


def test_import_tqdm_without_the_notebook_widget_falls_back_to_the_console_bar(monkeypatch):
    _fake_ipython(monkeypatch, SimpleNamespace(config={"IPKernelApp": {}}))
    monkeypatch.setitem(sys.modules, "tqdm.notebook", None)

    assert import_tqdm() is tqdm_package.tqdm


def test_import_tqdm_result_is_usable_as_a_progress_bar(monkeypatch):
    _fake_ipython(monkeypatch, None)
    progress = import_tqdm()

    stream = io.StringIO()
    with progress(total=3, file=stream) as bar:
        bar.update(3)
        assert bar.n == 3
    assert list(progress(range(3), file=stream)) == [0, 1, 2]
    assert "3/3" in stream.getvalue()
