"""``import rupsycho`` must stay light: no provider SDK may be imported eagerly."""

import subprocess
import sys

PROVIDER_MODULES = [
    "langchain_openai",
    "langchain_ollama",
    "langchain_google_genai",
    "langchain_deepseek",
    "langchain_huggingface",
    "scipy",
    "IPython",
    "streamlit",
]


def run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)


def test_import_does_not_load_provider_sdks():
    code = (
        "import sys, rupsycho\n"
        f"loaded = [m for m in {PROVIDER_MODULES!r} if m in sys.modules]\n"
        "print(','.join(loaded))"
    )
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_submodules_load_on_first_access():
    code = (
        "import rupsycho as rup\n"
        "assert 'rupsycho.parsers' not in __import__('sys').modules\n"
        "rup.parsers.cleaners.BasicCleaner\n"
        "assert 'rupsycho.parsers' in __import__('sys').modules\n"
    )
    result = run_python(code)
    assert result.returncode == 0, result.stderr


def test_unknown_attribute_raises():
    result = run_python("import rupsycho\nrupsycho.does_not_exist")
    assert result.returncode != 0
    assert "AttributeError" in result.stderr


def test_version_is_exposed_and_consistent_with_the_package_metadata():
    from pathlib import Path

    import tomllib

    import rupsycho

    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    assert rupsycho.__version__ == declared
