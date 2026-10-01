import importlib.util
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    """Start the configurator app.

    Args:
        argv: Arguments passed on to ``streamlit run`` (e.g. ``["--server.port", "8600"]``).
            Defaults to the command line arguments of the process.

    Returns:
        The exit code of Streamlit (1 if Streamlit is not installed).
    """
    if importlib.util.find_spec("streamlit") is None:
        print(
            'The configurator needs Streamlit: pip install "rupsycho[configurator]"',
            file=sys.stderr,
        )
        return 1
    app_path = Path(__file__).parent / "rupsycho_experiment_configurator.py"
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            *(sys.argv[1:] if argv is None else argv),
        ]
    )
