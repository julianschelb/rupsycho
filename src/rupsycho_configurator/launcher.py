import importlib.util
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Start the configurator app; further command line arguments are passed to streamlit."""
    if importlib.util.find_spec("streamlit") is None:
        print(
            'The configurator needs Streamlit: pip install "rupsycho[configurator]"',
            file=sys.stderr,
        )
        return 1
    app_path = Path(__file__).parent / "rupsycho_experiment_configurator.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]])
