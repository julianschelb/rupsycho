import subprocess
import sys
from pathlib import Path


def main():
    app_path = Path(__file__).parent / "rupsycho_experiment_configurator.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
