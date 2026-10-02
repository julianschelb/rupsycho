"""Execute the example notebooks offline, check that they run, and optionally store the outputs.

The notebooks in this folder are committed *with* their outputs. Use this script to re-create
them after the API changed, or to check that they still run::

    python examples/execute_notebooks.py                 # run every notebook in a scratch copy
    python examples/execute_notebooks.py quickstart      # only the notebooks whose name matches
    python examples/execute_notebooks.py --write         # run in place and store the outputs

Conventions that make the stored outputs small and reproducible:

* Cells tagged ``skip-execution`` are not run. They load real models (a download) or call hosted
  APIs, and keep no output in the repository.
* No execution timings are stored, consecutive chunks of one stream are merged and the kernel
  specification is fixed, so re-executing an unchanged notebook gives an unchanged file.
* Without ``--write`` everything runs in a temporary copy of this folder, so nothing in the
  repository changes (the notebooks write result files next to themselves).

The script needs nbclient, nbformat and ipykernel, which the ``dev`` extra installs.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError
except ImportError as error:  # pragma: no cover - depends on the environment
    raise SystemExit(
        f"{error}\nExecuting notebooks needs nbclient, nbformat and ipykernel: "
        "pip install 'rupsycho[dev]'"
    ) from error

EXAMPLES_DIR = Path(__file__).resolve().parent
SKIP_TAG = "skip-execution"
KERNELSPEC = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}


def notebook_paths(root: Path = EXAMPLES_DIR, *names: str) -> list[Path]:
    """The notebooks directly inside ``root`` (the user-study folder is not included)."""
    paths = sorted(root.glob("*.ipynb"))
    return [p for p in paths if not names or any(name.lower() in p.stem.lower() for name in names)]


def coalesce_streams(outputs: list[Any]) -> list[Any]:
    """Merge consecutive stream outputs of the same name.

    The kernel flushes ``stdout`` in chunks whose boundaries depend on timing; merged streams
    do not.
    """
    merged: list[Any] = []
    for output in outputs:
        if (
            merged
            and output.output_type == "stream"
            and merged[-1].output_type == "stream"
            and merged[-1].name == output.name
        ):
            merged[-1].text += output.text
        else:
            merged.append(output)
    return merged


def normalise(notebook: nbformat.NotebookNode) -> None:
    """Remove everything from an executed notebook that changes from run to run."""
    notebook.metadata["kernelspec"] = dict(KERNELSPEC)
    for cell in notebook.cells:
        for key in ("execution", "collapsed", "scrolled"):
            cell.metadata.pop(key, None)
        if cell.cell_type == "code":
            cell.outputs = coalesce_streams(cell.outputs)
    nbformat.validate(notebook)


def execute(
    path: Path,
    *,
    cwd: Path | None = None,
    timeout: int = 300,
    skip_tag: str | None = SKIP_TAG,
    setup_code: str | None = None,
) -> nbformat.NotebookNode:
    """Execute a notebook in a fresh kernel and return it (the file is not modified).

    Args:
        path: The notebook.
        cwd: Working directory of the kernel (default: the folder of the notebook).
        timeout: Seconds a single cell may take.
        skip_tag: Cells with this tag are skipped; ``None`` executes every cell.
        setup_code: Code to run in a first, extra cell (used by the tests to replace the model
            loader, so that notebooks which normally download a model run offline).

    Returns:
        The executed notebook with normalised outputs.

    Raises:
        nbclient.exceptions.CellExecutionError: If a cell raises.
    """
    notebook = nbformat.read(path, as_version=4)
    if setup_code is not None:
        notebook.cells.insert(0, nbformat.v4.new_code_cell(setup_code, id="setup"))
    NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(cwd or path.parent)}},
        record_timing=False,
        skip_cells_with_tag=skip_tag or "__no_such_tag__",
    ).execute()
    if setup_code is not None:
        del notebook.cells[0]
    normalise(notebook)
    return notebook


def describe(error: Exception) -> str:
    """One line about why a notebook failed (the exception of the cell, if there is one)."""
    if isinstance(error, CellExecutionError):
        return f"{error.ename}: {error.evalue}"
    return f"{type(error).__name__}: {(str(error).splitlines() or [''])[-1]}"


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the notebooks; return 1 if any of them fails."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("names", nargs="*", help="only notebooks whose name contains one of these")
    parser.add_argument(
        "--write", action="store_true", help="run in place and store the executed notebooks"
    )
    args = parser.parse_args(argv)

    paths = notebook_paths(EXAMPLES_DIR, *args.names)
    if not paths:
        parser.error("no notebook matches")

    failed = 0
    scratch_output = EXAMPLES_DIR / "output"  # created by Callbacks.ipynb
    had_output = scratch_output.exists()
    with tempfile.TemporaryDirectory(prefix="rupsycho-notebooks-") as scratch:
        root = EXAMPLES_DIR
        if not args.write:
            root = Path(scratch) / "examples"
            shutil.copytree(EXAMPLES_DIR, root, ignore=shutil.ignore_patterns("__pycache__"))
        for path in paths:
            target = root / path.name
            try:
                executed = execute(target)
            except Exception as error:  # report every failing notebook, not only the first
                failed += 1
                print(f"FAILED  {path.name}: {describe(error)}")
                continue
            if args.write:
                nbformat.write(executed, path)
            print(f"ok      {path.name}{'  (written)' if args.write else ''}")
    if args.write and not had_output and scratch_output.is_dir():
        shutil.rmtree(scratch_output)  # created by the notebooks just now, nothing of yours
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
