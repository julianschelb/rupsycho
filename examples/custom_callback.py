"""Store every answer in a SQLite database with a custom callback.

A callback receives each answer **as soon as it is generated**, so nothing is lost when a long
run crashes. This example implements ``SQLiteCallback`` with the standard library only
(``sqlite3``) and runs it offline: a scripted stand-in model (not a language model) answers the
bundled Big Five example, one question fails on purpose, and the database is queried afterwards::

    python examples/custom_callback.py              # in-memory database, gone after the run
    python examples/custom_callback.py answers.db   # keep the database in a file

Things to note when writing a callback:

* ``save_answer`` is called once per model call, in the order of the run (also when the calls
  run concurrently with ``max_concurrency``), from the thread that called ``run()``.
* The answer is ``None`` when the model call failed. Here these rows are stored with a NULL
  answer, so failures stay visible instead of silently missing.
* An exception inside a callback is reported as a warning and does not stop the experiment.
* The primary key makes re-runs idempotent: a repeated call replaces its row.
"""

from __future__ import annotations

import argparse
import random
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models.llms import LLM

import rupsycho as rup
from rupsycho.callbacks import Callback


class SQLiteCallback(Callback):
    """Write each answer to the table ``answers`` of a SQLite database.

    Args:
        path: Database file, or ``":memory:"`` for a database that lives as long as the callback.

    Example:
        ```python
        with SQLiteCallback("answers.db") as sink:
            experiment.run(callbacks=[sink])
        ```
    """

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS answers (
            experiment TEXT NOT NULL,
            item_id    INTEGER NOT NULL,
            question   TEXT,
            model_id   TEXT NOT NULL,
            persona_id TEXT NOT NULL,
            seed       TEXT NOT NULL,
            seconds    REAL,
            answer     TEXT,
            PRIMARY KEY (experiment, item_id, model_id, persona_id, seed)
        )
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.connection = sqlite3.connect(str(path))
        self.connection.execute(self.SCHEMA)
        self.connection.commit()

    def save_answer(
        self,
        experiment: Any,
        instruction_item_id: int,
        instruction_item: Any,
        model_id: str,
        profile_id: str,
        random_seed: Any,
        time: float,
        answer: Any,
    ) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                experiment.name or "",
                instruction_item_id,
                instruction_item.question,
                model_id,
                profile_id,
                str(random_seed),
                time,
                None if answer is None else str(answer),
            ),
        )
        self.connection.commit()  # one commit per answer: a crash loses at most the current call

    def close(self) -> None:
        """Close the database connection."""
        self.connection.close()

    def __enter__(self) -> SQLiteCallback:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class ScriptedLLM(LLM):
    """STAND-IN FOR A REAL MODEL: replies with an option number, offline and deterministic.

    The number is drawn from the seed and the prompt. Prompts that contain ``fail_on`` raise an
    error, to imitate an outage or a rate limit.
    """

    seed: int | None = None
    fail_on: str | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        if self.fail_on and self.fail_on in prompt:
            raise RuntimeError(f"cannot answer prompts containing {self.fail_on!r}")
        return str(random.Random(f"{self.seed}|{prompt}").randint(1, 5))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the demo and print what the database contains."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "database", nargs="?", default=":memory:", help="SQLite file (default: in memory)"
    )
    args = parser.parse_args(argv)

    experiment = rup.load_example_experiment("bfi", models={})
    # The last question of the questionnaire fails for every persona
    experiment.add_model(ScriptedLLM(fail_on="is depressed, blue"), identifier="scripted")

    with SQLiteCallback(args.database) as sink:
        summary = experiment.run(callbacks=[sink], on_error="ignore", show_progress=False)
        count = "SELECT COUNT(*) FROM answers"
        print(f"model calls: {summary.n_calls}, failed: {summary.n_failed}")
        print(f"rows in the database: {sink.connection.execute(count).fetchone()[0]}")

        experiment.run(callbacks=[sink], on_error="ignore", show_progress=False)
        print(f"after a second run  : {sink.connection.execute(count).fetchone()[0]} (replaced)")

        print("\nmean option number per persona (answered calls only):")
        query = """
            SELECT persona_id, COUNT(answer), AVG(CAST(answer AS REAL))
            FROM answers GROUP BY persona_id ORDER BY persona_id
        """
        for persona, answered, mean in sink.connection.execute(query):
            print(f"  {persona:<22} {answered} answers, mean {mean:.2f}")

        print("\nfailed calls (stored with a NULL answer):")
        failed = "SELECT item_id, persona_id, question FROM answers WHERE answer IS NULL"
        for item_id, persona, question in sink.connection.execute(failed):
            print(f"  item {item_id} ({question!r}) as {persona}")

    if args.database != ":memory:":
        print(f"\nThe database was written to {Path(args.database).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
