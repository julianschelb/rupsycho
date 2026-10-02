# parser.py
"""``BasicParser``: the original name of :class:`~rupsycho.parsers.cleaners.BasicCleaner`."""

from __future__ import annotations

from rupsycho.parsers.cleaners import BasicCleaner

__all__ = ["BasicParser"]


class BasicParser(BasicCleaner):
    """Removes line breaks, unusual white space and non-ASCII characters from a text.

    Identical to [`BasicCleaner`][rupsycho.parsers.cleaners.BasicCleaner]; the name is kept for
    backwards compatibility.

    Example:
        ```python
        from rupsycho.parsers.parser import BasicParser

        BasicParser().invoke("Hello,\\nworld \U0001f60a")  # 'Hello, world'
        ```
    """

    @property
    def _type(self) -> str:
        """Identifier of the parser type."""
        return "basic_parser"
