"""Backwards-compatible alias for :mod:`rupsycho.parsers.parser`.

``rupsycho.parser`` used to be a verbatim copy of ``rupsycho.parsers.parser``.
Import :class:`BasicParser` from ``rupsycho.parsers`` instead.
"""

from rupsycho.parsers.parser import BasicParser

__all__ = ["BasicParser"]
