===========
Parsers
===========

This document provides an overview of the various parser classes available in the `rupsycho.parsers` module. The parsers are organized into three categories: cleaners, judges, and validators.

Cleaners
========

This section defines specialized parser classes called Cleaners, designed to process and clean textual answers by removing artifacts that may have been introduced during the text generation process.

.. automodule:: rupsycho.parsers.cleaners
    :members:
    :undoc-members:
    :show-inheritance:


Judges
======

This section defines specialized parser classes called Judges, designed to evaluate and process input answers. The Judges parser takes in a textual answer and applies a series of checks and transformations to determine a final prediction or verdict.

.. automodule:: rupsycho.parsers.judges
    :members:
    :undoc-members:
    :show-inheritance:


Validators
==========

This section defines specialized parser classes called Validators, designed to assess and validate textual answers. The Validator parser takes an answer as input and returns the original answer along with a validation status that indicates whether the input meets specified criteria or rules.

.. automodule:: rupsycho.parsers.validators
    :members:
    :undoc-members:
    :show-inheritance: