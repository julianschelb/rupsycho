.. RUPsycho documentation master file, created by
   sphinx-quickstart.

Welcome to RUPsycho documentation!
==================================

RUPsycho is a Python package designed for the application of large language models (LLMs) in social science research. It facilitates exploring humanlike behaviors through LLMs, offering a unique perspective in the field of natural language processing (NLP).

.. warning::

   **RUPsycho is under active development.** The API is subject to change, and updates may introduce breaking changes. Please refer to the documentation frequently for the latest information.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   init
   reader
   experiment
   run
   parser
   parsers
   models


Installation
============

To install RUPsycho, run the following command in your Python environment:

.. code-block:: bash

    pip install rupsycho # not yet working

Example Usage
=============

After installation, import RUPsycho in your Python script or Jupyter notebook to begin. Here's a simple example to get you started:

.. code-block:: python

    import rupsycho as rup

    # Load experiment data
    experiment = rup.experiment_from_file("bfi.json")

    # Run the experiment
    experiment.run()

    # Save the results
    experiment.export_to_file("results.json")