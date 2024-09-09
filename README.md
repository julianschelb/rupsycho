
# RUPsycho Python Package

## Introduction
RUPsycho is a Python package designed for the application of large language models (LLMs) in social science research. It facilitates exploring humanlike behaviors through LLMs, offering a unique perspective in the field of natural language processing (NLP). This README guide will help you in setting up and effectively utilizing the RUPsycho package for your research needs.

## Installation

To install RUPsycho, run the following command in your Python environment:

```bash
pip install rupsycho # not yet working
```

## Example Usage

After installation, import RUPsycho in your Python script or Jupyter notebook to begin. Here's a simple example to get you started:

```python
import rupsycho as rup

# Load experiment data
experiment = rup.experiment_from_file("bfi.json")

# Load model
model = rup.LocalModel(
    model_name="google/flan-t5-base", device_map=None, params={"max_new_tokens": 100}
)

# Run the experiment
experiment.run()

# Save the results
experiment.export_to_file("results.json")
```


## Installation via Git

Make sure your project is already managed by Poetry. If not, you can initiate Poetry in your project directory by running.

```
poetry init
```

Once your project is set up with Poetry, you can build the package by running:

```
poetry build
```

After building the package, you can install it locally using pip. Navigate to the root directory of your project (where the dist/ folder is located) and run:

```
pip install dist/rupsycho-0.1.0-py3-none-any
```

Alternatively, you can install the package directly from the project directory without building it by running:

```
pip install .
```

## For Devs

### Installation using Poetry
To install this package after cloning this repository, use Poetry for dependency management and package installation.

1. Install Poetry:
   ```
   pip install poetry
   ```

2. Install dependencies:
   ```
   poetry install
   ```

### Generating Documentation
Navigate to the `docs/` directory and run:

```bash
make html
```

This will generate HTML documentation in the `docs/_build/html` directory.