
# RUPsycho Python Package

## Introduction
RUPsycho is a Python package designed for the application of large language models (LLMs) in social science research. It facilitates exploring humanlike behaviors through LLMs, offering a unique perspective in the field of natural language processing (NLP). This README guide will help you in setting up and effectively utilizing the RUPsycho package for your research needs.

## Installation

To install RUPsycho, run the following command in your Python environment:

```bash
pip install git+https://github.com/julianschelb/rupsycho.git
```

## Example Usage

After installation, import RUPsycho in your Python script or Jupyter notebook to begin. Here's a simple example to get you started:

```python
import rupsycho as rup

# Load experiment data
config_file_path = "./data/bfi_demo_config.json"
experiment = rup.experiment_from_file(config_file_path)

# Run the experiment
experiment.run()

# Save the results
answers = experiment.get_answers_as_dataframe()
```

## For Devs

### Install Dependencies
After cloning this repository, use Poetry to manage dependencies and install the necessary packages for development.

1. Install Poetry:
   ```
   pip install poetry
   ```

2. Install dependencies:
   ```
   poetry install
   ```

## Build and Install the Package

You can build the package by running:

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

Or, you can install it directly from GitHub using pip:

```
pip install git+https://github.com/julianschelb/rupsycho.git
```


### Generating Documentation
Navigate to the `docs/` directory and run:

```bash
make html
```

This will generate HTML documentation in the `docs/_build/html` directory.

### Running Tests

To ensure everything is working correctly, run the test cases using pytest:

```bash
pytest -v 
```