
# RUPsycho Python Package

## Introduction
RUPsycho is a Python package designed for the application of large language models (LLMs) in social science research. It facilitates exploring humanlike behaviors through LLMs, offering a unique perspective in the field of natural language processing (NLP).

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

### Using the Configurator App
To open the R.U.Psycho experiment configurator web-app in your browser, navigate to the project folder and run:

```
rup-configurator
```

The configurator web-app is split into two sections: The 'Configurator' section contains text fields where you can manually enter the content of different parts of the experiment. It also contains tools to facilitate this process. The 'Input/Output' section contains a text field where you can enter the text of your questionnaire as input to a language model. It also contains the experiment configuration in its current state.

To create a R.U.Psycho experiment configuration in the configurator there are different workflows and tools at your disposal.

A purely ```manual configuration``` is the most straight forward method. For this you can ignore the 'Tools' and 'Questionnaire Text' tabs in the configurator and manually put in the entire content of the experiment via the remaining four tabs.

- `Experiment Info` contains the basic administrative information about the experiment. 
- `Demographic Profiles` allows you to add the personas to the configuration that can later in the experiment be imitated by a language model. You can add, edit, delete and duplicate personas to you liking here. 
- Note that when the configurator is loaded with large amounts of data e.g. a lot of profiles, it is normal that it takes a moment to respond to changes.
- `Questionnaire Info` contains the basic information about the questionnaire.
- `Questionnaire Items` allows you to add  the questionnaire itself to the configuration as a collection of its items. A questionnaire item is a question and its set of answer options. You can add, edit, delete and duplicate items to you liking here. 
- If your questionnaire uses the same set of answer options for all questions, then you can switch on the `Global answer set` option. This allows you to put in this answer set only once instead of putting it in for every question. It is then automatically applied to all your added questions in the configuration.
- As you add content, the 'Resulting Configuration' is updated constantly. At any point during the process you can download the configuration in its current state as a JSON file with the `download` button.
- The configurator currently does not support the editing of the sections 'parameters', 'prompt_template', 'models' and 'attributes' of a configuration due to their variable structure.

You can use the built in `Language Model` (OpenAI GPT-4o mini) to facilitate this process. This function is located in the 'Tools' tab and it automatically creates the questionnaire-section of the configuration for you as you would do it manually. Especially on larger questionnaires this can safe a lot of time. After the language model is done, you can seamlessly continue editing the results as usual. It works as follows:

- Put in an `OpenAI API-key` in the corresponding field. The average devaluation of the API-key from one run of the model currently is a fraction of a cent. 
- `Put in the questionnaire` either by uploading it as a PDF in the file browser function or by directly pasting its text into the Questionnaire Text input field. This textfield is editable.
- `Cleaning the text` can significantly improve the output quality of the language model. During the extraction of the text from the pdf, the formatting can get mixed up. Especially listings of answer options are susceptible to this. Although the model displays a strong resilience to noise, it can help to remove irrelevant sections from the text. For this you can use the slider to select a range of relevant pages from the questionnaire.
- You can now `run` the model. Depending on the size and complexity of your questionnaire this might take a moment. This will overwrite all content from the questionnaire part of the configuration that you previously put in. If the model fails to produce a valid output then the entire config is reset to its empty initial state. So it is advisable to run the model at the beginning of the configuration.
- After the model successfully finished you can continue with `editing and downloading` the results as you like. You will find the questionnaire part of the configuration filled out as if you had done it manually.
- You can `rerun` the model as often as you like. The text of the questionnaire remains in its input field for any further runs of the model. But there is not need to do anything with it.

You can `import a configuration` that already exists and that you want to further edit. This function is located in the 'Tools' tab.

- `Upload the configuration` as a JSON file. The configuration is accepted if ALL expected keys ('name', 'attributes', 'parameters', items, answers etc.) are found in the correct positions. Any other keys are ignored. If the configuration is accepted it will overwrite the entire current configuration. If it is not accepted then the entire current configuration is reset to its empty initial state.
- `Edit the configuration` as you like. You will find that the configurator has automatically taken on the state that corresponds to the imported configuration.
- Note that attributes whose editing is not supported ('parameters', 'prompt_template', 'models' and 'attributes') will be imported regardless and will just 'pass through' the editing process into the downloaded configuration.

You can `import demographic profiles` that are stored as a CSV file into the current configuration. This function is located in the 'Demographic Profiles' tab. 

- `Create a CSV file` with all your profiles. Its header has to contain 'title', 'name' and 'ethnicity' and no value can be left empty.
- `Upload the file`. If it follows the required structure then it is accepted and the profiles contained in the file will replace the profiles that are currently in the configuration. If the file is rejected at any point, all current profile elements are deleted.

