**Welcome!**  
Thanks for your time and effort in helping to test and improve the RUPsycho package!  

RUPsycho is a Python package designed for the application of large language models (LLMs) in social science research. It facilitates exploring humanlike behaviors through LLMs.  

With RUPsycho, you can load experiments from files. Each experiment defines a list of questions to ask the language model. To perform these tasks, you will also need to specify:  
- **What model** to use (e.g., a specific LLM).  
- **What prompt template** to apply for asking questions.  
- **What questions** to ask.  
- **What answer options** are available.  
- **What type of persona** the model should answer as, using demographic profiles.  

Good luck and enjoy exploring the RUPsycho package! 🚀


## Task 1: Setup the Package

**Your task:** Install the necessary dependencies for the RUPsycho project and test if the package can be imported successfully.

**Please follow the following steps:**

1. **Clone the Repository**:
   - Clone the RUPsycho package from the following URL:
     ```bash
     git clone git@github.com:julianschelb/rupsycho.git
     ```

2. **Install Poetry**:
   - If you do not have Poetry installed, use the following command to install it:
     ```bash
     pip install poetry
     ```

3. **Install the Package**:
   - Navigate to the cloned repository directory and install all required dependencies using Poetry:
     ```bash
     cd rupsycho
     poetry install
     ```

4. **Test Package Import**:
   - Open a Python shell or create a script and attempt to import the package to ensure it was installed correctly:
     ```python
     import rupsycho as rup
     ```
   - If the package imports without errors, the installation is successful.

🎉 **Success!** If you've made it this far and the package imports without any issues, you're all set to start exploring the RUPsycho package. Great work!


## Task 2: Load an Experiment from a file

**Your task**: Create a new Jupyter notebook, load an experiment, run it using a large language model (LLM), and examine the results.

**Please follow the following steps:**

1. **Create a new Jupyter notebook and import the RUPsycho package**.

2. **Load an experiment file**. You can find the file named `example-questionnaire.json` in the `data` folder.

3. **Look into the `example-questionnaire.json` file** to understand the basics of how the file is organized and what kind of data is being used in the experiment. Consider the following questions:
	   - Do you understand what the general instruction does?
	   - What are the instruction items?
	   - What are demographic profiles?
	   - What model is specified to be used?

4. **Generate a HuggingFace API token**:
   - Go to [HuggingFace](https://huggingface.co/) and create an account if you don't have one.
   - Generate a new API token from your account settings.

5. **Login via the HuggingFace CLI**:
   - Open a terminal and log in using your HuggingFace API token by running the following command:
     ```bash
     huggingface-cli login
     ```

6. **Run the experiment** using the `run()` method.

7. **Examine the results** by calling `get_answers_as_dataframe()` to view the answers in a DataFrame format.

---

🎉 **Success!** If the experiment runs successfully and the results are visible in the DataFrame, you’re all set. Great job!


## Task 3: Test Reproducibility

**Your task**: Run the experiment twice to test for reproducibility, then change the seed by modifying the value in the JSON file, and verify that the results are different.

**Please follow these steps:**

1. **Run the experiment twice with the same seed**:
   - Ensure that the seed is set in the experiment configuration (in the `example-questionnaire.json` file).
   - Run the experiment once and save the results.
   - Run the experiment a second time with the same seed and verify that the results are identical to the first run.

2. **Change the seed**:
   - Open the `example-questionnaire.json` file and modify the seed value.
   - Save the file and run the experiment again.
   - Verify that the results are different from the previous runs with the original seed.

---

🎉 **Success!** If you confirm that running the experiment with the same seed produces identical results and changing the seed produces different results, your reproducibility test is successful. Well done!


## Task 4: Load a Local Model and Save Output with Callbacks

**Your task**: Load a local model, add it to the experiment, and specify callbacks to save the output as JSON and print the results.

**Please follow these steps:**

1. **Load a local model**:
   - In the configuration file (e.g., `example-questionnaire.json`), include a local model configuration like the one below to load a local HuggingFace model:
     ```json
     "local_model": {
         "type": "local_huggingface",
         "name_or_path": "HuggingFaceTB/SmolLM-1.7b-Instruct",
         "task": "text-generation",
         "device": 0,
         "parameters": {
             "max_new_tokens": 128,
             "max_length": 1024,
             "do_sample": true,
             "repetition_penalty": 1.2,
             "temperature": 0.8,
             "top_k": 50,
             "top_p": 0.95,
             "return_full_text": false
         }
     }
     ```

2. **Add the local model to the experiment**:
   - Ensure the local model is added to the experiment using the `add_model()` method.

3. **Confirm the model was added**:
   - Use the `experiment.list_models()` function to verify that the model has been successfully added to the experiment.

4. **Specify callbacks to save the output and print the results**:
   - Use the provided callbacks to save the experiment's output as a JSON file and print the results in a table.
   - Example code to include in your notebook:
     ```python
     from rupsycho.callbacks.answer_saving_callbacks import JSONLCallback, CSVCallback, PrintTableCallback

     # Define the callbacks
     jsonl_callback = JSONLCallback(file_path="output.jsonl")
     print_callback = PrintTableCallback()

     # Confirm the model has been added
     print(experiment.list_models())

     # Run the experiment with the specified callbacks
     experiment.run(callbacks=[jsonl_callback, print_callback])
     ```

5. **Run the experiment** with the local model and callbacks.

---

🎉 **Success!** If the experiment runs successfully, the model is listed, and the results are saved to the specified JSON file and printed as a table, you're all set. Great job!