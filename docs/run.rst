Running an Experiment 
=====================

This guide will walk you through running an experiment using the LLaMA2 chat-based model in combination with a simplified prompt template. The experiment setup leverages the `rupsycho` library along with the HuggingFace `transformers` pipeline.

Overview
--------

The experiment involves loading a pre-trained LLaMA2 model, setting up a prompt template, and running the experiment to generate responses. Finally, the results are exported to a file for further analysis.

Steps to Run the Experiment
---------------------------

1. **Create an Experiment**

   First, create an experiment using the `rup.example_experiment_bfi()` function, which sets up a Big Five Inventory (BFI) questionnaire.

   .. code-block:: python

      import rupsycho as rup

      # Step 1: Create an experiment from a dictionary or using an existing method
      experiment = rup.example_experiment_bfi()  # Example setup for BFI questionnaire

2. **Load the LLaMA2 Chat-Based Model**

   Load the LLaMA2 model using the HuggingFace `pipeline`. This model will be used for generating responses based on the prompt.

   .. code-block:: python

      from transformers import pipeline
      from langchain.chains import HuggingFacePipeline

      # Step 2: Load the LLaMA2 chat-based model
      pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct")
      model_local = HuggingFacePipeline(pipeline=pipe)

3. **Add the Model to the Experiment**

   Add the loaded LLaMA2 model to the experiment.

   .. code-block:: python

      # Step 3: Add the local LLaMA2-based model to the experiment
      experiment.add_model(model_local)

4. **Define the Prompt Template**

   Create a simplified prompt template that will be used by the model to generate answers.

   .. code-block:: python

      from langchain_core.prompts import ChatPromptTemplate

      # Step 4: Define the simplified prompt template
      user_message_template = "{general_instruction} {persona_description} was asked: {question} Options: {answer_options} Answer:"
      chat_prompt = ChatPromptTemplate.from_messages([
          ("user", user_message_template)
      ])

5. **Set the Prompt in the Experiment**

   Set the defined prompt in the experiment.

   .. code-block:: python

      # Step 5: Set the prompt in the experiment
      experiment.set_prompt(chat_prompt)

6. **Set a Parser (Optional)**

   Optionally, set a parser to process the output. If no parser is set, a default string parser will be used.

   .. code-block:: python

      from rupsycho.parser import BasicParser

      # Step 6: Optionally set a parser (if not set, a default will be used)
      basic_parser = BasicParser()
      experiment.set_parser(basic_parser)

7. **Run the Experiment**

   Run the experiment to generate responses based on the prompt and model.

   .. code-block:: python

      # Step 7: Run the experiment
      experiment.run()

8. **Export the Results**

   After running the experiment, export the results to a file in JSON format for further analysis.

   .. code-block:: python

      # Step 8: Export the results to a file
      experiment.export_to_file("experiment_results.json")

Conclusion
----------

By following these steps, you can easily set up and run an experiment using a LLaMA2 chat-based model. The experiment involves defining a prompt template, processing the input with a model, and exporting the generated responses for further analysis.