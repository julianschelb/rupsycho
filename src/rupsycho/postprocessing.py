import rupsycho as rup
import pandas as pd
import glob
from tqdm import tqdm
tqdm.pandas()


class PostprocessingPipeline:
    """
    A pipeline for postprocessing experiment results by cleaning responses, validating them, 
    and determining the best answer option.

    This class loads CSV files matching specified patterns, applies text cleaning, validation, 
    and judgment processes, and then saves the processed results.

    Attributes:
        config_file_path (str): Path to the JSON configuration file defining the experiment structure.
        results_file_patterns (list): List of file patterns to locate result CSVs.
        cleaner (object): An instance of a cleaner that preprocesses text responses.
        validator (object): An instance of a validator that determines the validity of responses.
        judge (object): An instance of a judge that selects the best answer option.
        output_path (str): Path where the processed results CSV will be saved.
    """
    def __init__(self, config_file_path, results_file_patterns, cleaner, validator, judge, output_path="processed_results.csv"):
        """
        Initialize the PostprocessingPipeline.

        Args:
            config_file_path (str): Path to the JSON configuration file.
            results_file_patterns (list): List of file patterns to locate result CSVs.
            cleaner (object): Cleaner instance to preprocess text.
            validator (object): Validator instance to validate responses.
            judge (object): Judge instance to determine the best answer option.
            output_path (str): Path to save the processed results.
        """
        self.config_file_path = config_file_path
        self.results_file_patterns = results_file_patterns
        self.output_path = output_path
        self.cleaner = cleaner
        self.validator = validator
        self.judge = judge

        # Load the configuration file
        self.experiment = rup.experiment_from_file(path=config_file_path)
        self.experiment.clear_models()

    def _load_csv_files(self):
        """
        Load CSV files matching the specified patterns.

        Returns:
            pd.DataFrame: Combined DataFrame of all loaded CSVs.
        """
        dataframes = []
        for pattern in self.results_file_patterns:
            csv_files = glob.glob(pattern)
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                df['answer'] = df['answer'].astype('string') # in case model answers with just numbers
                dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True)

    def _process_data(self, df):
        """
        Apply cleaning, validation, and judgment to the data.

        Args:
            df (pd.DataFrame): DataFrame containing the results to process.

        Returns:
            pd.DataFrame: Processed DataFrame with added columns.
        """
        # Apply cleaning
        df['cleaned_answer'] = df['answer'].progress_apply(self.cleaner.parse)
  
        # Apply validation
        df['validation_status'] = df['cleaned_answer'].progress_apply(
            self.validator.parse)

        # TODO: Use default answer options if not provided in the instruction item
        # possible_answers = self.experiment.questionnaire.instruction_items[0].get_answer_options_as_list()
 
        # print(possible_answers)

        # Apply judgment
        df['decision'] = df.progress_apply(
            lambda row: self.judge.parse(
                text=row['cleaned_answer'],
                possible_answers=self.experiment.questionnaire.instruction_items[row['instruction_item_id']].get_answer_options_as_list(
                )
            ), axis=1
        )

        return df

    def run(self):
        """
        Run the postprocessing pipeline: load data, process it, and save the results.

        Outputs:
            Saves the processed results to a CSV file specified by the output_path.
        """
        # Load data
        print("Loading data...")
        data = self._load_csv_files()

        # Process data
        print("Processing data...")
        processed_data = self._process_data(data)

        # Save the results
        processed_data.to_csv(self.output_path, index=False)
        print(f"Processed results saved to {self.output_path}")
