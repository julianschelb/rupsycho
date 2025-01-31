import unittest
import rupsycho as rup
import os


class TestExperimentRun(unittest.TestCase):

    def setUp(self):
        """Set up file paths and any other shared resources."""
        self.file_path = "tests/data/bfi_test_config.json"
        self.results_path = "tests/data/bfi_test_results.json"

    # def tearDown(self):
    #     """Clean up any files created during the tests."""
    #     if os.path.exists(self.results_path):
    #         os.remove(self.results_path)

    def test_experiment_loading(self):
        """Test if an experiment can be loaded without errors."""
        experiment = rup.experiment_from_file(self.file_path)
        self.assertIsNotNone(experiment, "Failed to load experiment.")

    def test_experiment_running(self):
        """Test if an experiment can be loaded and run without errors."""
        experiment = rup.experiment_from_file(self.file_path)
        experiment.run()  # Will only fail if an exception occurs

    def test_experiment_running_and_saving(self):
        """Test if an experiment can be loaded, run, and results saved successfully."""
        experiment = rup.experiment_from_file(self.file_path)
        experiment.run()
        answers = experiment.get_answers()

        self.assertIsInstance(answers, list, "Answers should be a list.")


if __name__ == '__main__':
    unittest.main(verbosity=2)
