import unittest
import rupsycho as rup
import os


class TestExperimentRun(unittest.TestCase):

    def setUp(self):
        """Set up file paths and any other shared resources."""
        self.file_path = "./data/bfi_demo_config.json"
        self.results_path = "./data/bfi_demo_config_results.json"

    def tearDown(self):
        """Clean up any files created during the tests."""
        if os.path.exists(self.results_path):
            os.remove(self.results_path)

    def test_experiment_run(self):
        """Test if an experiment can be loaded and run without errors."""
        # Load the experiment from the file and run it
        experiment = rup.experiment_from_file(self.file_path)
        experiment.run()

    # def test_experiment_reloading(self):
    #     """Test if an experiment can be reloaded and run without errors."""
    #     # Load the experiment from the file and run it
    #     experiment = rup.experiment_from_file(self.file_path)
    #     experiment.run()

    #     # Export the experiment to a file
    #     experiment.export_to_file(self.results_path)

    #     # Reload the experiment from the results file and run it
    #     reloaded_experiment = rup.experiment_from_file(self.results_path)
    #     reloaded_experiment.run()

    # def test_experiment_reproducibility(self):
    #     """Test if an experiment produces reproducible results."""
    #     # Run the experiment for the first time
    #     experiment_first = rup.experiment_from_file(self.file_path)
    #     experiment_first.run()

    #     # Run the experiment for the second time
    #     experiment_second = rup.experiment_from_file(self.file_path)
    #     experiment_second.run()

    #     # Compare the results to ensure reproducibility
    #     self.assertEqual(experiment_first.get_answers(),
    #                      experiment_second.get_answers())


if __name__ == '__main__':
    unittest.main(verbosity=2)
