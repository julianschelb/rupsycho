import unittest
from rupsycho import LocalModel


class TestLocalModel(unittest.TestCase):
    def setUp(self):
        # Set up an instance of LocalModel for testing
        self.model = LocalModel(
            model_name="google/flan-t5-base",
            device_map=None,
            params={"max_new_tokens": 100},
        )

    def test_set_prompt_template_with_all_variables(self):
        # Test case where all required variables are present
        valid_template = """
        {{test.general_instructions}}
        Persona: {{persona.name}}, {{persona.description}}
        Question: {{question.question}}
        Answer Options: {{question.answer_options}}
        """
        try:
            self.model.set_prompt_template(valid_template)
        except ValueError:
            self.fail("set_prompt_template raised ValueError unexpectedly!")

    def test_set_prompt_template_with_missing_variables(self):
        # Test case where some required variables are missing
        invalid_template = """
        Persona: {{persona.name}}, {{persona.description}}
        Answer Options: {{question.answer_options}}
        """
        with self.assertRaises(ValueError):
            self.model.set_prompt_template(invalid_template)


if __name__ == "__main__":
    unittest.main()
