import pytest
from pydantic import ValidationError

import rupsycho as rup
from rupsycho.models.parameters import ExperimentParameters


class TestSeeds:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ([1, 2, 3], ["1", "2", "3"]),
            (["1", "2"], ["1", "2"]),
            ([1, "2"], ["1", "2"]),
            (7, ["7"]),
            ("7", ["7"]),
            (["-3", " 4 "], ["-3", "4"]),
            ([], []),
            (None, None),
        ],
    )
    def test_integers_and_numeric_strings_are_accepted(self, given, expected):
        assert ExperimentParameters(seeds=given).seeds == expected

    @pytest.mark.parametrize("bad", [["a"], [1.5], [True], [None], ["1.0"], [[1]], 1.5])
    def test_non_integer_seeds_are_rejected_when_the_config_is_loaded(self, bad):
        with pytest.raises(ValidationError, match="seeds"):
            ExperimentParameters(seeds=bad)

    def test_omitted_seeds_are_drawn_per_experiment(self):
        draws = {ExperimentParameters().seeds[0] for _ in range(20)}
        assert len(draws) > 1  # not frozen at import time

    def test_extra_keys_are_kept(self):
        assert ExperimentParameters(seeds=[1], custom="x").custom == "x"  # type: ignore[attr-defined]

    def test_integer_seeds_work_in_a_whole_experiment(self, config_dict, fake_experiment):
        config = {**config_dict, "parameters": {"seeds": [1, 2]}}
        experiment = rup.experiment_from_dict(config)
        assert experiment.parameters.seeds == ["1", "2"]
