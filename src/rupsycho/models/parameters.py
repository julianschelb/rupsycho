# ===========================================================================
#               Data Model: Experiment Parameters
# ===========================================================================
# This file contains the data model for the parameters used in experiments.


from random import randint
from pydantic import BaseModel, Field
from typing import List, Optional


class ExperimentParameters(BaseModel):
    """Represents the parameters for an experiment."""

    # ------------------- Experiment Parameters -------------------

    seeds: Optional[List[str]] = Field(
        default=[str(randint(0, 999999))],
        description="A list of seeds for random number generation in the experiment."
    )

    # output_directory: Optional[str] = Field(
    #     default="./output",
    #     description="Directory path where the experiment results will be saved."
    # )

    # output_format: Optional[str] = Field(
    #     default="csv",
    #     description="Format for saving the experiment results. E.g., 'csv', 'json', etc."
    # )

    lazy_load_models: bool = Field(
        default=True,
        description="If True, models will be loaded only when they are needed during the experiment."
    )

    class Config:
        extra = 'allow'  # Allow extra fields
