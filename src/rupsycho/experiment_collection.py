# ===========================================================================
#                      ExperimentCollection Class Implementation
# ===========================================================================
# This module defines the ExperimentCollection class, designed to store and
# manage a collection of ExperimentDocument objects. The class includes
# methods for processing all experiments within the collection and handling
# metadata associated with the collection.


from typing import List, Any, Literal
from langchain_core.documents.base import BaseMedia
from rupsycho.experiment import ExperimentDocument
from rupsycho.utils import import_tqdm
tqdm = import_tqdm()  # Import tqdm based on the environment

# ================================= Experiments Class ================================


class ExperimentCollection(BaseMedia):
    """Class for storing a collection of ExperimentDocument objects along with metadata.

    Example:

        .. code-block:: python

            from my_module import ExperimentCollection, ExperimentDocument

            experiment_doc1 = ExperimentDocument(
                name="Experiment 1",
                description="Test Experiment",
                demographic_profiles=[...],
                models=[...],
                questionnaire=...,
                metadata={"source": "Lab A"}
            )

            experiment_doc2 = ExperimentDocument(
                name="Experiment 2",
                description="Another Test",
                demographic_profiles=[...],
                models=[...],
                questionnaire=...,
                metadata={"source": "Lab B"}
            )

            experiment_collection = ExperimentCollection(
                experiments=[experiment_doc1, experiment_doc2],
                metadata={"source": "Lab Collection"}
            )
    """

    # --------------------------------- Attributes --------------------------------

    experiments: List[ExperimentDocument]
    """List of ExperimentDocument objects representing the experiments."""
    type: Literal["ExperimentCollection"] = "ExperimentCollection"

    # --------------------------------- Config --------------------------------
    # Adding Pydantic Config to allow custom types
    class Config:
        arbitrary_types_allowed = True

    # --------------------------------- Initialization --------------------------------

    def __init__(self, experiments: List[ExperimentDocument], **kwargs: Any) -> None:
        """Initialize an ExperimentCollection with a list of ExperimentDocument objects."""
        super().__init__(experiments=experiments, **kwargs)

    # --------------------------------- Other Methods --------------------------------

    def __str__(self) -> str:
        """Override __str__ to restrict it to the experiments and metadata."""
        return f"experiments=[{', '.join(str(exp) for exp in self.experiments)}], metadata={self.metadata}"

    def __len__(self) -> int:
        """Return the number of ExperimentDocument objects in the collection."""
        return len(self.experiments)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the LangChain object."""
        return ["langchain", "schema", "experiment_collection"]

    # --------------------------------- Run Experiments --------------------------------

    def run_all(self) -> None:
        """
        Run the experiment processing for all ExperimentDocuments in the collection.
        """

        for experiment_doc in self.experiments:
            experiment_doc.run()
