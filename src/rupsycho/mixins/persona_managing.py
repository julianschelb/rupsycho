# persona_managing.py
"""Add, inspect and remove the personas (demographic profiles) of an experiment."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from rupsycho.models.questionnaire import DemographicProfile

__all__ = ["PersonaManagementMixin"]


class PersonaManagementMixin:
    """Methods to manage the personas of an experiment.

    Personas are the demographic profiles the models answer *as*. They are stored in
    ``demographic_profiles`` under unique identifiers, which also label the rows of the results.
    """

    if TYPE_CHECKING:
        # Provided by ExperimentDocument, which mixes this class in.
        demographic_profiles: dict[str, Any]

    def add_persona(self, persona: DemographicProfile, identifier: str | None = None) -> None:
        """Add a persona to the experiment.

        Args:
            persona: The demographic profile to add.
            identifier: Name of the persona in the results. Defaults to the object id.

        Example:
            ```python
            from rupsycho.models.questionnaire import DemographicProfile

            experiment.add_persona(
                DemographicProfile(
                    attributes={"name": "Alex", "age": 34},
                    template="{name} is {age} years old.",
                ),
                identifier="Alex",
            )
            ```

        Note:
            Adding a persona under an identifier that exists replaces it and emits a warning.
        """
        key = identifier if identifier else str(id(persona))
        if key in self.demographic_profiles:
            warnings.warn(
                f"A persona with the identifier '{key}' already exists and is replaced.",
                UserWarning,
                stacklevel=2,
            )
        self.demographic_profiles[key] = persona

    def get_persona(self, identifier: str) -> DemographicProfile | None:
        """Return the persona with this identifier, or ``None`` if there is none."""
        return self.demographic_profiles.get(identifier, None)

    def remove_persona(self, identifier: str) -> None:
        """Remove a persona.

        Args:
            identifier: Identifier of the persona. Unknown identifiers only emit a warning.
        """
        if identifier in self.demographic_profiles:
            del self.demographic_profiles[identifier]
        else:
            warnings.warn(
                f"No persona found with the identifier '{identifier}'.", UserWarning, stacklevel=2
            )

    def list_personas(self) -> list[str]:
        """Return the identifiers of all personas."""
        return list(self.demographic_profiles.keys())

    def clear_personas(self) -> None:
        """Remove all personas from the experiment."""
        self.demographic_profiles.clear()
