# ===========================================================================
#                             Persona Management Mixin
# ===========================================================================
# This module defines a mixin class for managing personas within an experiment.
# It provides methods to add and manage demographic profiles (personas), ensuring
# that they are correctly handled, stored, and accessed.


import warnings
from rupsycho.models.questionnaire import DemographicProfile
from typing import Optional


class PersonaManagementMixin:
    """
    Mixin providing methods to manage personas in the experiment.

    This includes adding personas (demographic profiles) and ensuring they are
    stored in a dictionary with unique identifiers.
    """

    def add_persona(self, persona: DemographicProfile, identifier: Optional[str] = None) -> None:
        """
        Adds a demographic profile (persona) to the experiment.

        :param persona: The DemographicProfile instance to be added.
        :param identifier: Optional identifier for the persona. If not provided,
                           a unique identifier is generated based on the persona's id.
        """
        key = identifier if identifier else str(id(persona))
        if key in self.demographic_profiles:
            warnings.warn(
                f"A persona with the identifier '{key}' already exists.", UserWarning)
        else:
            self.demographic_profiles[key] = persona

    def get_persona(self, identifier: str) -> Optional[DemographicProfile]:
        """
        Retrieves a persona by its identifier.

        :param identifier: The identifier of the persona.
        :return: The DemographicProfile instance if found, otherwise None.
        """
        return self.demographic_profiles.get(identifier, None)

    def remove_persona(self, identifier: str) -> None:
        """
        Removes a persona from the experiment by its identifier.

        :param identifier: The identifier of the persona to be removed.
        """
        if identifier in self.demographic_profiles:
            del self.demographic_profiles[identifier]
        else:
            warnings.warn(
                f"No persona found with the identifier '{identifier}'.", UserWarning)

    def list_personas(self) -> list:
        """
        Lists all persona identifiers in the experiment.

        :return: A list of persona identifiers.
        """
        return list(self.demographic_profiles.keys())

    def clear_personas(self) -> None:
        """
        Removes all personas from the experiment.
        """
        self.demographic_profiles.clear()
