# ===========================================================================
#               Data Model: Demographic Profile
# ===========================================================================
# This file contains the data model for a demographic profile.

from typing import Any, List, Dict
from pydantic import BaseModel


# class ProfileAttributes(BaseModel):
#     age: int
#     gender: str

#     class Config:
#         """Pydantic model configuration."""
#         arbitrary_types_allowed = True
#         extra = 'allow'  # Allow extra fields


class DemographicProfile(BaseModel):
    attributes: Dict[str, Any]
    template: str

    class Config:
        """Pydantic model configuration."""
        arbitrary_types_allowed = True
        extra = 'allow'  # Allow extra fields
