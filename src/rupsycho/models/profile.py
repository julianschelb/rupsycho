# ===========================================================================
#               Data Model: Demographic Profile
# ===========================================================================
# This file contains the data model for a demographic profile.

# from typing import Any, List, Dict
# from pydantic import BaseModel


# # class ProfileAttributes(BaseModel):
# #     age: int
# #     gender: str

# #     class Config:
# #         """Pydantic model configuration."""
# #         arbitrary_types_allowed = True
# #         extra = 'allow'  # Allow extra fields


# class DemographicProfile(BaseModel):
#     attributes: Dict[str, Any]
#     template: str

#     class Config:
#         """Pydantic model configuration."""
#         arbitrary_types_allowed = True
#         extra = 'allow'  # Allow extra fields

#     def get_profile_desc(self):
#         """ Returns a string representation of the profile. """
#         return self.__str__()


# if __name__ == '__main__':
#     # Example usage
#     profile = DemographicProfile(
#         attributes={"age": 18, "name": "John Doe"},
#         template="My name is {name} and I am {age} years old."
#     )
#     profile_desc = profile.get_profile_desc()
#     print(profile_desc)
