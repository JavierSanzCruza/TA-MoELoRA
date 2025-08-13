from typing import Dict, List, Type

from ..utils_typing import Entity, Relation, dataclass

"""Entity definitions"""


ENTITY_DEFINITIONS: List[Type] = [
    Entity
]


"""Relation definitions"""

@dataclass
class CityOfResidence(Relation):
    """{kbp37_cities_of_residence}"""
    
    arg1: str
    arg2: str

@dataclass
class CountryOfResidence(Relation):
    """{kbp37_countries_of_residence}"""
    
    arg1: str
    arg2: str

@dataclass
class TopMembersEmployees(Relation):
    """{kbp37_top_members_employees}"""
    
    arg1: str
    arg2: str

@dataclass
class MemberOf(Relation):
    """{kbp37_member_of}"""
    
    arg1: str
    arg2: str

@dataclass
class AlternateName(Relation):
    """{kbp37_alternate_names}"""
    
    arg1: str
    arg2: str

@dataclass
class Origin(Relation):
    """{kbp37_origin}"""
    
    arg1: str
    arg2: str

@dataclass
class StateOrProvinceOfResidence(Relation):
    """{kbp37_state_or_province_of_residence}"""
    
    arg1: str
    arg2: str

@dataclass
class TitleOfPerson(Relation):
    """{kbp37_title_of_person}"""
    
    arg1: str
    arg2: str

@dataclass
class Spouse(Relation):
    """{kbp37_spouse}"""
    
    arg1: str
    arg2: str

@dataclass
class CityOfHeadquarters(Relation):
    """{kbp37_city_of_headquarters}"""
    
    arg1: str
    arg2: str

@dataclass
class FoundedBy(Relation):
    """{kbp37_founded_by}"""
    
    arg1: str
    arg2: str

@dataclass
class StateOrProvinceOfHeadquarters(Relation):
    """{kbp37_state_or_province_of_headquarters}"""
    
    arg1: str
    arg2: str

@dataclass
class EmployeeOf(Relation):
    """{kbp37_employee_of}"""
    
    arg1: str
    arg2: str

@dataclass
class CountryOfBirth(Relation):
    """{kbp37_country_of_birth}"""
    
    arg1: str
    arg2: str

@dataclass
class Founded(Relation):
    """{kbp37_founded}"""
    
    arg1: str
    arg2: str

@dataclass
class Subsidiary(Relation):
    """{kbp37_subsidiary}"""
    
    arg1: str
    arg2: str

@dataclass
class CountryOfHeadquarters(Relation):
    """{kbp37_country_of_headquarters}"""
    
    arg1: str
    arg2: str

COARSE_RELATION_DEFINITIONS: List[Type] = [
    CityOfResidence,
    CountryOfResidence,
    TopMembersEmployees,
    MemberOf,
    AlternateName,
    Origin,
    StateOrProvinceOfResidence,
    TitleOfPerson,
    Spouse,
    CityOfHeadquarters,
    FoundedBy,
    StateOrProvinceOfHeadquarters,
    EmployeeOf,
    CountryOfBirth,
    Founded,
    Subsidiary,
    CountryOfHeadquarters
]