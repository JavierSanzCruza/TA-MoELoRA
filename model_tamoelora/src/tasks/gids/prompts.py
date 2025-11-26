from typing import Dict, List, Type

from ..utils_typing import Entity, Relation, dataclass

"""Entity definitions"""
@dataclass
class Person(Entity):
    """{gids_person}"""
    
    span: str

@dataclass
class Place(Entity):
    """{gids_place}"""
    
    span: str

@dataclass
class EducationInstitution(Entity):
    """{gids_educationinstitution}"""
    
    span: str

@dataclass
class EducationalDegree(Entity):
    """{gids_educationaldegree}"""
    
    span: str


ENTITY_DEFINITIONS: List[Type] = [
    Person,
    Place,
    EducationInstitution,
    EducationalDegree
]


"""Relation definitions"""


@dataclass
class GraduatedFrom(Relation):
    """{gids_graduatedfrom}"""
    
    arg1: str
    arg2: str

@dataclass
class HasDegree(Relation):
    """{gids_hasdegree}"""
    
    arg1: str
    arg2: str

@dataclass
class PlaceOfBirth(Relation):
    """{gids_placeofbirth}"""
    
    arg1: str
    arg2: str

@dataclass
class PlaceOfDeath(Relation):
    """{gids_placeofdeath}"""
    
    arg1: str
    arg2: str


COARSE_RELATION_DEFINITIONS: List[Type] = [
    GraduatedFrom,
    HasDegree,
    PlaceOfBirth,
    PlaceOfDeath
]