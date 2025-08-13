from typing import Dict, List, Type

from ..utils_typing import Entity, Relation, dataclass

"""Entity definitions"""


ENTITY_DEFINITIONS: List[Type] = [
    Entity
]


"""Relation definitions"""
@dataclass
class Nationality(Relation):
    """{nyt11_nationality}"""
    
    arg1: str
    arg2: str

@dataclass
class CaptialOfCountry(Relation):
    """{nyt11_captialofcountry}"""
    
    arg1: str
    arg2: str

@dataclass
class PlaceOfBirth(Relation):
    """{nyt11_placeofbirth}"""
    
    arg1: str
    arg2: str

@dataclass
class PlaceOfDeath(Relation):
    """{nyt11_placeofdeath}"""
    
    arg1: str
    arg2: str

@dataclass
class Children(Relation):
    """{nyt11_children}"""
    
    arg1: str
    arg2: str

@dataclass
class LocationContains(Relation):
    """{nyt11_locationcontains}"""
    
    arg1: str
    arg2: str

@dataclass
class PlaceLived(Relation):
    """{nyt11_placelived}"""
    
    arg1: str
    arg2: str

@dataclass
class AdministrativeDivisionsOfCountry(Relation):
    """{nyt11_administrativedivisionsofcountry}"""
    
    arg1: str
    arg2: str

@dataclass
class CountryOfAdministrativeDivisions(Relation):
    """{nyt11_countryofadministrativedivisions}"""
    
    arg1: str
    arg2: str

@dataclass
class WorkFor(Relation):
    """{nyt11_workfor}"""
    
    arg1: str
    arg2: str

@dataclass
class NeighborhoodOf(Relation):
    """{nyt11_neighborhoodof}"""
    
    arg1: str
    arg2: str

@dataclass
class FoundedBy(Relation):
    """{nyt11_foundedby}"""
    
    arg1: str
    arg2: str

COARSE_RELATION_DEFINITIONS: List[Type] = [
    Nationality,
    CaptialOfCountry,
    PlaceOfBirth,
    PlaceOfDeath,
    Children,
    LocationContains,
    PlaceLived,
    AdministrativeDivisionsOfCountry,
    CountryOfAdministrativeDivisions,
    WorkFor,
    NeighborhoodOf,
    FoundedBy 

]