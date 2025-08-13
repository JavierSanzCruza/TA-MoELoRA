from typing import List, Type

from ..utils_typing import Entity, Relation, dataclass


@dataclass
class Person(Entity):
    """{conll_person}"""

    span: str  # {ner_person_examples}


@dataclass
class Organization(Entity):
    """{conll_organization}"""

    span: str  # {ner_organization_examples}


@dataclass
class Location(Entity):
    """{conll_location}"""

    span: str  # {ner_location_examples}


@dataclass
class Miscellaneous(Entity):
    """{conll_miscellaneous}"""

    span: str  # {ner_miscellaneous_examples}


ENTITY_DEFINITIONS: List[Entity] = [
    Person,
    Organization,
    Location,
    Miscellaneous,
]

ENTITY_DEFINITIONS_woMISC: List[Entity] = [
    Person,
    Organization,
    Location,
]

"""Relation definitions"""

@dataclass
class WorkFor(Relation):
    """{conll_workfor}"""
        
    arg1: str
    arg2: str

@dataclass
class Kill(Relation):
    """{conll_kill}"""

    arg1: str 
    arg2: str

@dataclass
class LocatedIn(Relation):
    """{conll_locatedin}"""

    arg1: str
    arg2: str

@dataclass
class OrganizationBasedIn(Relation):
    """{conll_orgbasedin}"""

    arg1: str
    arg2: str

@dataclass
class LiveIn(Relation):
    """{conll_livein}"""

    arg1: str
    arg2: str

COARSE_RELATION_DEFINITIONS: List[Type] = [
    WorkFor,
    Kill,
    LocatedIn,
    OrganizationBasedIn,
    LiveIn,
]
