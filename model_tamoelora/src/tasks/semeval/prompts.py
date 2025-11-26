from typing import Dict, List, Type

from ..utils_typing import Entity, Relation, dataclass

"""Entity definitions"""


ENTITY_DEFINITIONS: List[Type] = [
    Entity
]


"""Relation definitions"""

@dataclass
class CauseEffect(Relation):
    """{semeval_causeeffect}"""

    arg1: str
    arg2: str

@dataclass
class InstrumentAgency(Relation):
    """{semeval_instrumentagency}"""

    arg1: str
    arg2: str

@dataclass
class ProductProducer(Relation):
    """{semeval_productproducer}"""

    arg1: str
    arg2: str

@dataclass
class ContentContainer(Relation):
    """{semeval_contentcontainer}"""

    arg1: str
    arg2: str

@dataclass
class EntityOrigin(Relation):
    """{semeval_entityorigin}"""

    arg1: str
    arg2: str

@dataclass
class EntityDestination(Relation):
    """{semeval_entitydestination}"""

    arg1: str
    arg2: str

@dataclass
class ComponentWhole(Relation):
    """{semeval_componentwhole}"""

    arg1: str
    arg2: str

@dataclass
class MemberCollection(Relation):
    """{semeval_membercollection}"""

    arg1: str
    arg2: str

@dataclass
class MessageTopic(Relation):
    """{semeval_messagetopic}"""

    arg1: str
    arg2: str

@dataclass
class Else(Relation):
    """{semeval_else}"""

    arg1: str
    arg2: str

COARSE_RELATION_DEFINITIONS: List[Type] = [
    CauseEffect,
    InstrumentAgency,
    ProductProducer,
    ContentContainer,
    EntityOrigin,
    EntityDestination,
    ComponentWhole,
    MemberCollection,
    MessageTopic,
    Else,
]