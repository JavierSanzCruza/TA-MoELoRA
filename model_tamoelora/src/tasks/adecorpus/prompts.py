from typing import Dict, List, Type

from ..utils_typing import Entity, Relation, dataclass

"""Entity definitions"""
@dataclass
class Drug(Entity):
    """{ade_drug}"""

    span: str # {ade_drug_examples}

@dataclass
class Effect(Entity):
    """{ade_effect}"""

    span: str # {ade_effect_examples}


ENTITY_DEFINITIONS: List[Type] = [
    Drug,
    Effect,
]


"""Relation definitions"""

@dataclass
class AdverseEffect(Relation):
    """{ade_adverse_effect}"""

    arg1: str 
    arg2: str 

COARSE_RELATION_DEFINITIONS: List[Type] = [
    AdverseEffect,
]