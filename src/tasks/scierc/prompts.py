from typing import Dict, List, Type

from ..utils_typing import Entity, Relation, dataclass

"""Entity definitions"""
@dataclass
class Task(Entity):
    """{scierc_task}"""

    span: str # {scierc_task_examples}

@dataclass
class Material(Entity):
    """{scierc_material}"""

    span: str # {scierc_material_examples}

@dataclass
class Generic(Entity):
    """{scierc_generic}"""

    span: str # {scierc_generic_examples}

@dataclass
class Method(Entity):
    """{scierc_method}"""

    span: str # {scierc_method_examples}

@dataclass
class OtherScientificTerm(Entity):
    """{scierc_other_scientific_term}"""

    span: str # {scierc_other_scientific_term_examples}

@dataclass
class Metric(Entity):
    """{scierc_metric}"""

    span: str # {scierc_metric_examples}

ENTITY_DEFINITIONS: List[Type] = [
    Task,
    Material,
    Generic,
    Method,
    OtherScientificTerm,
    Metric,
]


"""Relation definitions"""

@dataclass
class PartOf(Relation):
    """{scierc_part_of}"""

    arg1: str 
    arg2: str 

@dataclass
class UsedFor(Relation):
    """{scierc_used_for}"""

    arg1: str 
    arg2: str

@dataclass
class HyponymOf(Relation):
    """{scierc_hyponym_of}"""

    arg1: str 
    arg2: str

@dataclass
class Conjuction(Relation):
    """{scierc_conjuction}"""

    arg1: str 
    arg2: str

@dataclass
class FeatureOf(Relation):
    """{scierc_feature_of}"""

    arg1: str 
    arg2: str

@dataclass
class Compare(Relation):
    """{scierc_compare}"""

    arg1: str 
    arg2: str

@dataclass
class EvaluateFor(Relation):
    """{scierc_evaluate_for}"""

    arg1: str 
    arg2: str

COARSE_RELATION_DEFINITIONS: List[Type] = [
    PartOf,
    UsedFor,
    HyponymOf,
    Conjuction,
    FeatureOf,
    Compare,
    EvaluateFor,
]