from typing import Dict, List, Type

from src.tasks.conll04.prompts import (
    COARSE_RELATION_DEFINITIONS,
    ENTITY_DEFINITIONS,
)
from src.tasks.utils_scorer import EventScorer, RelationScorer, SpanScorer
from src.tasks.utils_typing import Entity, Value


class Conll04CoarseRelationScorer(RelationScorer):
    """Conll04 Relation identification scorer."""

    valid_types: List[Type] = COARSE_RELATION_DEFINITIONS