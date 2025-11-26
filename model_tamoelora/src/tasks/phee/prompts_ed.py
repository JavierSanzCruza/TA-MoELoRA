from typing import List, Type

from ..utils_typing import Event, dataclass



@dataclass
class PotentialTherapeutic(Event):
    """{potential_therapeutic_main}"""

    mention: str
    """The text span that triggers the event.
    {potential_therapeutic_examples}
    """

@dataclass
class Adverse(Event):
    """{adverse_main}"""

    mention: str
    """The text span that triggers the event.
    {adverse_examples}
    """


ED_EVENT_DEFINITIONS: List[Type] = [
    PotentialTherapeutic,
    Adverse,
]
