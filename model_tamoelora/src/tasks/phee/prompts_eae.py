from typing import List, Type

from ..utils_typing import Event, dataclass


"""Event definitions

The events definitions are derived from the original paper:
https://aclanthology.org/2022.emnlp-main.376.pdf
 ['Subject.Age',
   'Subject.Disorder',
   'Effect',
   'Treatment.Drug',
   'Treatment.Time_elapsed',
   'Subject.Population',
   'Subject.Gender',
   'Treatment.Freq',
   'Treatment.Disorder',
   'Treatment.Duration',
   'Treatment.Dosage',
   'Treatment',
   'Combination.Drug',
   'Subject',
   'Subject.Race',
   'Treatment.Route']}
"""



@dataclass 
class PotentialTherapeutic(Event):
    """{potential_therapeutic_main}"""

    mention: str
    """The text span that triggers the event.
    {potential_therapeutic_examples}
    """
    subject: List[str] # The paitent involved in the medical event
    subject_age: List[str] # The concrete age or span that indicates an age range of the patient
    subject_gender: List[str] # The span that indicates the subject's gender 
    subject_population: List[str] # The number of patients receiving the treatment
    subject_race: List[str] # The span that indicates the subject's race/nationality
    subject_disorder: List[str] # Preexisting conditions, or disorders that the patient suffers other than the target disorder of the treatment
    effect: List[str] # The outcome of the treatment
    treatment: List[str] # The description of the therapy administered to the patients
    treatment_drug: List[str] # The drug used as therapy in the event
    treatment_dosage: List[str] # The amount of drug is given
    treatment_freq: List[str] # The frequency of drug use
    treatment_route: List[str] # The route of drug administration
    treatment_time_elapsed: List[str] # The time elapsed after the drug was administered to the occurrence of the (side) event
    treatment_duration: List[str] # How long the patient has been taking the medicine (usually for long-term medication)
    treatment_disorder: List[str] # The target disorder of the medicine administration
    combination_drug: List[str] # The additional drugs administered alongside the main treatment when multiple drugs are used

@dataclass
class Adverse(Event):
    """{adverse_main}"""

    mention: str
    """The text span that triggers the event.
    {adverse_examples}
    """
    subject: List[str] # The paitent involved in the medical event
    subject_age: List[str] # The concrete age or span that indicates an age range of the patient
    subject_gender: List[str] # The span that indicates the subject's gender 
    subject_population: List[str] # The number of patients receiving the treatment
    subject_race: List[str] # The span that indicates the subject's race/nationality
    subject_disorder: List[str] # Preexisting conditions, or disorders that the patient suffers other than the target disorder of the treatment
    effect: List[str] # The outcome of the treatment
    treatment: List[str] # The description of the therapy administered to the patients
    treatment_drug: List[str] # The drug used as therapy in the event
    treatment_dosage: List[str] # The amount of drug is given
    treatment_freq: List[str] # The frequency of drug use
    treatment_route: List[str] # The route of drug administration
    treatment_time_elapsed: List[str] # The time elapsed after the drug was administered to the occurrence of the (side) event
    treatment_duration: List[str] # How long the patient has been taking the medicine (usually for long-term medication)
    treatment_disorder: List[str] # The target disorder of the medicine administration
    combination_drug: List[str] # The additional drugs administered alongside the main treatment when multiple drugs are used

    


EAE_EVENT_DEFINITIONS: List[Type] = [
    PotentialTherapeutic,
    Adverse,
]
