import json
from collections import defaultdict
from typing import Tuple, Union
import itertools
import rich

from src.tasks.conll04.guidelines import GUIDELINES
# from src.tasks.conll04.guidelines_gold import EXAMPLES
from src.tasks.conll04.prompts import (
    ENTITY_DEFINITIONS,
    COARSE_RELATION_DEFINITIONS,
    Person,
    Organization,
    Location,
    WorkFor,
    Kill,
    LocatedIn,
    OrganizationBasedIn,
    LiveIn,

)

from ..utils_data import DatasetLoader, Sampler
from ..utils_typing import Relation, Template, dataclass


@dataclass
class NoneRelation(Relation):
    arg1: str
    arg2: str


class Conll04DatasetLoader(DatasetLoader):
    """
    A `DatasetLoader` for the Conll04 dataset.

    Args:
        path (`str`):
            The location of the dataset directory.

    Raises:
        `ValueError`:
            raised when a not defined value found.
    """
    ENTITY_TO_CLASS_MAPPING = {
        "Loc": Location,
        "Org": Organization,
        "Peop": Person
    }
    RELATION_TO_CLASS_MAPPING = {
        "Kill": Kill,
        "Live_In": LiveIn,
        "Located_In": LocatedIn,
        "OrgBased_In": OrganizationBasedIn,
        "Work_For": WorkFor,
    }
    
    

    def __init__(self, path: str, **kwargs) -> None:
        self.elements = {}
        lines = []
        with open(path, "r") as in_f:
            for line in in_f:
                line = json.loads(line)
                lines.append(line)
            
        for i, line in enumerate(lines):
            line_relations = line['relation']
            sentence = line['text']
            key = str(i)
            for triplet in line_relations:
                if key not in self.elements:
                    self.elements[key] = {
                        "id": key,
                        "doc_id": key,
                        "text": sentence,
                        "entities": [],
                        "coarse_relations": [],
                        "relations": [],
                        "gold": [],
                    }
                # print(triplet)
                entities_span = [triplet['head'], triplet['tail']]
                entities_types = [triplet['head_type'], triplet['tail_type']]
                label = triplet['relation']
                entities = [
                        self.ENTITY_TO_CLASS_MAPPING[entities_types[i]](span=entity)
                        for i, entity in enumerate(entities_span)
                        if entities_types[i] in self.ENTITY_TO_CLASS_MAPPING
                ]
                
                coarse_relations, relations = [], []

                
                if label in self.RELATION_TO_CLASS_MAPPING:
                    coarse_relations.append(
                        self.RELATION_TO_CLASS_MAPPING[label](
                            arg1=entities_span[0],
                            arg2=entities_span[1],
                        )
                    )

                self.elements[key]["entities"] += entities
                self.elements[key]["coarse_relations"] += coarse_relations
                self.elements[key]["relations"] += relations
                self.elements[key]["gold"] += entities  # Is not used anyway


class Conll04Sampler(Sampler):
    """
    A data `Sampler` for the Conll04 dataset.

    Args:
        dataset_loader (`Conll04Sampler`):
            The dataset loader that contains the data information.
        task (`str`, optional):
            The task to sample. It must be one of the following: NER, VER, RE, EE.
            Defaults to `None`.
        split (`str`, optional):
            The split to sample. It must be one of the following: "train", "dev" or
            "test". Depending on the split the sampling strategy differs. Defaults to
            `"train"`.
        parallel_instances (`Union[int, Tuple[int, int]]`, optional):
            The number of sentences sampled in parallel. Options:

                * **`int`**: The amount of elements that will be sampled in parallel.
                * **`tuple`**: The range of elements that will be sampled in parallel.

            Defaults to 1.
        max_guidelines (`int`, optional):
            The number of guidelines to append to the example at the same time. If `-1`
            is given then all the guidelines are appended. Defaults to `-1`.
        guideline_dropout (`float`, optional):
            The probability to dropout a guideline definition for the given example. This
            is only applied on training. Defaults to `0.0`.
        seed (`float`, optional):
            The seed to sample the examples. Defaults to `0`.
        prompt_template (`str`, optional):
            The path to the prompt template. Defaults to `"templates/prompt.txt"`.
        ensure_positives_on_train (bool, optional):
            Whether to ensure that the guidelines of annotated examples are not removed.
            Defaults to `False`.
        dataset_name (str, optional):
            The name of the dataset. Defaults to `None`.
        scorer (`str`, optional):
           The scorer class import string. Defaults to `None`.
        sample_only_gold_guidelines (`bool`, optional):
            Whether to sample only guidelines of present annotations. Defaults to `False`.
    """
    
    def __init__(
        self,
        dataset_loader: Conll04DatasetLoader,
        task: str = None,
        split: str = "train",
        parallel_instances: Union[int, Tuple[int, int]] = 1,
        max_guidelines: int = -1,
        guideline_dropout: float = 0.0,
        seed: float = 0,
        ensure_positives_on_train: bool = True,
        dataset_name: str = None,
        scorer: str = None,
        sample_only_gold_guidelines: bool = False,
        **kwargs,
    ) -> None:
        assert task in [
            "NER",
            "RE",
        ], f"{task} must be either 'NER', 'VER', 'RE', 'RC', 'EE', 'EAE'."

        task_definitions, task_target, task_template = {
            "NER": (ENTITY_DEFINITIONS, "entities", "templates/prompt.txt"),
            "RE": (COARSE_RELATION_DEFINITIONS, "coarse_relations", "templates/prompt_ace_re.txt"),
        }[task]

        is_coarse_to_fine = False
        COARSE_TO_FINE = None
        FINE_TO_COARSE = None

        kwargs.pop("prompt_template")

        super().__init__(
            dataset_loader=dataset_loader,
            task=task,
            split=split,
            parallel_instances=parallel_instances,
            max_guidelines=max_guidelines,
            guideline_dropout=guideline_dropout,
            seed=seed,
            prompt_template=task_template,
            ensure_positives_on_train=ensure_positives_on_train,
            sample_only_gold_guidelines=sample_only_gold_guidelines,
            dataset_name=dataset_name,
            scorer=scorer,
            task_definitions=task_definitions,
            task_target=task_target,
            is_coarse_to_fine=is_coarse_to_fine,
            coarse_to_fine=COARSE_TO_FINE,
            fine_to_coarse=FINE_TO_COARSE,
            definitions=GUIDELINES,
            # examples=EXAMPLES,
            **kwargs,
        )

