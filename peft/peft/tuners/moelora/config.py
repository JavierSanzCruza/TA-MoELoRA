from ...tuners.lora.config import LoraConfig
from dataclasses import dataclass, field
from ...utils import PeftType

@dataclass
class MoELoraConfig(LoraConfig):
    """
    Add lora_nums
    """

    lora_nums: int = field(default=None, metadata={"help": "Numbers of Lora"})
    moe_type: str = field(default=None, metadata={"help": "Type of MoE layer"})
    task_embedding_model: str = field(default=None, metadata={"help": "Task embedding model"})
    task_id_mapping_path: str = field(default=None, metadata={"help": "Task id mapping path"})
    task_dim: int = field(default=None, metadata={"help": "Task embedding dimension"})
    turn_off_last_layer_expert: int = field(default=None, metadata={"help": "Turn off last layer expert"})
    def __post_init__(self):
        self.peft_type = PeftType.MOELORA