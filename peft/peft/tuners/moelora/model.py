import torch
import re
from .layer import MoELoraLinear, LoraLayer, LoraLinear
from dataclasses import asdict
from enum import Enum
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer
import torch
import json
from itertools import islice
import torch.nn.functional as F
import numpy as np
class MoELoraModel(torch.nn.Module):
    """
    Creates MoE-style Lora model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.
        adapter_name (str): The name of the adapter.


    Returns:
        `torch.nn.Module`: The Lora model.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.peft_config = config
        self.adapter_name = adapter_name
        self.model = model
        self._find_and_replace()
        self._mark_only_lora_as_trainable()
        self.task_embedding_model = self.peft_config[self.adapter_name].task_embedding_model
        self.first_flag = 1
        if self.task_embedding_model:
            print('loading task embedding model:', self.task_embedding_model)
            self.base_tokenizer = AutoTokenizer.from_pretrained(self.peft_config[self.adapter_name].base_model_name_or_path)
            self.task_model = AutoModel.from_pretrained(self.task_embedding_model, trust_remote_code=True).to(self.model.device)
            self.task_model.eval()
            self.task_tokenizer = AutoTokenizer.from_pretrained(self.task_embedding_model)
            for param in self.task_model.parameters():
                param.requires_grad = False
        else:
            self.forward = self.model.forward

    def get_task_embeddings(self, input_ids):
        input_texts = self.base_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # remove ground truth from input_texts
        if 'schema' in self.peft_config[self.adapter_name].moe_type:
            split_keyword = '# This is the text to analyze'
            input_texts = [text.split(split_keyword)[0] for text in input_texts]
        elif 'text' in self.peft_config[self.adapter_name].moe_type:
            split_keyword = '# This is the text to analyze'
            input_texts = [split_keyword + text.split(split_keyword)[-1] for text in input_texts]
            split_keyword = 'result ='
            input_texts = [text.split(split_keyword)[0]+split_keyword for text in input_texts]
        else:
            split_keyword = 'result ='
            input_texts = [text.split(split_keyword)[0]+split_keyword for text in input_texts]
        if self.first_flag:
            print(input_texts)
            self.first_flag = 0
        inputs = self.task_tokenizer(input_texts, return_tensors='pt', truncation=True, padding=True).to(self.model.device)
        if 'jina' in self.task_embedding_model:
            with torch.no_grad():
                encoder_outputs = self.task_model(**inputs)
            token_embeddings = encoder_outputs[0]
            attention_mask = inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        elif 'codet5p' in self.task_embedding_model:
            with torch.no_grad():
                encoder_outputs = self.task_model(**inputs)
            embeddings = encoder_outputs
        elif 'bge' in self.task_embedding_model:  
            with torch.no_grad():
                encoder_outputs = self.task_model(**inputs)
                sentence_embeddings = encoder_outputs[0][:, 0]
            embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        elif 'gte' in self.task_embedding_model:
            with torch.no_grad():
                encoder_outputs = self.task_model(**inputs)
                embeddings = encoder_outputs.last_hidden_state[:, 0]
        elif 'codebert' in self.task_embedding_model:
            with torch.no_grad():
                encoder_outputs = self.task_model(**inputs)
                embeddings = encoder_outputs.last_hidden_state[:, 0, :] # cls token
        else:
            with torch.no_grad():
                encoder_outputs = self.task_model.encoder(**inputs)
                embeddings = encoder_outputs.last_hidden_state.mean(dim=1)
            
        return embeddings
    
    def forward(  
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_padded_inputs: Optional[bool] = False,
        task_ids=None,
    ):
        task_ids = self.get_task_embeddings(input_ids)


        return self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_padded_inputs=is_padded_inputs,
            task_ids=task_ids,
        )


    def _find_and_replace(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit):
            raise ImportError(
                "To use Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        config = self.peft_config[self.adapter_name]
        kwargs = {
            "r": config.r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "lora_nums": config.lora_nums,
            "moe_type": config.moe_type,
            "fan_in_fan_out": config.fan_in_fan_out,
            "task_dim": config.task_dim,
            "turn_off_last_layer_expert": config.turn_off_last_layer_expert,  
            #"merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode) and not is_hf_device_map_available,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(config.target_modules, str):
                target_module_found = re.fullmatch(config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in config.target_modules)
            if target_module_found: # here
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name, layer_idx = self._get_submodules(key)
                bias = target.bias is not None

                if isinstance(target, torch.nn.Linear):
                    
                    # if 'mlp' in key:
                    new_module = MoELoraLinear(target.in_features, target.out_features, bias=bias, layer_idx=layer_idx, layer_name=target_name, **kwargs)
                    # elif 'self_attn' in key:
                    #     new_module = LoraLinear(target.in_features, target.out_features, bias=bias, **kwargs)
                    # else:
                    #     raise ValueError(f"Unsupported module {key} for MoELoraModel.")
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        layer_idx = int(re.search(r"\d+", key).group())
        target = self.model.get_submodule(key)
        return parent, target, target_name, layer_idx
    

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def _mark_only_lora_as_trainable(self) -> None:
        config = self.peft_config[self.adapter_name]
        for n, p in self.model.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False
        if config.bias == "none":
            return
        elif config.bias == "all":
            for n, p in self.model.named_parameters():
                if "bias" in n:
                    p.requires_grad = True
        elif config.bias == "lora_only":
            for m in self.model.modules():
                if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad = True
        else:
            raise NotImplementedError