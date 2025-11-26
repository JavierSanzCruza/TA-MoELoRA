import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from ...utils import transpose

class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False

class LoraLinear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        lora_nums: int = 2,
        moe_type: str = 'dense',
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self._reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        
    def _reset_parameters(self):
        nn.Linear.reset_parameters(self)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def train(self, mode: bool = True):
        """Set the model in training mode"""
        nn.Module.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)

    def eval(self):
        nn.Module.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            raise ImportError(":(")
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            lora_output = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            result = result + lora_output.to(result.dtype)
        return result


class MoELoraLinear(nn.Linear, LoraLayer):
    # MoE Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_idx: int,
        layer_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        moe_type: str = 'dense',
        task_dim: int = 728,
        turn_off_last_layer_expert: int=None,
        **kwargs,
    ):

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.lora_nums = lora_nums
        self.fan_in_fan_out = fan_in_fan_out
        self.moe_type = moe_type
        self.layer_idx = layer_idx
        self.layer_name = layer_name
        self.turn_off_last_layer_expert = turn_off_last_layer_expert
        if self.turn_off_last_layer_expert is not None and self.layer_idx == 31 and self.layer_name == 'down_proj':
            print(f"Turn off the last layer expert {self.turn_off_last_layer_expert}")
        if self.training:
            pass
        else:
            self._reset_route_cache()
        # Actual trainable parameters
        if r > 0:
            if self.moe_type == 'dense':
                self.lora_route = nn.Linear(in_features, self.lora_nums, bias=False)

            elif 'pt_task_aware' in self.moe_type: 
                self.lora_route = nn.Linear(in_features+task_dim, self.lora_nums, bias=False)
            else:
                pass
            # Stack the weights for all experts
            self.lora_A_weight = nn.Parameter(torch.empty(self.lora_nums, in_features, self.r))
            self.lora_B_weight = nn.Parameter(torch.empty(self.lora_nums, self.r, out_features))

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self._reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        

    def _reset_route_cache(self):
        self.route_weights_sum = torch.zeros(self.lora_nums)
        self.count = 0
    
    def _reset_parameters(self):
        nn.Linear.reset_parameters(self)

        if hasattr(self, 'lora_A_weight'):
            nn.init.kaiming_uniform_(self.lora_A_weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_weight)
            
        if hasattr(self, 'lora_route'):
            nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))

    def train(self, mode: bool = True):
        """Set the model in training mode"""
        nn.Linear.train(self, mode)
        if hasattr(self, 'lora_route'):
            self.lora_route.train(mode)


    def eval(self):
        nn.Linear.eval(self)
        if hasattr(self, 'lora_route'):
            self.lora_route.eval()

    def forward(self, x: torch.Tensor, task_ids):
        if self.disable_adapters:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            raise ImportError(":(") 
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            
            if self.r > 0:
                if self.moe_type == 'dense':
                    route_weight = nn.functional.softmax(self.lora_route(x), dim=-1).to(result.dtype)
                    if hasattr(self, 'route_weights_sum'):
                        route_weight_slice = route_weight.mean(dim=1)
                elif 'pt_task_aware' in self.moe_type: 
                    task_embedding = task_ids.to(result.dtype)
                    seq_length = x.size(1)
                    task_embedding_expanded = task_embedding.unsqueeze(1).repeat(1, seq_length, 1)  # [batch_size, seq_length, task_embedding_dim]
                    
                    gating_input = torch.cat([x, task_embedding_expanded], dim=-1)  # [batch_size, seq_length, gating_input_dim]
                    logits = self.lora_route(gating_input)

                    if 'top2' in self.moe_type:
                        k = 2  
                        topk = torch.topk(logits, k, dim=-1)
                        topk_values, topk_indices = topk.values, topk.indices
                        # Create a mask with the same shape as the logits
                        mask = torch.zeros_like(logits, dtype=torch.bool)  
                        mask.scatter_(-1, topk_indices, True)  # Set the topk indices to True
                        # Set the values of the logits that are not in the topk to -inf
                        masked_logits = logits.masked_fill(~mask, float('-inf'))
                        # Compute the softmax
                        route_weight = nn.functional.softmax(masked_logits, dim=-1).to(result.dtype)
                        if hasattr(self, 'route_weights_sum'):
                            route_weight_slice = route_weight.mean(dim=1)
                    else:
                        route_weight = nn.functional.softmax(logits, dim=-1).to(result.dtype)
                        if hasattr(self, 'route_weights_sum'):
                            route_weight_slice = route_weight.mean(dim=1)
                else:
                    raise ValueError(f"Invalid moe_type {self.moe_type}")

                if self.turn_off_last_layer_expert is not None:
                    # modify the route weight of the last layer of the model
                    pass
                    
                if hasattr(self, 'route_weights_sum'):
                    self.route_weights_sum += route_weight_slice.sum(dim=0).detach().cpu()
                    self.count += route_weight_slice.size(0)
                x_dropped = self.lora_dropout(x) # Apply dropout
                x_lora_A = torch.einsum('bsi,nir->bsnr', x_dropped, self.lora_A_weight)
                x_lora_B = torch.einsum('bsnr,nro->bsno', x_lora_A, self.lora_B_weight)
                
                weighted_outputs = route_weight.unsqueeze(-1) * x_lora_B
                delta = weighted_outputs.sum(dim=2).to(result.dtype) * self.scaling
                result = result + delta


        return result