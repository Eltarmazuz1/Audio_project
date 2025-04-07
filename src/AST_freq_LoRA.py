import torch
import torch.nn as nn
import math
from transformers import ASTModel
from dataclasses import dataclass
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import (
    ASTLayer, ASTEncoder, ASTAttention, ASTSelfAttention, ASTSelfOutput
)
from typing import Optional, Tuple, Union

def init_ssf_scale_shift(blocks, dim):
    """
    SSF: Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning
    https://github.com/dongzelian/SSF/blob/main/models/vision_transformer.py
    """
    scale = nn.Parameter(torch.ones(blocks, dim))
    shift = nn.Parameter(torch.zeros(blocks, dim))

    nn.init.normal_(scale, mean=1, std=.02)
    nn.init.normal_(shift, std=.02)

    return scale, shift

def ssf_ada(x, scale, shift):
    """
    SSF: Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning
    https://github.com/dongzelian/SSF/blob/main/models/vision_transformer.py
    """
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')
        
# FreqFit Module
class FreqFit(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.block = 8
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.scale, mean=1, std=.02)
        nn.init.normal_(self.shift, std=.02)
        self.complex_weight = nn.Parameter(torch.randn(self.block, h, dim, 2, dtype=torch.float32) * 0.02)
        self.ssf_scale, self.ssf_shift = init_ssf_scale_shift(self.block, dim)
    
    def forward(self, x):
        B, a, C = x.shape
        x = x.to(torch.float32)
        res = x
        dimension_fourie =  1
        x = torch.fft.rfft(x, dim=dimension_fourie, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight[0].squeeze())

        weight_real = torch.nn.functional.interpolate(weight.unsqueeze(0).unsqueeze(0).real, size=(x.shape[1], C), mode='bilinear', align_corners=False).squeeze(0)
        weight_imag = torch.nn.functional.interpolate(weight.unsqueeze(0).unsqueeze(0).imag, size=(x.shape[1], C), mode='bilinear', align_corners=False).squeeze(0)
       
        
        weight = torch.complex(weight_real, weight_imag)
        weight = weight.expand(B, -1, -1)  # Ensure same batch size

        x = x * weight
        x = torch.fft.irfft(x, n=a, dim=dimension_fourie, norm='ortho')
        x = ssf_ada(x, self.ssf_scale[0], self.ssf_shift[0])
        x = x + res
        return x


    # def forward(self, x):
    #     B, N, C = x.shape
    #     # a = b = int(math.sqrt(N))  # Assuming square spatial dimensions
    #     def infer_spatial_dims(N):
    #         a = int(math.sqrt(N))
    #         while N % a != 0 and a > 1:
    #             a -= 1  # Find the largest divisor of N closest to sqrt(N)
    #         b = N // a  # Ensure a * b = N
    #         return a, b

    #     a, b = infer_spatial_dims(N)

    #     print(f"x.shape before view: {x.shape}, expected: ({B}, {a}, {b}, {C})")
    #     x = x.view(B, a, b, C).to(torch.float32)
    #     res = x
    #     x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
    #     # weight = torch.view_as_complex(self.filter_weight.squeeze())
    #     # Ensure weight matches x.shape
    #     # weight = torch.view_as_complex(self.filter_weight[:a, :b, :C].contiguous())
    #     # Resize weight to match x.shape[2] (24) instead of 8
    #     # weight = torch.view_as_complex(self.filter_weight[:, :x.shape[2], :C].contiguous())
    #     # weight = torch.view_as_complex(self.filter_weight[:x.shape[1], :x.shape[2], :C].contiguous())
    #     # Interpolate weight along dimension 2 to match x.shape[2]
    #     weight = torch.nn.functional.interpolate(
    #         # self.filter_weight.permute(2, 0, 1).unsqueeze(0),  # (1, 768, 10, 8)
    #         self.filter_weight.permute(3, 2, 0, 1).unsqueeze(0),
    #         size=(10, x.shape[2]),  # Resize spatial dimension
    #         mode="bilinear", align_corners=False
    #     ).squeeze(0).permute(1, 2, 0)  # Back to (10, 24, 768)
    #     weight = torch.view_as_complex(weight.contiguous())

    #     print(f"x.shape: {x.shape}, weight.shape: {weight.shape}")  # Debugging line
    #     x = x * weight
    #     x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
    #     x = x * self.scale + self.shift
    #     x = x + res
    #     return x.reshape(B, N, C)

@dataclass
class LoRA_config:
    RANK: int
    ALPHA: int = 1

class ASTModel_LoRA(ASTModel):
    def __init__(self, config, lora_config: LoRA_config):
        super().__init__(config)
        self.lora_config = lora_config
        self.encoder = ASTEncoder_LoRA(config, lora_config)
    
class ASTEncoder_LoRA(ASTEncoder):
    def __init__(self, config, lora_config):
        super().__init__(config)
        self.layer = nn.ModuleList([ASTLayer_LoRA(config, lora_config, i) for i in range(config.num_hidden_layers)])

class ASTLayer_LoRA(ASTLayer):
    def __init__(self, config, lora_config, layer_index):
        super().__init__(config)
        self.attention = ASTAttention_LoRA(config, lora_config)
        self.freqfit = FreqFit(config.hidden_size)  # Apply Frequency-Based adaptation
        self.layer_index = layer_index
    
    def forward(self, hidden_states, *args, **kwargs):
        hidden_states = self.attention(hidden_states, *args, **kwargs)[0]
        hidden_states = self.freqfit(hidden_states)  # Apply FreqFit after attention
        return (hidden_states,)

class ASTAttention_LoRA(ASTAttention):
    def __init__(self, config, lora_config):
        super().__init__(config)
        self.attention = ASTSelfAttention_LoRA(config, lora_config)

class ASTSelfAttention_LoRA(ASTSelfAttention):
    def __init__(self, config, lora_config):
        super().__init__(config)
        self.rank = lora_config.RANK
        self.scaling = lora_config.ALPHA/self.rank
        hid_size = config.hidden_size
        self.lora_down_q = nn.Linear(hid_size, round(hid_size/self.rank), bias=False)
        self.lora_up_q = nn.Linear(round(hid_size/self.rank), hid_size, bias=False)
        self.lora_down_v = nn.Linear(hid_size, round(hid_size/self.rank), bias=False)
        self.lora_up_v = nn.Linear(round(hid_size/self.rank), hid_size, bias=False)
        nn.init.zeros_(self.lora_down_q.weight)
        nn.init.kaiming_uniform_(self.lora_up_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_down_v.weight)
        nn.init.kaiming_uniform_(self.lora_up_v.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False):
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        Q_lora = self.lora_up_q(self.lora_down_q(hidden_states))
        V_lora = self.lora_up_v(self.lora_down_v(hidden_states))
        query_layer = query_layer + Q_lora * self.scaling
        value_layer = value_layer + V_lora * self.scaling
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs



class AST_LoRA_freq(nn.Module):
    def __init__(self, max_length: int, num_classes: int, final_output: str, rank: int, alpha: int, model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()
        
        self.lora_config = LoRA_config(rank, alpha)
        self.model = ASTModel_LoRA.from_pretrained(model_ckpt, self.lora_config, max_length=max_length, ignore_mismatched_sizes=True)
        self.model_config = self.model.config
        self.final_output = final_output
        
        assert final_output in ['CLS','ALL'], ("Classification can be only applied to the [CLS] token or to the entire sequence!")
        
        self.embeddings = self.model.embeddings
        self.encoder = self.model.encoder
        self.layernorm = self.model.layernorm
        
        self.classification_head = nn.Linear(self.model_config.hidden_size, num_classes)
        
        self.embeddings.requires_grad_(False)  
        self.encoder.requires_grad_(False)
        
        self._unfreeze_loras()
        
    def _unfreeze_loras(self):
        for block_idx in range(self.model_config.num_hidden_layers):
            self.encoder.layer[block_idx].attention.attention.lora_down_q.requires_grad_(True)
            self.encoder.layer[block_idx].attention.attention.lora_up_q.requires_grad_(True)
            self.encoder.layer[block_idx].attention.attention.lora_down_v.requires_grad_(True)
            self.encoder.layer[block_idx].attention.attention.lora_up_v.requires_grad_(True)
            
            # Optional: finetune also the LayerNorm before the MHSA layer.
            #self.encoder.layer[block_idx].layernorm_before.requires_grad_(True)
            
    def train(self, mode=True):
        if mode:
            self.encoder.eval()
            self.embeddings.eval()
            
            # Just in case the LayerNorm before MHSA is finetuned.
            #for block_idx in range(self.model_config.num_hidden_layers):
            #    self.encoder.layer[block_idx].layernorm_before.train()
            
            self.layernorm.train() 
            self.classification_head.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)
    
    def forward(self, x):
        x = self.embeddings(x)
        hidden_states = self.encoder(x)[0]
        hidden_states = self.layernorm(hidden_states)

        if self.final_output == 'CLS':
            return self.classification_head(hidden_states[:,0])
        else:
            return self.classification_head(hidden_states.mean(dim=1))
