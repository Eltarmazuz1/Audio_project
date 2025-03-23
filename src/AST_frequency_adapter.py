# import math
# import torch
# import torch.fft
# import torch.nn as nn

# import torch 
# import torch.nn as nn
# from transformers import ASTModel
# from dataclasses import dataclass
# from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTLayer, ASTEncoder, ASTOutput
# from typing import Optional, Tuple, Union

# class FrequencyBasedAdapter(nn.Module):
#     def __init__(self, blocks, dim, h=14, w=8):
#         super().__init__()
#         self.filter_weight = nn.Parameter(torch.randn(blocks, h, w, dim, dtype=torch.float32) * 0.02)
#         self.h = h
#         self.w = w
#         self.ssf_scale, self.ssf_shift = self.init_ssf_scale_shift(blocks, dim)

#     def init_ssf_scale_shift(self, blocks, dim):
#         scale = nn.Parameter(torch.ones(blocks, dim))
#         shift = nn.Parameter(torch.zeros(blocks, dim))
#         nn.init.normal_(scale, mean=1, std=.02)
#         nn.init.normal_(shift, std=.02)
#         return scale, shift

#     def ssf_ada(self, x, scale, shift):
#         assert scale.shape == shift.shape
#         if x.shape[-1] == scale.shape[0]:
#             return x * scale + shift
#         elif x.shape[1] == scale.shape[0]:
#             return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
#         else:
#             raise ValueError('Input tensor shape does not match the scale factor shape.')

#     def forward(self, block, x, spatial_size=None):
#         B, N, C = x.shape
#         if spatial_size is None:
#             a = b = int(math.sqrt(N))
#         else:
#             a, b = spatial_size

#         x = x.view(B, a, b, C).to(torch.float32)
#         res = x
#         x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
#         x = x * self.filter_weight[block].squeeze()
#         x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
#         x = self.ssf_ada(x, self.ssf_scale[block], self.ssf_shift[block])
#         x = x + res
#         x = x.reshape(B, N, C)
#         return x


# class ASTLayer_adapter(ASTLayer):
#     def __init__(self, config, adapter_config):
#         super().__init__(config)
        
#         self.adapter_config = adapter_config
#         self.config = config
        
#         if adapter_config.ADAPTER_CONF == 'sequential':
#             self.output = ASTOutput_adapter(config)
        
#         self.adapter_module_FFN = self.make_adapter(config.hidden_size, adapter_config)
        
#         if adapter_config.ADAPTER_TYPE == 'Houlsby':
#             self.adapter_module_MHSA = self.make_adapter(config.hidden_size, adapter_config)
    
#     def make_adapter(self, hidden_size, adapter_config):
#         return FrequencyBasedAdapter(adapter_config.REDUCTION_RATE, hidden_size, h=14, w=8)

# class ASTEncoder_adapter(ASTEncoder):
#     def __init__(self, config, adapter_config):
#         super().__init__(config)
        
#         self.layer = nn.ModuleList([ASTLayer_adapter(config, adapter_config) for _ in range(config.num_hidden_layers)])
        
# @dataclass
# class Adapter_config:
#     REDUCTION_RATE: int 
#     ADAPTER_TYPE: str
#     ADAPTER_CONF: str
#     APPLY_RESIDUAL: bool
#     ADAPTER_BLOCK: str
#     KERNEL_SIZE: int # the kernel size for the conformer. 

# class ASTModel_adapter(ASTModel):
#     def __init__(self, config, adapter_config: Adapter_config):
#         super().__init__(config)
        
#         self.adapter_config= adapter_config
        
#         self.encoder = ASTEncoder_adapter(config, adapter_config)


# class AST_adapter_freq(nn.Module):
#     def __init__(self, max_length: int, num_classes: int, final_output: str, reduction_rate: int, adapter_type: str, seq_or_par: str, apply_residual: bool, adapter_block: str, kernel_size: int, model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"):
#         ''' The reduction rate decides the bottleneck dimension of the adapter module --> bottleneck_dim = in_dim/reduction_rate.
#             The adapter_type param specifies the type of the adapter. Supported types: "Houlsby" and "Pfeiffer".
#             LN_train: whether the LN layers are trained along with the adapters. Original papers train the LNs.
#         '''
        
#         super().__init__()
        
#         self.adapter_config = Adapter_config(reduction_rate, adapter_type, seq_or_par, apply_residual, adapter_block, kernel_size)
#         self.model = ASTModel_adapter.from_pretrained(model_ckpt, self.adapter_config, max_length=max_length, ignore_mismatched_sizes=True)
#         self.model_config = self.model.config
#         self.final_output = final_output
        
#         assert final_output in ['CLS','ALL'], ("Classification can be only applied to the [CLS] token or to the entire sequence!")
#         assert adapter_type in ['Pfeiffer','Houlsby'], ('Only Pfeiffer and Houlsby adapter is supported for AST!')
        
        
#         self.embeddings = self.model.embeddings
#         self.encoder = self.model.encoder
#         self.layernorm = self.model.layernorm
        
#         self.classification_head = nn.Linear(self.model_config.hidden_size, num_classes)
        
#         self.embeddings.requires_grad_(False)  
#         self.encoder.requires_grad_(False)
        
#         self._unfreeze_adapters()
        
#     def _unfreeze_adapters(self):
#         for block_idx in range(self.model_config.num_hidden_layers):
#             self.encoder.layer[block_idx].adapter_module_FFN.requires_grad_(True)
#             self.encoder.layer[block_idx].layernorm_after.requires_grad_(True)
#             if self.adapter_config.ADAPTER_TYPE == 'Houlsby':
#                 self.encoder.layer[block_idx].adapter_module_MHSA.requires_grad_(True)
#                 self.encoder.layer[block_idx].layernorm_before.requires_grad_(True)
    
#     def train(self, mode=True):
        
#         if mode:
#             self.encoder.eval()
#             self.embeddings.eval()
#             for block_idx in range(self.model_config.num_hidden_layers):
                
#                 if self.adapter_config.ADAPTER_BLOCK =='conformer':
#                     self.encoder.layer[block_idx].adapter_module_FFN.bnorm.train()
#                     self.encoder.layer[block_idx].adapter_module_FFN.lnorm.train()
            
#                 if self.adapter_config.ADAPTER_TYPE == 'Houlsby':
#                     self.encoder.layer[block_idx].layernorm_before.train()
#                     if self.adapter_config.ADAPTER_BLOCK =='conformer':
#                         self.encoder.layer[block_idx].adapter_module_MHSA.bnorm.train()
#                         self.encoder.layer[block_idx].adapter_module_MHSA.lnorm.train()
                        
#                 self.encoder.layer[block_idx].layernorm_after.train()
            
#             self.layernorm.train() 
#             self.classification_head.train()
#         else:
#             # eval:
#             for module in self.children():
#                 module.train(mode)
    
#     def forward(self, x):
#         x = self.embeddings(x)
#         hidden_states = self.encoder(x)[0]
#         hidden_states = self.layernorm(hidden_states)

#         if self.final_output == 'CLS':
#             return self.classification_head(hidden_states[:,0])
#         else:
#             return self.classification_head(hidden_states.mean(dim=1))

#     def forward_tsne(self,x):
#         x = self.embeddings(x)
#         hidden_states = self.encoder(x)[0]
#         hidden_states = self.layernorm(hidden_states)
        
#         return hidden_states[:,0], hidden_states.mean(dim=1)
        