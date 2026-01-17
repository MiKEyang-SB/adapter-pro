"""
action_heads.py

Implementations of various action heads, which serve as alternatives to VLM sequential token prediction.
"""

import math
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX, NUM_TOKENS
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D
from torch.nn import ModuleList
from .transformer_modules import AdaLNModulation, SelfAttnLayer, SpaTempAttnLayer, FFN, modulate
import copy
from .tools import *
def learnable_random_perturbations(seq_len, dim, device, dtype):
    random_perturbations = nn.Parameter(torch.zeros(seq_len, dim, device=device, dtype=dtype))
    nn.init.normal_(random_perturbations, mean=0.0, std=0.02)
    return random_perturbations



class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096, #vla.module.llm_dim
        hidden_dim=4096, #vla.module.llm_dim
        action_dim=7,
        num_task_tokens=512,
        use_pro_version=False,
    ):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.model = MLPResNet(
            num_blocks=24, 
            input_dim=input_dim*ACTION_DIM, 
            hidden_dim=hidden_dim, 
            output_dim=action_dim,
            use_pro_version=use_pro_version
            )

    def predict_action(
            self, 
            actions_hidden_states, #(b, 25, 576, 896) (b, 层, 512图+64动作, 896)
            proprio=None, #(b, 8)
            proprio_projector=None,
            phase="Inference"
            ):
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
        proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
        proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)

        task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :] #(b, 25, 512, 896)
        actions_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :] #(b, 25, 64, 896)

        cond_actions_hidden_states = torch.zeros(
            (batch_size, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim),
            device=device, dtype=actions_hidden_states.dtype
        ).detach()  #(B, action_dim * NUM_ACTIONS_CHUNK, hidden_dim)

        rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(
            batch_size, NUM_ACTIONS_CHUNK, -1
        )  # (batch, chunk_len, action_dim * hidden_dim)

        if phase == "Training":
            batch_size, seq_len, dim = rearranged_actions_hidden_states.shape
            random_perturbations = learnable_random_perturbations(seq_len, dim, device=rearranged_actions_hidden_states.device, dtype=rearranged_actions_hidden_states.dtype) 
            rearranged_actions_hidden_states = (rearranged_actions_hidden_states + random_perturbations) # (1, seq_len, dim)

        action = self.model(
            rearranged_actions_hidden_states, # (batch, chunk_len, action_dim * hidden_dim) (b, 7, 8*896)
            h_a=actions_hidden_states, #transformer中和的动作hidden_state #(b, 25, 64, 896)
            p=proprio_features,  # (bsz, 1, llm_dim)
            h_t=task_hidden_states #视觉语言 torch.Size([b, 25, 512, 896])
            )

        return action #(batch, chunk_len, action_dim)
    

class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(
            self, 
            num_blocks, 
            input_dim, 
            hidden_dim, 
            output_dim,
            use_pro_version=False
            ):
        
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()

        for _ in range(num_blocks):
            if use_pro_version:
                self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
            else:
                self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
                
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x, h_a=None, h_t=None, p= None):
 
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for i, block in enumerate(self.mlp_resnet_blocks):
            x = block(x, h_t = h_t[:,i+1,:], h_a = h_a[:,i+1,:], p=p)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x   

class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # [bs, ntokens, input_feats]
        x = x.permute((1, 0, 2)) # [seqen, bs, input_feats]
        # print(x.shape)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x
class SpaTempPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(SpaTempPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.positional_encoding = PositionalEncoding2D(d_model)
    
    def forward(self, x, window_size=10):
        seqlen, bs, input_feats = x.shape
        t = window_size//5
        # x1, sep, x2  = x.split([seqlen//2, 1, seqlen//2])
        #这里的时空注意力出问题了，到底该怎么办啊
        def add_positional_encoding(x):
            x = x.permute(1,0,2) # [seqen, bs, input_feats] -> [bs, seqen, input_feats]
            x = x.reshape(x.shape[0], t, x.shape[1]//t, x.shape[2])

            x = x + self.positional_encoding(x)

            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
            x = x.permute(1,0,2)

            return x

        x = add_positional_encoding(x)
        # x2 = add_positional_encoding(x2)
        
        # x = torch.cat([x1, sep, x2], dim=0)
        return self.dropout(x)
class OutputProcess_adaLN(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        
        self.LayerNorm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_mod = AdaLNModulation(latent_dim, nchunks=2)
        
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor, cond:torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        
        shift, scale = self.adaLN_mod(cond)
        hidden_states = modulate(self.LayerNorm(hidden_states), shift, scale)

        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, e, seqlen]
        return output
    
class VLATransformerAdaLNBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, nbp):
        super().__init__()
        self.spa_dim = nbp #时间步
        self.adaLN_mod_combined = AdaLNModulation(d_model, 6)
        self.self_attn = SelfAttnLayer(d_model, nhead, dropout) #self-attention
        self.ffn_combined = FFN(d_model, dim_feedforward, dropout) #feed-forward
        self.adaLN_mod_split = AdaLNModulation(d_model, 9)
        self.spa_temp_attn = SpaTempAttnLayer(d_model, nhead, dropout, spa_dim=self.spa_dim) #spatio-temporal-attn
        # self.local_inter_attn = LocalInteractionAttnLayer(d_model, nhead, dropout, spa_dim=self.spa_dim) #cross-attn
        self.ffn_spa = FFN(d_model, dim_feedforward, dropout) #feed-forward
    
    def forward(self, src, cond, src_key_padding_mask=None):
        N, B, d = src.shape
        # AdaLN modulation
        shift_self, scale_self, gate_self, \
            shift_ffn_c, scale_ffn_c, gate_ffn_c = self.adaLN_mod_combined(cond) #adal-n all (b, latent_dim)
        
        # Self-Attention
        src = self.self_attn(src, 
                            shift_self, scale_self, gate_self,
                            src_key_padding_mask=src_key_padding_mask)
        
        # FFN
        src = self.ffn_combined(src, shift_ffn_c, scale_ffn_c, gate_ffn_c)

        # AdaLN modulation
        shift_spa, scale_spa, gate_spa, shift_temp, scale_temp, gate_temp, \
                shift_ffn_s, scale_ffn_s, gate_ffn_s = self.adaLN_mod_split(cond) #(b, latent_dim)
        # shift_cross, scale_cross, shift_cross2, scale_cross2, gate_cross, \
        src_spa_temp = self.spa_temp_attn(src, 
                            shift_spa, scale_spa, gate_spa, 
                            shift_temp, scale_temp, gate_temp, 
                            src_key_padding_mask=src_key_padding_mask)
        src_after = self.ffn_spa(src_spa_temp, shift_ffn_s, scale_ffn_s, gate_ffn_s)
        return src_after

class VLATransformer(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout, nbp):
        super().__init__()
        block = VLATransformerAdaLNBlock(d_model=d_model,
                                nhead=nhead,
                                dim_feedforward=dim_feedforward,
                                dropout=dropout,
                                nbp=nbp)
        #方案a,使用adaln把条件信息注入
        #方案b,使用cross-attention注入条件信息
        module_list = []
        
        for _ in range(num_layers):
            module_list.append(copy.deepcopy(block))

        self.blocks =  ModuleList(module_list)

    def forward(self, src, cond, src_key_padding_mask=None):
        """
        Args:
            src: Tensor, shape (751, B, d_model)
            src_key_padding_mask: Tensor, shape (B, 751) 

        Additional token is for the condition
        """

        for block in self.blocks:
            src = block(src, cond, src_key_padding_mask=src_key_padding_mask)
        return src

class MaskTransformer(nn.Module):
    def __init__(self, 
                 mask_type = '1D',
                 code_dim = 512,
                 latent_dim = 896,
                 num_decoder_layers = 24,
                #  num_tokens = 256, #discret num
                 num_heads = 8,
                 dropout = 0.1,
                 bins = 256,
                 ):
        super().__init__()
        self.mask_type = mask_type
        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.noise_schedule = cosine_schedule
        self.noise_schedule_backward = cosine_schedule_backward
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position_enc = SpaTempPositionalEncoding(self.latent_dim, self.dropout)
        self.Transformer = VLATransformer(d_model=self.latent_dim,
                                            nhead=num_heads,
                                            dim_feedforward=4 * latent_dim,
                                            dropout=dropout,
                                            num_layers=num_decoder_layers,
                                            nbp=self.nbp)
        _num_tokens = bins + 1 #看一下这里
        self.mask_id = bins
        self.output_process = OutputProcess_adaLN(out_feats=_num_tokens, latent_dim=latent_dim)
        self.token_emb = nn.Embedding(_num_tokens, self.code_dim)

    def mask_cond(self, cond, force_mask=False):#条件掩码
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:#1
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond
    def trans_forward(self, x_ids, cond, force_mask=False):
        cond = self.mask_cond(cond, force_mask=force_mask)
        x = self.token_emb(x_ids) 
        #这里看一下位置编码的细节
        x = self.input_process(x) #(8, b, 896)
        # x = self.position_enc(x)
        output = self.Transformer(x, cond)#特定层,每一层都有位置编码
        #明天把维度都对对齐
        #解决离散动作没有0的问题
        #解决位置编码的问题
        #初步完成训练
        logits = self.output_process(output, cond) #final
        return logits

    def forward(self, discretized_action, actions_hidden_states, task_hidden_states): 
        '''
        x: 
        注意这里的x是原始tokenizer之后的action_id
        discretized_action: (b, w, 7) all by discretized [1, 256]
        task_description:(b, 512, 896)
        action_query: (b, 64, 896)
        '''
        force_mask = False
        bs, xtokens, ytokens = discretized_action.shape
        ntokens = xtokens * ytokens
        if self.mask_type == '1D':
            rand_time = uniform((bs,), device=self.device)#(bs,)
            rand_mask_probs = self.noise_schedule(rand_time)
            num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)
            batch_randperm = torch.rand((bs, ntokens), device=self.device).argsort(dim=-1)
            mask = batch_randperm < num_token_masked.unsqueeze(-1) #(bs, ntokens)
            labels = torch.where(mask, discretized_action, self.mask_id)
            x_ids = discretized_action.clone()
            mask_rid = get_mask_subset_prob(mask, 0.1)
            rand_id = torch.randint_like(x_ids, high=self.mask_id)
            x_ids = torch.where(mask_rid, rand_id, x_ids)#10%替换为随机的值
            mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
            x_ids = torch.where(mask_mid, self.mask_id, x_ids)#(b, x*y)
        elif self.mask_type == '2D':
            rand_time = uniform((bs,), device=self.device)
            rand_mask_probs = self.noise_schedule(rand_time)

            # ========== temporal mask
            #因为太少了，min就设置为0吧
            num_token_masked = (xtokens * rand_mask_probs).round().clamp(min=1)
            batch_randperm = torch.rand((bs, xtokens), device=self.device).argsort(dim=-1)
            # Positions to be MASKED are ALL TRUE
            mask = batch_randperm < num_token_masked.unsqueeze(-1)
            # Positions to be MASKED must also be NON-PADDED
            # mask = mask & non_pad_mask[..., 0]
            # Note this is our training target, not input
            labels = torch.where(mask[..., None].repeat(1, 1, ytokens), x, self.mask_id)
            x_ids = discretized_action.clone()
            # Further Apply Bert Masking Scheme
            # Step 1: 10% replace with an incorrect token
            mask_rid = get_mask_subset_prob(mask, 0.1)
            rand_id = torch.randint_like(x_ids, high=self.mask_id)
            x_ids = torch.where(mask_rid[..., None].repeat(1, 1, ytokens), rand_id, x_ids)
            # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
            mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
            x_ids = torch.where(mask_mid[..., None].repeat(1, 1, ytokens), self.mask_id, x_ids)
            mask_time = mask
            mask_time = mask_time[..., None].repeat(1, 1, ytokens)       # keep temperal mask still masked
            # print((x_ids==512).sum(), mask_time.sum(), mask.sum(), (labels!=512).sum(), mask_rid.sum())

            # ========== spatial mask
            num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=0)
            batch_randperm = torch.rand((bs, ntokens), device=self.device).argsort(dim=-1)
            # Positions to be MASKED are ALL TRUE
            mask = batch_randperm < num_token_masked.unsqueeze(-1)
            # Positions to be MASKED must also be NON-PADDED
            # mask = mask & non_pad_mask.reshape(bs, -1)
            mask = mask & ~mask_time.reshape(bs, -1)
            # Note this is our training target, not input
            labels = torch.where(mask, x_ids.reshape(bs, -1), labels.reshape(bs, -1))
            x_ids = x_ids.reshape(bs, -1)
            # Further Apply Bert Masking Scheme
            # Step 1: 10% replace with an incorrect token
            mask_rid = get_mask_subset_prob(mask, 0.1)

            rand_id = torch.randint_like(x_ids, high=self.mask_id)
            x_ids = torch.where(mask_rid, rand_id, x_ids)
            # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
            mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
            # mask_mid = mask
            x_ids = torch.where(mask_mid, self.mask_id, x_ids)#(b, x*y)
        cond = torch.cat([actions_hidden_states, task_hidden_states], dim=2) #(b, 25, 64+512, 896)
        logits = self.trans_forward(x_ids, cond, force_mask)


def apply_rope(q, k, cos, sin):
    """
    RoPE:
    q, k: (B, H, T, D)   # D must be an even number
    cos/sin: (T, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)


    def rotate_half(x):
        # Swap even and odd dimensions and flip the signs
        x1 = x[..., ::2]   # Even subdimension
        x2 = x[..., 1::2]  # odd subdimension

        return torch.stack((-x2, x1), dim=-1).reshape_as(x)


    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot



class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        """
        dim = head_dim
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be an even number"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (T, dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)



class MLPResNetBlock(nn.Module):
    """
    One residual MLP block with cross-attention conditioning.

    This block applies multi-head attention over:
      - token features (self-attention),
      - task-related hidden states (h_t),
      - action/proprioception-related hidden states (h_a, p).
    The outputs are combined via a gating mechanism, projected back to the
    hidden dimension, and passed through a small feedforward sub-network with
    residual connection.

    Args:
        dim (int): Dimensionality of the hidden features. Must be divisible by num_heads.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        h_t (torch.Tensor, optional): Task-related hidden states of shape
                                      (batch_size, K, hidden_dim).
        h_a (torch.Tensor, optional): Action-related hidden states of shape
                                      (batch_size, 1, hidden_dim).
        p (torch.Tensor, optional): Additional conditioning features
                                    (e.g., proprioception), shape (batch_size, 1, hidden_dim).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Main feedforward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.num_heads = 8
        self.head_dim = dim // self.num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.gating_factor = nn.Parameter(torch.zeros(1))



    def forward(self, x, h_t=None, h_a=None, p=None):
        """
        x: (batch_size, seq_len, hidden_dim)
        h, t, p: (batch_size, 1, hidden_dim) or None
        """

        g = self.gating_factor
        ratio_g = nn.Tanh()(g)

        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)

        h = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)

        B = x.size(0)
        T = x.size(1)
        C = x.size(2)
        K_t = h.size(1)
        K = h_t.size(1)

        task_k = h
        task_v = h

        adapter_k = h_t
        adapter_v = h_t

        q_1 = self.q_proj(x) # (B, T, C)
        k_tokens = self.k_proj(x)             # (B, T, C)
        v_tokens = self.v_proj(x)             # (B, T, C)
        k_task = self.k_proj(task_k)    # (B, K, C)
        v_task = self.v_proj(task_v)    # (B, K, C)

        k_adapter = self.k_proj(adapter_k)    # (B, K, C)
        v_adapter = self.v_proj(adapter_v)    # (B, K, C)

        # (B, seq_len, C) -> (B, num_heads, seq_len, head_dim)
        q_1 = q_1.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_tokens = k_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_tokens = v_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_task = k_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)
        v_task = v_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)

        k_adapter = k_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v_adapter = v_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores_tokens = torch.matmul(q_1, k_tokens.transpose(-2, -1)) # (B, H, T, T)
        attn_scores_task = torch.matmul(q_1, k_task.transpose(-2, -1)) * 1 # (B, H, T, K)
        attn_scores_adapter = torch.matmul(q_1, k_adapter.transpose(-2, -1)) * ratio_g # (B, H, T, K)

        attn_scores = torch.cat([attn_scores_tokens, attn_scores_task, attn_scores_adapter], dim=-1) # (B, H, T, T+K)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1) # (B, H, T, T+K)

        v_combined = torch.cat([v_tokens, v_task, v_adapter], dim=2) # (B, H, T+K, head_dim)
        output = torch.matmul(attn_weights, v_combined) # (B, H, T, head_dim)

        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        x = self.ffn(output + x) 

        return x



class MLPResNetBlock_Pro(nn.Module):
    """One MLP ResNet block with separate projections for self, adapter, task + RoPE, now with FiLM modulation."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            )

        # Q (from x only)
        self.q_proj = nn.Linear(dim, dim)

        # Self-Attention: K, V
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)

        # Adapter cross-attention: K, V
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)

        # Task cross-attention: K, V
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)

        # gating
        self.gating_factor = nn.Parameter(torch.zeros(1))

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

        # ---- FiLM ----
        # FiLM is useless; to avoid conflict with chkpt, it can be kept as is for now.
        self.film_gen = nn.Sequential(
            nn.Linear(dim, dim * 2),  # output γ and β
            )


    def apply_film(self, x, gamma, beta):
        """FiLM: per-channel modulation"""
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


    def forward(self, x, h_a=None, h_t=None, p=None):
        """
        x: (B, 8, 896)
        h_a: adapter tokens #(b, 64, 896)
        h_t: task tokens #(b, 512, 896)
        p:   possible conditioning vector (for FiLM) #(b, 1, )
        """
        g = self.gating_factor
        ratio_g = torch.tanh(g)

        # concat h_a and p
        h_adapter = torch.cat((h_a, p),dim=1) #torch.Size([4, 65, 896])

        h_task = h_t #torch.Size([4, 512, 896])
        B, T, C = x.shape
        K_a = h_adapter.size(1) if h_a is not None else 0
        K_t = h_task.size(1) if h_task is not None else 0

        # Q
        q_1 = self.q_proj(x)

        # self tokens
        k_tokens = self.k_self(x)
        v_tokens = self.v_self(x)

        # adapter tokens
        k_adapter = self.k_adapter(h_adapter)
        v_adapter = self.v_adapter(h_adapter)

        # task tokens
        k_task = self.k_task(h_task)
        v_task = self.v_task(h_task)


        # reshape -> multi-head
        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)


        q_1 = reshape_heads(q_1, B, T)
        k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T)
        k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a)
        k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)

        # RoPE
        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main)
        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)     
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

        # attention scores
        attn_scores = [torch.matmul(q_1, k_tokens.transpose(-2, -1))]
        attn_scores.append(torch.matmul(q_1, k_adapter.transpose(-2, -1)))
        attn_scores.append(torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g)
        attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)#(B, H, T, T + K_a + K_t)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # combine V
        v_list = [v_tokens,v_adapter,v_task]
        v_combined = torch.cat(v_list, dim=2)

        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        # # ---- FiLM ---- 
        # gamma_beta = self.film_gen(p)  # [B, 2C]
        # gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, C], [B, C]
        # output = self.apply_film(output, gamma, beta)

        # residual + FFN
        x = self.ffn(output + x)
        return x


# ============================================================================
# Discrete Diffusion Action Head with MLPResNetBlock_Pro Architecture
# 融合 VLA-Adapter 的多层注入结构 和 DiscreteDiffusionVLA 的 mask-remask 机制
# ============================================================================



# ============ Mask Schedule (from DiscreteDiffusionVLA) ============
def cosine_schedule(ratio, unknown_init):
    """
    Cosine mask scheduling: ratio [0, 1] -> mask_ratio
    Args:
        ratio: (scalar or tensor) current progress [0, 1]
        unknown_init: (B,) initial number of masked tokens per sample
    Returns:
        mask_ratio: (B,) ratio of tokens to keep masked
    """
    return torch.cos(ratio * torch.pi / 2)


# ============ Mask by Random Top-K (from DiscreteDiffusionVLA) ============
def mask_by_random_topk(probs, mask_len, temperature=1.0):
    """
    Args:
        probs: (B, L) confidence scores for each position
        mask_len: (B,) number of tokens to mask per sample
        temperature: float, Gumbel noise temperature
    Returns:
        mask: (B, L) boolean tensor, True = keep masked
    """
    # Gumbel noise
    gumbel = -torch.log(-torch.log(torch.rand_like(probs) + 1e-20) + 1e-20)
    confidence = torch.log(probs + 1e-20) + temperature * gumbel  # (B, L)

    # Find k-th smallest threshold
    sorted_conf, _ = confidence.sort(dim=1)
    B, L = probs.shape
    k = mask_len.clamp(min=1, max=L-1)
    batch_idx = torch.arange(B, device=probs.device)
    threshold = sorted_conf[batch_idx, k]

    # Positions below threshold are masked
    return confidence < threshold.unsqueeze(1)


# ============ Main Action Head ============
class DiscreteDiffusionActionHead(nn.Module):
    """
    Discrete Diffusion Action Head with MLPResNetBlock_Pro architecture

    Key features:
    - Multi-layer injection from VLA backbone hidden states
    - MaskGIT-style iterative decoding
    - Outputs discrete token logits (not continuous actions)
    - Uses MLPResNetBlock_Pro with RoPE and independent projections
    """
    def __init__(
        self,
        hidden_dim=896,               # VLA hidden state dimension
        action_head_dim=896,          # Internal dimension for action decoder
        num_blocks=24,                # Number of MLPResNetBlock_Pro layers
        num_heads=8,                  # Multi-head attention heads
        num_action_tokens=56,         # NUM_ACTIONS_CHUNK * ACTION_DIM (e.g., 8*7=56)
        vocab_size=256,               # Discrete action vocabulary size (256 bins)
        mask_token_id=256,            # Special token ID for <mask>
        num_diffusion_iters=12,       # Number of iterative decoding steps
        use_proprio=False,            # Whether to use proprioception
        proprio_dim=8,                # Proprioception dimension
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_head_dim = action_head_dim
        self.num_blocks = num_blocks
        self.num_action_tokens = num_action_tokens
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.num_diffusion_iters = num_diffusion_iters
        self.use_proprio = use_proprio

        # ========== Input Projection ==========
        # Learnable action query embeddings
        self.action_query_embed = nn.Parameter(
            torch.randn(1, num_action_tokens, action_head_dim)
        )

        # Proprio projector (if needed)
        if use_proprio:
            self.proprio_projector = nn.Linear(proprio_dim, action_head_dim)

        # Token embedding layer (for masked tokens during iterative decoding)
        self.token_embedding = nn.Embedding(vocab_size + 1, action_head_dim)  # +1 for mask token

        # ========== 24 MLPResNetBlock_Pro layers ==========
        self.blocks = nn.ModuleList([
            MLPResNetBlock_Pro(dim=action_head_dim, num_heads=num_heads)
            for _ in range(num_blocks)
        ])

        # ========== Output head: predict discrete token logits ==========
        self.output_head = nn.Sequential(
            nn.LayerNorm(action_head_dim),
            nn.Linear(action_head_dim, vocab_size)
        )

    def forward(
        self,
        multi_layer_hidden_states,  # (B, num_layers, num_tokens, hidden_dim)
        proprio=None,                # (B, proprio_dim) or None
        input_tokens=None,           # (B, num_action_tokens) optional input tokens for conditioning
    ):
        """
        Forward pass through the action decoder

        Args:
            multi_layer_hidden_states: (B, L_layers, num_tokens, hidden_dim)
                - L_layers: number of VLA backbone layers (e.g., 25)
                - num_tokens: task tokens (visual patches) + action tokens
            proprio: (B, proprio_dim) proprioception features
            input_tokens: (B, num_action_tokens) optional discrete token IDs for conditioning

        Returns:
            logits: (B, num_action_tokens, vocab_size) discrete token logits
        """
        B = multi_layer_hidden_states.size(0)
        device = multi_layer_hidden_states.device

        # ========== Extract task and adapter features ==========
        # Assume multi_layer_hidden_states shape: (B, num_layers, num_total_tokens, hidden_dim)
        # We need to separate visual tokens and action tokens
        # For VLA-Adapter, this is typically done via masking in the training script

        # Here we assume the structure matches VLA-Adapter's extraction:
        # - First 512 tokens: visual features (h_t)
        # - Remaining tokens: action hidden states (h_a)
        num_visual_tokens = 512  # Adjust based on your vision encoder (e.g., 576 for OpenVLA)

        h_task = multi_layer_hidden_states[:, :, :num_visual_tokens, :]   # (B, L, 512, hidden_dim)
        h_adapter = multi_layer_hidden_states[:, :, num_visual_tokens:, :]  # (B, L, K, hidden_dim)

        # Process proprio
        if self.use_proprio and proprio is not None:
            p = self.proprio_projector(proprio).unsqueeze(1)  # (B, 1, action_head_dim)
        else:
            p = None

        # ========== Initialize action queries ==========
        if input_tokens is not None:
            # If input tokens provided, embed them
            x = self.token_embedding(input_tokens)  # (B, num_action_tokens, action_head_dim)
        else:
            # Use learnable query embeddings
            x = self.action_query_embed.expand(B, -1, -1)  # (B, num_action_tokens, action_head_dim)

        # ========== Pass through MLPResNetBlock_Pro layers ==========
        for i, block in enumerate(self.blocks):
            # Extract corresponding layer features
            h_t_i = h_task[:, i, :, :]      # (B, num_visual_tokens, hidden_dim)
            h_a_i = h_adapter[:, i, :, :]   # (B, K, hidden_dim)

            # Project to action_head_dim if needed
            if h_t_i.size(-1) != self.action_head_dim:
                # For dimension mismatch, we need projection layers
                # TODO: Add projection layers in __init__ if needed
                pass

            x = block(x, h_a=h_a_i, h_t=h_t_i, p=p)  # (B, num_action_tokens, action_head_dim)

        # ========== Output discrete token logits ==========
        logits = self.output_head(x)  # (B, num_action_tokens, vocab_size)

        return logits

    def predict_action(
        self,
        multi_layer_hidden_states,
        proprio=None,
        num_diffusion_iters=None,
        temperature=1.0,
        use_remask=False,
    ):
        """
        Inference: MaskGIT-style iterative decoding

        Args:
            multi_layer_hidden_states: (B, num_layers, num_tokens, hidden_dim)
            proprio: (B, proprio_dim)
            num_diffusion_iters: number of decoding iterations (default: self.num_diffusion_iters)
            temperature: sampling temperature
            use_remask: whether to allow remasking of previously decoded tokens

        Returns:
            final_actions: (B, num_action_tokens) discrete token IDs
        """
        if num_diffusion_iters is None:
            num_diffusion_iters = self.num_diffusion_iters

        B = multi_layer_hidden_states.size(0)
        device = multi_layer_hidden_states.device

        # Initialize with all masked tokens
        cur_seqs = torch.full(
            (B, self.num_action_tokens),
            self.mask_token_id,
            dtype=torch.long,
            device=device
        )

        unknown_init = (cur_seqs == self.mask_token_id).sum(dim=1)  # (B,) = num_action_tokens

        # ========== Iterative decoding ==========
        for step in range(num_diffusion_iters):
            # 1) Forward pass to get logits
            logits = self.forward(
                multi_layer_hidden_states,
                proprio=proprio,
                input_tokens=cur_seqs
            )  # (B, L, vocab_size)

            probs = torch.softmax(logits, dim=-1)  # (B, L, vocab_size)

            # 2) Sample from categorical distribution
            flat_probs = probs.view(-1, probs.size(-1))  # (B*L, vocab_size)
            sampled_flat = torch.multinomial(flat_probs, 1)  # (B*L, 1)
            sampled = sampled_flat.view(B, self.num_action_tokens)  # (B, L)

            # 3) Only update masked positions
            unknown_map = cur_seqs == self.mask_token_id
            sampled = torch.where(unknown_map, sampled, cur_seqs)

            # 4) Calculate mask ratio for next iteration
            ratio = float(step + 1) / num_diffusion_iters
            mask_ratio = cosine_schedule(torch.tensor(ratio, device=device), unknown_init)
            mask_len = torch.floor(unknown_init.float() * mask_ratio).long()
            mask_len = torch.clamp(mask_len, min=0, max=(unknown_init - 1).clamp(min=0))

            # Early stop if no more tokens to mask
            if mask_len.max() == 0:
                break

            # 5) Calculate confidence scores
            selected_probs = probs.gather(2, sampled.unsqueeze(-1)).squeeze(-1)  # (B, L)

            if use_remask:
                # Allow remasking with decreasing probability
                p_remask = 1.0 - ratio
                selected_probs = torch.where(
                    unknown_map,
                    selected_probs,
                    selected_probs * p_remask
                )
            else:
                # Already decoded tokens have infinite confidence
                selected_probs = torch.where(
                    unknown_map,
                    selected_probs,
                    torch.tensor(float('inf'), device=device)
                )

            # 6) Select tokens to remask using Gumbel top-k
            masking = mask_by_random_topk(
                selected_probs,
                mask_len,
                temperature=temperature * (1.0 - ratio)
            )

            # 7) Update sequence: remask low-confidence positions
            next_seqs = torch.where(masking, self.mask_token_id, sampled)
            cur_seqs = next_seqs

        # ========== Final sampling ==========
        logits = self.forward(
            multi_layer_hidden_states,
            proprio=proprio,
            input_tokens=cur_seqs
        )
        probs = torch.softmax(logits, dim=-1)
        flat_probs = probs.view(-1, probs.size(-1))
        final_tokens = torch.multinomial(flat_probs, 1).view(B, self.num_action_tokens)

        return final_tokens

    def compute_loss(
        self,
        multi_layer_hidden_states,
        target_actions,  # (B, num_action_tokens) discrete token IDs
        proprio=None,
        mask_ratio_range=(0.0, 1.0),  # Random masking ratio range (deprecated, use cosine schedule instead)
        no_mask_token_prob=0.0,        # Probability to unmask some masked tokens
        use_cosine_schedule=True,      # Use cosine schedule like DiscreteDiffusionVLA
    ):
        """
        Training loss: randomly mask some tokens and predict them

        Uses cosine schedule (like DiscreteDiffusionVLA) to determine mask ratio

        Args:
            multi_layer_hidden_states: (B, num_layers, num_tokens, hidden_dim)
            target_actions: (B, num_action_tokens) ground truth discrete token IDs
            proprio: (B, proprio_dim)
            mask_ratio_range: (deprecated) tuple (min_ratio, max_ratio) for random masking
            no_mask_token_prob: probability to unmask some already-masked tokens
            use_cosine_schedule: if True, use cosine schedule; if False, use random uniform

        Returns:
            loss: scalar cross-entropy loss
        """
        B = target_actions.size(0)
        device = target_actions.device

        # ========== Step 1: Calculate total maskable tokens ==========
        # All tokens are maskable in our case
        total_unknown = torch.full((B,), self.num_action_tokens, dtype=torch.float32, device=device)  # (B,)

        if use_cosine_schedule:
            # ========== Step 2: Sample random time ratio in [0, 1) ==========
            rand_time = torch.rand(B, device=device)  # (B,)

            # ========== Step 3: Use cosine schedule to compute mask ratio ==========
            # mask_ratios: (B,), values in (0, 1]
            mask_ratios = cosine_schedule(rand_time, total_unknown)  # (B,)

            # ========== Step 4: Compute number of tokens to mask per sample ==========
            num_mask = torch.clamp((total_unknown * mask_ratios).round(), min=1).long()  # (B,)
        else:
            # Use old random uniform masking
            mask_ratio = torch.rand(1, device=device) * (mask_ratio_range[1] - mask_ratio_range[0]) + mask_ratio_range[0]
            num_mask = torch.full((B,), int(self.num_action_tokens * mask_ratio.item()), dtype=torch.long, device=device)

        # ========== Step 5: Generate random scores for each position ==========
        vals = torch.rand(B, self.num_action_tokens, device=device)  # (B, num_action_tokens)

        # ========== Step 6: Sort and select top-k positions to mask ==========
        perm = vals.argsort(dim=1)                    # (B, num_action_tokens) - indices after sorting
        ranks = perm.argsort(dim=1)                   # (B, num_action_tokens) - rank of each position
        masked_mask = ranks < num_mask[:, None]       # (B, num_action_tokens) - True if masked

        # ========== Step 7: Optional: unmask some positions with no_mask_token_prob ==========
        # 再次随机取消一部分已 mask 的位置
        if no_mask_token_prob > 0:
            # Generate random probabilities
            prob = torch.rand(B, self.num_action_tokens, device=device)
            # Unmask positions where prob < no_mask_token_prob AND already masked
            unmask = (prob < no_mask_token_prob) & masked_mask
            masked_mask = masked_mask & (~unmask)

        # ========== Step 8: Create input tokens with masked positions ==========
        input_tokens = target_actions.clone()
        input_tokens[masked_mask] = self.mask_token_id

        # ========== Step 9: Get predictions ==========
        logits = self.forward(
            multi_layer_hidden_states,
            proprio=proprio,
            input_tokens=input_tokens
        )  # (B, num_action_tokens, vocab_size)

        # ========== Step 10: Compute loss only on masked positions ==========
        loss = torch.nn.functional.cross_entropy(
            logits[masked_mask].view(-1, self.vocab_size),
            target_actions[masked_mask].view(-1),
            reduction='mean'
        )

        return loss


# ============================================================================
# Usage Example: How to use ActionTokenizer with DiscreteDiffusionActionHead
# ============================================================================
"""
完整使用流程示例:

## 1. 初始化 Tokenizer 和 Action Head

```python
from prismatic.models.action_heads import ActionTokenizer, DiscreteDiffusionActionHead

# 初始化 action tokenizer
action_tokenizer = ActionTokenizer(
    bins=256,           # 256 个离散 bins
    min_action=-1.0,    # action 范围下界
    max_action=1.0      # action 范围上界
)

# 初始化 discrete diffusion action head
action_head = DiscreteDiffusionActionHead(
    hidden_dim=896,              # VLA hidden state 维度
    action_head_dim=896,         # action decoder 内部维度
    num_blocks=24,               # MLPResNetBlock_Pro 层数
    num_heads=8,                 # 注意力头数
    num_action_tokens=56,        # 8 chunks × 7 dims = 56 tokens
    vocab_size=256,              # 离散词汇表大小
    mask_token_id=255,           # mask token ID (通常是 vocab_size - 1)
    num_diffusion_iters=12,      # 推理时的迭代解码步数
    use_proprio=True,            # 是否使用 proprio
    proprio_dim=7,               # proprio 维度
)
```

## 2. 训练时的数据流

```python
# ========== Step 1: 获取 ground truth actions ==========
# 假设从数据集中获取连续 actions
gt_actions = batch["actions"]  # (B, NUM_CHUNKS, ACTION_DIM) = (4, 8, 7)
                                # 值范围: [-1, 1]

# ========== Step 2: Tokenize 连续 actions → 离散 tokens ==========
# 转换为 numpy 进行 tokenization
gt_actions_np = gt_actions.cpu().numpy()  # (4, 8, 7)

# Encode: (4, 8, 7) → (4, 56)
target_tokens_np = action_tokenizer.encode(gt_actions_np)
# target_tokens_np shape: (4, 56), 值范围: [0, 255]

# 转回 torch tensor
target_tokens = torch.from_numpy(target_tokens_np).to(device)  # (4, 56)

# ========== Step 3: 前向传播 + Loss 计算 ==========
# 获取 VLA backbone 的 multi-layer hidden states
output = vla_model(
    images=batch["images"],
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    output_hidden_states=True,  # 关键: 必须输出所有层的 hidden states
)

# 提取所有层的 hidden states
all_hidden_states = output.hidden_states  # tuple of (B, num_tokens, 896)
multi_layer_hidden_states = torch.stack(all_hidden_states, dim=1)  # (B, 25, num_tokens, 896)

# 计算 loss
loss = action_head.compute_loss(
    multi_layer_hidden_states=multi_layer_hidden_states,
    target_actions=target_tokens,  # (B, 56) 离散 token IDs
    proprio=batch["proprio"],      # (B, 7)
    mask_ratio_range=(0.0, 1.0),   # 随机 mask 0-100%
)

loss.backward()
optimizer.step()
```

## 3. 推理时的数据流

```python
# ========== Step 1: 获取 VLA hidden states ==========
with torch.no_grad():
    output = vla_model(
        images=observation_images,
        input_ids=text_input_ids,
        attention_mask=text_attention_mask,
        output_hidden_states=True,
    )

    all_hidden_states = output.hidden_states
    multi_layer_hidden_states = torch.stack(all_hidden_states, dim=1)  # (1, 25, num_tokens, 896)

# ========== Step 2: MaskGIT 迭代解码 ==========
predicted_tokens = action_head.predict_action(
    multi_layer_hidden_states=multi_layer_hidden_states,
    proprio=current_proprio,      # (1, 7)
    num_diffusion_iters=12,       # 12 步迭代解码
    temperature=1.0,              # 采样温度
    use_remask=False,             # 是否允许 remask
)
# predicted_tokens shape: (1, 56), 值范围: [0, 255]

# ========== Step 3: Detokenize 离散 tokens → 连续 actions ==========
# 转换为 numpy
predicted_tokens_np = predicted_tokens.cpu().numpy()  # (1, 56)

# Decode: (1, 56) → (1, 8, 7)
predicted_actions_np = action_tokenizer.decode(
    token_ids=predicted_tokens_np,
    action_dim=7  # ACTION_DIM
)
# predicted_actions_np shape: (1, 8, 7), 值范围: [-1, 1]

# 转回 torch tensor
predicted_actions = torch.from_numpy(predicted_actions_np).to(device)  # (1, 8, 7)

# ========== Step 4: 执行 actions ==========
# 通常只执行第一个 chunk
current_action = predicted_actions[0, 0, :]  # (7,)
robot.execute_action(current_action.cpu().numpy())
```

## 4. 关键点总结

### 4.1 Tokenization 流程
```
连续 actions (B, 8, 7)
    ↓ [action_tokenizer.encode()]
离散 tokens (B, 56) [值: 0-255]
    ↓ [action_head.forward()]
logits (B, 56, 256)
    ↓ [cross_entropy_loss]
loss (scalar)
```

### 4.2 Detokenization 流程 (推理)
```
初始: 全 mask tokens (B, 56) [全部为 255]
    ↓ [12 步 MaskGIT 迭代解码]
最终 tokens (B, 56) [值: 0-255]
    ↓ [action_tokenizer.decode()]
连续 actions (B, 8, 7) [值: -1 到 1]
```

### 4.3 与 VLA-Adapter 原版的区别

| 特性 | VLA-Adapter 原版 | DiscreteDiffusionActionHead |
|------|------------------|----------------------------|
| **输入** | 连续 actions (B, 8, 7) | 离散 tokens (B, 56) |
| **输出** | 连续 actions (B, 8, 7) | logits (B, 56, 256) |
| **Loss** | L1 Loss | Cross-Entropy Loss |
| **推理** | 单步前向 | 12 步 MaskGIT 迭代 |
| **Tokenization** | 不需要 | 需要 encode/decode |

### 4.4 注意事项

1. **mask_token_id 通常设置为 vocab_size - 1**:
   - vocab_size = 256 → mask_token_id = 255

2. **训练时的 random masking**:
   - mask_ratio_range=(0.0, 1.0) 表示随机 mask 0-100% 的 tokens
   - 类似 BERT 的 MLM (Masked Language Modeling)

3. **推理时的迭代解码**:
   - 初始: 所有 tokens 都是 <mask> (255)
   - 每步: 解码一部分 tokens,重新 mask 低置信度的 tokens
   - 最终: 得到完整的 action token 序列

4. **Tokenization 的作用**:
   - 将连续 action space 离散化为 256 个 bins
   - 使得可以用 classification (CE loss) 代替 regression (L1 loss)
   - 支持 MaskGIT 风格的迭代解码

## 5. 完整训练脚本示例片段

```python
# 在 finetune.py 中的修改

from prismatic.models.action_heads import ActionTokenizer, DiscreteDiffusionActionHead

# 初始化
action_tokenizer = ActionTokenizer(bins=256, min_action=-1.0, max_action=1.0)
action_head = DiscreteDiffusionActionHead(
    hidden_dim=896,
    num_action_tokens=NUM_ACTIONS_CHUNK * ACTION_DIM,
    vocab_size=256,
    mask_token_id=255,
).to(device)

# 训练循环
for batch in dataloader:
    # 1. Tokenize ground truth actions
    gt_actions_np = batch["actions"].cpu().numpy()  # (B, 8, 7)
    target_tokens_np = action_tokenizer.encode(gt_actions_np)  # (B, 56)
    target_tokens = torch.from_numpy(target_tokens_np).to(device)

    # 2. VLA forward
    output = vla_model(..., output_hidden_states=True)
    multi_layer_hidden_states = torch.stack(output.hidden_states, dim=1)

    # 3. Compute loss
    loss = action_head.compute_loss(
        multi_layer_hidden_states,
        target_tokens,
        proprio=batch["proprio"]
    )

    # 4. Backward
    loss.backward()
    optimizer.step()
```
"""

