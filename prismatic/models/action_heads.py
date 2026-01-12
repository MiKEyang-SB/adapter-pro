"""
action_heads.py

Implementations of various action heads, which serve as alternatives to VLM sequential token prediction.
"""

import math
from turtle import forward
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
            actions_hidden_states, #(b, 25, 576, 896)
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

        return action
    

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
        h_adapter = torch.cat((h_a, p),dim=1)

        h_task = h_t
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
