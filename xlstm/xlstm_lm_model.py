# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .components.init import small_init_init_
from .utils import WeightDecayOptimGroupMixin
from .xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig


@dataclass
class xLSTMLMModelConfig(xLSTMBlockStackConfig):
    vocab_size: int = -1
    tie_weights: bool = False
    weight_decay_on_embedding: bool = False
    add_embedding_dropout: bool = False


class TinyAttention(nn.Module):
    def __init__(self, d_in, d_out=None, d_attn=64):
        super(TinyAttention, self).__init__()
        self.d_in = d_in
        self.d_out = d_out or d_in
        self.d_attn = d_attn
        self.qkv = nn.Linear(d_in, d_attn * 3)
        self.proj = nn.Linear(d_attn, d_out)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        w = torch.einsum('bnd, bmd->bnm', q, k)
        a = self.softmax(w * torch.rsqrt(torch.tensor(self.d_attn, dtype=torch.float32)))
        x = torch.einsum('bnm, bmd->bnd', a, v)
        out = self.proj(x)
        return out

class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len, tiny_attn=False):
        super(SpatialGatingUnit, self).__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.tiny_attn = tiny_attn
        self.tn = TinyAttention(2 * d_ffn, d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)  # dim维度分成两份 2 bs seql dim
        v = self.norm(v)        # 512, 1, 4096
        if self.tiny_attn:
            tn = self.tn(x)
            v = tn + self.spatial_proj(v)
        else:
            v = self.spatial_proj(v)  # 2 bs seql dim * seql seql
        out = u * v  # bs seql dim * bs seql dim     = bs seql dim       门控机制?
        return out

class xLSTMLMModel(WeightDecayOptimGroupMixin, nn.Module):
    config_class = xLSTMLMModelConfig

    def __init__(self, config: xLSTMLMModelConfig, **kwargs):
        super().__init__()
        self.config = config

        self.xlstm_block_stack = xLSTMBlockStack(config=config)
        self.token_embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim)
        self.emb_dropout = nn.Dropout(config.dropout) if config.add_embedding_dropout else nn.Identity()

        self.lm_head = nn.Linear(
            in_features=config.embedding_dim,
            out_features=config.vocab_size,
            bias=False,
        )
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.hidden_dim = 256
        self.U_map = torch.nn.Linear(config.embedding_dim, 2 * self.hidden_dim)
        self.V_map = torch.nn.Linear(self.hidden_dim, config.embedding_dim)
        self.layernorm = torch.nn.LayerNorm(config.embedding_dim)
        self.gelu = torch.nn.GELU()
        self.tn = TinyAttention(config.embedding_dim, config.embedding_dim)
        self.sgu = SpatialGatingUnit(self.hidden_dim, config.context_length)

    def reset_parameters(self):
        self.xlstm_block_stack.reset_parameters()
        small_init_init_(self.token_embedding.weight, dim=self.config.embedding_dim)

        if not self.config.tie_weights:
            small_init_init_(self.lm_head.weight, dim=self.config.embedding_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(idx)
        x = self.emb_dropout(x)
        skip = x
        x = self.U_map(x)
        x = self.gelu(x)
        x = self.sgu(x)
        x = self.V_map(x)
        x = self.layernorm(x)
        x = self.gelu(x)
        x = (x + skip) / 2
        x = self.xlstm_block_stack(x)
        x = self.lm_head(x)
        return x

    def step(
        self, idx: torch.Tensor, state: dict[str, dict[str, tuple[torch.Tensor, ...]]] = None, **kwargs
    ) -> tuple[torch.Tensor, dict[str, dict[str, tuple[torch.Tensor, ...]]]]:
        x = self.token_embedding(idx)
        x = self.emb_dropout(x)
        x, state = self.xlstm_block_stack.step(x, state=state, **kwargs)
        logits = self.lm_head(x)
        return logits, state

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        weight_decay, no_weight_decay = super()._create_weight_decay_optim_groups(**kwargs)
        # remove token embedding and add it to the correct group, accrording to the config
        weight_decay = list(weight_decay)
        removed = 0
        for idx in range(len(weight_decay)):
            if weight_decay[idx - removed] is self.token_embedding.weight:
                weight_decay.pop(idx - removed)
                removed += 1
        weight_decay = tuple(weight_decay)
        if self.config.weight_decay_on_embedding:
            weight_decay += (self.token_embedding.weight,)
        else:
            no_weight_decay += (self.token_embedding.weight,)

        return weight_decay, no_weight_decay
