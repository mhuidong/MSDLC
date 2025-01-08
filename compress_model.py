# -*- coding: utf-8 -*-
"""
Author: Huidong Ma
E-mail: mahd@nbjl.nankai.edu.cn
Date: 2024/8/28
Description: xLSTM Language Model
"""
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from dacite import from_dict
from dacite import Config as DaciteConfig
from xlstm import xLSTMLMModel, xLSTMLMModelConfig
class XLSTMModel(nn.Module):
    def __init__(self, layers, vocab_dim, timesteps, vocab_size):
        super(XLSTMModel, self).__init__()
        self.xlstm_cfg = """ 
            vocab_size: {}
            mlstm_block:
              mlstm:
                conv1d_kernel_size: 4
                qkv_proj_blocksize: 4
                num_heads: 4
            slstm_block:
              slstm:
                backend: cuda
                num_heads: 4
                conv1d_kernel_size: 4
                bias_init: powerlaw_blockdependent
              feedforward:
                proj_factor: 1.3
                act_fn: gelu
            context_length: {}
            num_blocks: {}
            embedding_dim: {}
            slstm_at: [1]
            """.format(vocab_size, timesteps, layers, vocab_dim)
        self.cfg = OmegaConf.create(self.xlstm_cfg)
        self.cfg = from_dict(data_class=xLSTMLMModelConfig, data=OmegaConf.to_container(self.cfg), config=DaciteConfig(strict=True))
        self.xlstm_stack = xLSTMLMModel(self.cfg)
    def forward(self, x):
        return self.xlstm_stack(x)