import torch
import torch.nn as nn

import BaseNeuralComponent

class FeedForwardLayer(BaseNeuralComponent):
    def __init__(self, embed_dim, dim_feedforward, state=None):
        super().__init__(state)
        self.embed_dim = embed_dim
        self.dim_feedforward = dim_feedforward
        
        self.linear1 = nn.Linear(self.embed_dim, self.dim_feedforward)
        self.linear2 = nn.Linear(self.dim_feedforward, self.embed_dim)
        self.norm_feed_forward = nn.LayerNorm(self.embed_dim)
        self.relu = nn.ReLU()
        
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm_feed_forward(out + x)
        
        return out