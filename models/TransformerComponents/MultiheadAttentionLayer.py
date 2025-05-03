import torch
import torch.nn as nn

import BaseNeuralComponent

class MultiheadAttentionLayer(BaseNeuralComponent):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dim_k, dim_v, dim_q, state=None):
        super().__init__(state)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q

        # Head #1
        self.k1 = nn.Linear(self.embed_dim, self.dim_k)
        self.v1 = nn.Linear(self.embed_dim, self.dim_v)
        self.q1 = nn.Linear(self.embed_dim, self.dim_q)
        
        # Head #2
        self.k2 = nn.Linear(self.embed_dim, self.dim_k)
        self.v2 = nn.Linear(self.embed_dim, self.dim_v)
        self.q2 = nn.Linear(self.embed_dim, self.dim_q)
        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.embed_dim)
        self.norm_mh = nn.LayerNorm(self.embed_dim)
        

    def forward(self, x):
        k1 = self.k1(x)
        v1 = self.v1(x)
        q1 = self.q1(x)
        
        k2 = self.k2(x)
        v2 = self.v2(x)
        q2 = self.q2(x)
        
        attention1 = torch.bmm(q1, k1.transpose(1,2)) / torch.sqrt(self.dim_k)
        attention1 = self.softmax(attention1)
        attention1 = torch.bmm(attention1, v1)

        attention2 = torch.bmm(q2, k2.transpose(1,2)) / torch.sqrt(self.dim_k)
        attention2 = self.softmax(attention2)
        attention2 = torch.bmm(attention2, v2)

        attention = torch.cat((attention1, attention2), dim=2)

        res = self.attention_head_projection(attention)
        outputs = self.norm_mh(res + x)

        return outputs