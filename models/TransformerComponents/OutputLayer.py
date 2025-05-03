import torch
import torch.nn as nn

import State

class OutputLayer(nn.Module):
    def __init__(self, embed_dim, output_size, state=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_size = output_size

        self.linear = nn.Linear(self.embed_dim, self.output_size)
        self.softmax = nn.LogSoftmax(dim=2)

        if (state is None):
            self.state = State()
        else:
            self.state = state

    def forward(self, x):
        output = self.linear(x)
        output = self.softmax(output)
        
        return output