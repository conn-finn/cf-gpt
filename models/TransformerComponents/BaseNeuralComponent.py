import torch.nn as nn
import State

class BaseNeuralComponent(nn.Module):
    def __init__(self, state=None):
        super().__init__()
        
        if (state is None):
            self.state = State()
        else:
            self.state = state
