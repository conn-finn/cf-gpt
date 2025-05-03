import torch
import random
import numpy as np

class State:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.random_state_num = 42
        
    def get_random_state_num(self):
        return self.random_state_num
    
    def set_random_state_num(self, random_state_num):
        self.random_state_num = random_state_num

    def get_device(self):
        return self.device
        
    def set_device(self, device):
        self.device = device
    
    
    def set_seeds(self):
        seed = self.random_state_num

        random.seed(seed)
        np.random.seed(seed)
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
