import torch

class State:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.random_state_num = 42

    def set_random_state_num(self, random_state_num):
        self.random_state_num = random_state_num
        
    def get_random_state_num(self):
        return self.random_state_num

    def get_device(self):
        return self.device
