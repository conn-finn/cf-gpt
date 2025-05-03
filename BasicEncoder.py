import torch

class BasicEncoder:
    def __init__(self, text):
        self.all_chars = sorted(list(set(text)))
        self.char_num = len(self.all_chars)

        self._stoi = { ch:i for i,ch in enumerate(self.all_chars) }
        self._itos = { i:ch for i,ch in enumerate(self.all_chars) }


    def encode(self, line):
        return torch.tensor([self._stoi[c] for c in line], dtype=torch.long)
    
    def decode(self, line):
        return torch.tensor(''.join([self._itos[i] for i in line]), dtype=torch.long)

