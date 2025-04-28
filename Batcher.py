import torch
from sklearn.model_selection import train_test_split

class Batcher:
    def __init__(self, data, state, test_size=0.2, batch_size=32, block_size=8):
        self.full_data = data
        self.batch_size = batch_size
        self.block_size = block_size

        self.training_data, self.test_data = train_test_split(data, test_size=test_size, random_state=state.get_random_state_num())
        

    # modified from https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1739s&ab_channel=AndrejKarpathy
    def get_random_batch(self, is_training):
        data = self.training_data if is_training else self.test_data

        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y