import torch
import torch.nn as nn

import BaseNeuralComponent

class EmbeddingLayer(BaseNeuralComponent):
    def __init__(self, input_size, max_length, embed_dim, state=None):
        super().__init__(state)

        self.input_size = input_size
        self.max_length = max_length
        self.embed_dim = embed_dim
        
        self.word_embedding = nn.Embedding(self.input_size, self.word_embedding_dim)
        self.position_embedding = nn.Embedding(self.max_length, self.word_embedding_dim)


    def forward(self, x):
        word_embeddings = self.word_embedding(x)
        position_embeddings = self.position_embedding(torch.arange(self.max_length).to(self.state.get_device()))

        return word_embeddings + position_embeddings