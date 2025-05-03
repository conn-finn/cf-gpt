import State
from TransformerComponents import EmbeddingLayer, MultiheadAttentionLayer, OutputLayer, FeedForwardLayer

import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43, state=None):
        """
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        """
        super(Transformer, self).__init__()
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q

        if (state is None):
            self.state = State()
        else:
            self.state = state

        self.state.set_seeds()


        self.input_embedding_layer = EmbeddingLayer(self.input_size, self.max_length, self.word_embedding_dim, self.state)
        self.encoder_attention = MultiheadAttentionLayer(self.word_embedding_dim, self.num_heads, self.dim_feedforward, self.dim_k, self.dim_v, self.dim_q, self.state)
        self.encoder_feed_forward = FeedForwardLayer(self.word_embedding_dim, self.dim_feedforward, self.state)


        # self.target_mask = self._get_target_mask(self.max_length)


        self.target_embedding_layer = EmbeddingLayer(self.output_size, self.max_length, self.word_embedding_dim, self.state)
        self.decoder_attention_1 = MultiheadAttentionLayer(self.word_embedding_dim, self.num_heads, self.dim_feedforward, self.dim_k, self.dim_v, self.dim_q, self.state)
        self.decoder_attention_2 = MultiheadAttentionLayer(self.word_embedding_dim, self.num_heads, self.dim_feedforward, self.dim_k, self.dim_v, self.dim_q, self.state)
        self.decoder_feed_forward = FeedForwardLayer(self.word_embedding_dim, self.dim_feedforward, self.state)

        self.output_layer = OutputLayer(self.word_embedding_dim, self.output_size, self.state)



    def _get_target_mask(self, max_length):
        mask = torch.triu(torch.ones(max_length, max_length), diagonal=1).type(torch.uint8)
        mask = mask.to(self.state.device)
        mask = mask.masked_fill(mask == 1, float('-inf'))

        target_key_padding_mask = torch.where(tgt == self.pad_idx, True, False)

        # target_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=self.device)
        # target_mask = torch.where(target_mask == 0, False, True)
        
        # target_key_padding_mask = torch.where(tgt == self.pad_idx, True, False)
        return mask

    def encode(self, x):
        out = self.input_embedding_layer(x)
        out = self.encoder_attention(out)
        out = self.encoder_feed_forward(out)
        return out
    
    def decode(self, x):
        out = self.target_embedding_layer(x)
        out = self.decoder_attention_1(out)
        out = self.decoder_attention_2(out)
        out = self.decoder_feed_forward(out)
        return out

    
    def forward(self, x):
        out = self.encode(x)
        out = self.decode(out)
        out = self.output_layer(out)
        return out
