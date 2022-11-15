# Copyright (c) 2022, Tri Dao.

import torch
import torch.nn as nn

from einops import repeat


class GPT2Embeddings(nn.Module):

    def __init__(self, embed_dim, vocab_size, max_position_embeddings, padding_idx=None):
        """
            If max_position_embeddings <= 0, there's no position embeddings
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim)

    def forward(self, input_ids, position_ids=None):
        """
            input_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        input_embeddings = self.word_embeddings(input_ids)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = repeat(torch.arange(seqlen, dtype=torch.long,
                                                   device=input_ids.device),
                                      's -> b s', b=batch_size)
            position_embeddings = self.position_embeddings(position_ids)
            return input_embeddings + position_embeddings
        else:
            return input_embeddings
