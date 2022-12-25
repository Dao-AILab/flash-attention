# Copyright (c) 2022, Tri Dao.

import torch
import torch.nn as nn

from einops import rearrange

from flash_attn.utils.distributed import reduce_scatter


class GPT2Embeddings(nn.Module):

    def __init__(self, embed_dim, vocab_size, max_position_embeddings, padding_idx=None,
                 device=None, dtype=None):
        """
            If max_position_embeddings <= 0, there's no position embeddings
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx,
                                            **factory_kwargs)
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim,
                                                    **factory_kwargs)

    def forward(self, input_ids, position_ids=None):
        """
            input_ids: (batch, seqlen)
            position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings


class BertEmbeddings(nn.Module):

    def __init__(self, embed_dim, vocab_size, max_position_embeddings, type_vocab_size,
                 padding_idx=None, device=None, dtype=None):
        """
            If max_position_embeddings <= 0, there's no position embeddings
            If type_vocab_size <= 0, there's no token type embeddings
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx,
                                            **factory_kwargs)
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim,
                                                    **factory_kwargs)
        if self.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(type_vocab_size, embed_dim,
                                                      **factory_kwargs)

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        """
            input_ids: (batch, seqlen)
            position_ids: (batch, seqlen)
            token_type_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        if self.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = torch.zeros(seqlen, dtype=torch.long, device=input_ids.device)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings
        return embeddings


class ParallelGPT2Embeddings(nn.Module):

    def __init__(self, embed_dim, vocab_size, max_position_embeddings, process_group,
                 padding_idx=None, device=None, dtype=None):
        """
            If max_position_embeddings <= 0, there's no position embeddings
        """
        world_size = torch.distributed.get_world_size(process_group)
        if vocab_size % world_size != 0:
            raise ValueError(f'vocab_size ({vocab_size}) must be divisible by '
                             f'world_size ({world_size})')
        if embed_dim % world_size != 0:
            raise ValueError(f'embed_dim ({embed_dim}) must be divisible by '
                             f'world_size ({world_size})')
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.process_group = process_group
        self.word_embeddings = nn.Embedding(vocab_size // world_size, embed_dim,
                                            padding_idx=padding_idx, **factory_kwargs)
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(
                max_position_embeddings, embed_dim // world_size, **factory_kwargs
            )

    def forward(self, input_ids, position_ids=None, combine_batch_seqlen_dim=False):
        """
            input_ids: (batch, seqlen)
            position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        world_size = torch.distributed.get_world_size(self.process_group)
        if world_size <= 1:
            embeddings = self.word_embeddings(input_ids)
            if self.max_position_embeddings > 0:
                if position_ids is None:
                    position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
                position_embeddings = self.position_embeddings(position_ids)
                embeddings = embeddings + position_embeddings
            if combine_batch_seqlen_dim:
                embeddings = rearrange(embeddings, 'b s d -> (b s) d')
            return embeddings
        else:
            rank = torch.distributed.get_rank(self.process_group)
            vocab_size = self.word_embeddings.num_embeddings
            vocab_start_index, vocab_end_index = rank * vocab_size, (rank + 1) * vocab_size
            # Create a mask of valid vocab ids (1 means it needs to be masked).
            input_ids_mask = (input_ids < vocab_start_index) | (input_ids >= vocab_end_index)
            input_ids = input_ids - vocab_start_index
            input_ids[input_ids_mask] = 0
            embeddings = self.word_embeddings(input_ids)
            embeddings[input_ids_mask] = 0.0
            if self.max_position_embeddings > 0:
                if position_ids is None:
                    position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
                position_embeddings = self.position_embeddings(position_ids)
                partition_dim = self.position_embeddings.embedding_dim
                embeddings[..., rank * partition_dim:(rank + 1) * partition_dim] += position_embeddings
            if combine_batch_seqlen_dim:
                embeddings = rearrange(embeddings, 'b s d -> (b s) d')
            return reduce_scatter(embeddings, self.process_group)
