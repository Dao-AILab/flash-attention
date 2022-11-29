# Inspired by https://github.com/NVIDIA/Megatron-LM/blob/main/tasks/zeroshot_gpt/datasets.py
# Except we don't pad the last block and don't use overlapping eval
# And we return both the input and the target
import math
import numpy as np

import torch


class LMDataset(torch.utils.data.Dataset):

    def __init__(self, tokens, seq_len, drop_last=True):
        """tokens should be a numpy array
        """
        self.seq_len = seq_len
        ntokens = len(tokens)
        if drop_last:
            ntokens = ((ntokens - 1) // seq_len) * seq_len + 1
        self.ntokens = ntokens
        # We're careful not to slice tokens, since it could be a memmap'ed array or H5 dataset,
        # and slicing would load it to memory.
        self.tokens = tokens
        self.total_sequences = math.ceil((self.ntokens - 1) / self.seq_len)

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        seq_len = min(self.seq_len, self.ntokens - 1 - start_idx)
        data = torch.as_tensor(self.tokens[start_idx:(start_idx + seq_len + 1)].astype(np.int64))
        return data[:-1], data[1:].clone()
