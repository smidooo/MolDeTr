"""Learned positional encoding for the deformable transformer."""

"""Learned positional encoding for the deformable transformer."""

import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Create a learned positional encoding parameter
        self.pos_embedding = nn.Parameter(torch.zeros(max_len, d_model))

    def forward(self, x):
        # x: Tensor, shape [batch_size, seq_len, d_model]

        # Get the sequence length
        seq_len = x.size(1)

        # Ensure seq_len does not exceed max_len
        if seq_len > self.max_len:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds maximum length ({self.max_len})")

        # Retrieve the positional encodings and add them to the input
        pos = self.pos_embedding[:seq_len, :]

        # Add the positional encodings to the input
        return x + pos

    def get_positional_encoding(self, x):
        # Assuming x is a tensor of shape [batch_size, seq_len, d_model]
        seq_len = x.size(1)  # Extract the sequence length as an integer
        # print(seq_len)
        if seq_len > self.max_len:
            raise ValueError(f"Requested sequence length ({seq_len}) exceeds maximum length ({self.max_len})")

        # Return the positional encoding for the requested sequence length
        return self.pos_embedding[:seq_len, :]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
