# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
#from torch.nn.modules.transformer import TransformerEncoderLayer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, long_seq_length, num_short_seqs, d_ff=1024, dropout=0.1, activation="relu"):
        super(CustomTransformerEncoder, self).__init__()
        
        self.attention_masks = [self.custom_mask(long_seq_length, num_short_seqs) for _ in range(num_layers)]
        self.transformer_encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_ff, dropout=dropout) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)

    def custom_mask(self, long_seq_length, num_short_seqs):
        """
        Creates a custom attention mask.
        
        Parameters:
        - long_seq_length (int): Length of the long sequence.
        - num_short_seqs (int): Number of short sequences, each of length 1.
        
        Returns:
        - attention_mask (torch.Tensor): Custom attention mask.
        """
        total_length = long_seq_length + num_short_seqs
        
        # Initialize mask with -inf
        mask = torch.full((total_length, total_length), float('-inf'))

        
        # Long sequence can attend to itself
        mask[:long_seq_length, :long_seq_length] = 0
        
        # Each short sequence can attend to the long sequence and itself
        for idx in range(long_seq_length, total_length):
            mask[idx, :long_seq_length] = 0  # Attend to the long sequence
            mask[idx, idx] = 0  # Attend to itself
            
        return mask.to(device)

    def forward(self, x):
        for attention_mask, layer in zip(self.attention_masks, self.transformer_encoder_layers):
            x = x.transpose(0, 1)  # (L, N, E) -> (N, L, E) 
            x = self.norm(x + layer(x))
            x = x.transpose(0, 1) # (N, L, E) -> (L, N, E)
        return x
