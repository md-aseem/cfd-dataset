import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, seq_len, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = num_layers
        self.d_model = d_model
        
        # Multi-Head Self-Attention layers
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        
        # Feed-Forward layers
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
        # Reduce sequence length
        self.fc_reduce = nn.Linear(seq_len * d_model, d_model)  # Alternative to pooling
        
        # MLP transformation
        self.fc1 = nn.Linear(d_model, d_model)  # You can choose the intermediate sizes

    def forward(self, x):

        assert x.shape[-1] == self.d_model, "Input dimension and d_model must be equal"
        
        # Multi-Head Self-Attention and Feed-Forward layers
        for i in range(self.num_layers):
            # Self-Attention
            x_attention, _ = self.self_attention_layers[i](x, x, x)
            x = x + self.dropout(x_attention)
            
            # Layer normalization
            x = self.layer_norms[i](x)
            
            # Feed-Forward
            x_ff = self.feed_forward_layers[i](x)
            x = x + self.dropout(x_ff)
            
            # Layer normalization
            x = self.layer_norms[i](x)

        x = x.view(1, -1)
        x = self.fc_reduce(x)

        # MLP Transformation
        x = nn.ReLU()(self.fc1(x))

        return x
    
