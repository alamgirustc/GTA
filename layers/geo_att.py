import torch
import torch.nn as nn
import torch.nn.functional as F

class GeoAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, reduction_ratio=16):
        super(GeoAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        # Linear layers to project the input features for multi-head attention
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Final linear layer to combine the outputs from each head
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Spatial attention layer (corresponding to sc_att.py)
        self.attention_last2 = nn.Linear(embed_dim, 1)  # for spatial attention

        # Squeeze-and-Excitation network for channel attention (for recalibrating feature channels)
        self.fc1 = nn.Linear(embed_dim, embed_dim // reduction_ratio, bias=False)  # First FC layer for bottleneck
        self.fc2 = nn.Linear(embed_dim // reduction_ratio, embed_dim, bias=False)  # Second FC layer for channel excitation

    def forward(self, att_feats, geo_feats):
        batch_size = att_feats.size(0)

        # Debugging: Print input shapes
        #print(f"att_feats shape: {att_feats.shape}")
        #print(f"geo_feats shape: {geo_feats.shape}")

        # Project the inputs to multiple heads for standard attention
        queries = self.query_proj(att_feats).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key_proj(geo_feats).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_proj(geo_feats).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute the scaled dot-product attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.head_dim ** 0.5
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights to the values
        attention_output = torch.matmul(attention_weights, values)

        # Concatenate the outputs from all heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # Debugging: Print attention output shape after multi-head attention
        #print(f"attention_output shape: {attention_output.shape}")

        ### Updated Spatial Attention (without collapsing the sequence length)
        alpha_spatial = self.attention_last2(attention_output).squeeze(-1)  # Shape: [batch_size, seq_len], e.g., [10, 57]
        alpha_spatial = F.softmax(alpha_spatial, dim=-1)  # Normalize across the spatial dimension
        attention_output_spatial = attention_output * alpha_spatial.unsqueeze(-1)  # Shape: [batch_size, seq_len, embed_dim]

        # Debugging: Print spatial attention output shape (should retain sequence length)
        #print(f"attention_output_spatial shape: {attention_output_spatial.shape}")

        ### Squeeze-and-Excitation (SE) Channel Attention
        # Squeeze: Global pooling over spatial dimensions to create channel descriptors
        attention_output_pool = attention_output.mean(dim=1)  # Shape: [batch_size, embed_dim], e.g., [10, 1024]

        # Excitation: Fully connected bottleneck layers for learning channel attention weights
        se = self.fc1(attention_output_pool)  # Bottleneck layer
        se = F.relu(se)  # Apply ReLU activation
        se = self.fc2(se)  # Expand back to original feature dimension
        se = torch.sigmoid(se)  # Apply sigmoid to get channel attention weights
        # Shape: [batch_size, embed_dim], e.g., [10, 1024]

        # Debugging: Print SE channel attention shape
        #print(f"SE channel attention shape: {se.shape}")

        ### Combine Spatial and SE Channel Attention
        # Recalibrate the spatially attended features with the SE channel attention
        attention_output_final = attention_output_spatial * se.unsqueeze(1)  # Multiply each feature by its channel attention

        # Debugging: Print final attention output shape (after combining spatial and channel attention)
        #print(f"attention_output_final shape: {attention_output_final.shape}")

        ### Residual Connection and Final Output
        # Apply residual connection and layer normalization
        output = self.layer_norm(att_feats + geo_feats * attention_output_final)

        # Debugging: Print output shape after residual connection
        #print(f"output shape: {output.shape}")

        return output
