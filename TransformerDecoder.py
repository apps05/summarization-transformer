import torch
import torch.nn as nn
import math


class Decoder(nn.Module):
    

    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model        = d_model
        self.number_of_heads = num_heads
        self.head_dim       = d_model // num_heads
        self.dropout        = nn.Dropout(dropout)

       # Self attention projection matrices, we should combine all into one for speed apparently.
        self.qkv      = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # This is for cross attention
        self.q_proj         = nn.Linear(d_model, d_model)
        self.kv_proj        = nn.Linear(d_model, 2 * d_model)
        self.cross_out_proj = nn.Linear(d_model, d_model)

        # Pre layer norms, we changed from post layer to pre layer.
        self.ln1 = nn.LayerNorm(d_model)  
        self.ln2 = nn.LayerNorm(d_model) 
        self.ln3 = nn.LayerNorm(d_model)

        # This is for the final feed forward network.
        self.activation = nn.GELU()
        self.W1 = nn.Linear(d_model, d_model * 4)
        self.W2 = nn.Linear(d_model * 4, d_model)

    # Normal multi head attention, we use this for both self and cross attention.
    def scaled_dot_product(self, q, k, v, mask=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask   = mask.bool()
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min / 2)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out     = torch.matmul(weights, v)
        B = q.size(0)
        T = q.size(2)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return out

    # Cross attention
    def cross_attention(self, x, encoder_output, cross_mask=None):

        B, T, _ = x.shape
        S = encoder_output.shape[1]

        q  = self.q_proj(x).view(B, T, self.number_of_heads, self.head_dim).transpose(1, 2)

        kv = self.kv_proj(encoder_output)
        kv = kv.view(B, S, 2, self.number_of_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)   # (2, B, H, S, head_dim)
        k, v = kv[0], kv[1]

        out = self.scaled_dot_product(q, k, v, cross_mask)
        return self.cross_out_proj(out)

    def forward(self, x, encoder_output,
                self_attn_mask=None,
                cross_attn_mask=None):

        B, T, D = x.shape

       # We mask also and then self attention
        residual = x
        x_norm   = self.ln1(x)

        qkv = self.qkv(x_norm)
        qkv = qkv.view(B, T, 3, self.number_of_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) 
        q, k, v = qkv[0], qkv[1], qkv[2]
        self_out = self.scaled_dot_product(q, k, v, self_attn_mask)
        self_out = self.dropout(self.out_proj(self_out))
        x = residual + self_out

        # Now cross attention with padding mask only
        residual   = x
        x_norm     = self.ln2(x)
        cross_out  = self.dropout(self.cross_attention(x_norm, encoder_output, cross_attn_mask))
        x = residual + cross_out

        residual = x
        x_norm   = self.ln3(x)
        ff_out   = self.dropout(self.W2(self.dropout(self.activation(self.W1(x_norm)))))
        x = residual + ff_out

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_layers, vocab_size,
                 max_len=5000, num_heads=8, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Sin waala positional encoding 
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000) / d_model)
        )


        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

        # This is the casual mask for self attention. We should combine this with the padding mask later
        causal = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer("causal_mask", causal.unsqueeze(0).unsqueeze(0))
        self.layers = nn.ModuleList([
            Decoder(d_model, num_heads, dropout) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output,
                padding_mask_for_self_attention=None,
                padding_mask_for_cross_attention=None):

        B, T = x.shape

        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pe[:, :T]
        x = self.dropout(x)

        causal_mask = self.causal_mask[:, :, :T, :T].bool()

        # Yeah nice variable name says it all
        if padding_mask_for_self_attention is not None:

            pad_key = padding_mask_for_self_attention.unsqueeze(1).unsqueeze(2)
            pad_key = pad_key.expand(B, 1, T, T)                               
            self_attn_mask = causal_mask & pad_key
        else:
            self_attn_mask = causal_mask 

       
        if padding_mask_for_cross_attention is not None:
            cross_attn_mask = (
                padding_mask_for_cross_attention.unsqueeze(1).unsqueeze(2)
            )
        else:
            cross_attn_mask = None
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)

        x = self.final_norm(x)
        return self.output_layer(x)