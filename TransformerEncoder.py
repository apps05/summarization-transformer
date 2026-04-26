import torch
import torch.nn as nn
import math


def create_padding_mask(x, pad_token=0):

    """
    x: (B, T)
    B = batch size
    T = no.of tokens in each sequence (after padding)

    At each position :
      if real token -> 1(True)
         padding token -> 0(False)  (should be ignored while computing attention)

    returns: mask of shape (B, 1, 1, T)
    """

    mask = (x != pad_token).unsqueeze(1).unsqueeze(2)
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # pe stores position info for for each position (0 to max_len)
        #each position gets a vector of size d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        #frequency scaling term
        div_term = torch.exp(torch.arange(0, d_model,2) * (-math.log(10000) / d_model))

        #sin to even indices, cos to odd indices
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        # Stores inside the model but as non-trainable tensor 
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model) 
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask=None, context=None):

        if context is None:
            context = x  # then self-attention
        
        """ B = Batch Size
            T = no.of tokens in each sequence (after padding)
            C = no.of features per token/ embedding dimension (d_model)
        """

        B,T,C = x.shape
        S=context.shape[1] # S = T for self-attention

        Q=self.Wq(x)
        K=self.Wk(context)
        V=self.Wv(context)

        # split into multiple heads
        # (B, T, d_model) -> (B, heads, T, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1,2)

        scores = (Q @ K.transpose(-2,-1)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.bool()
            # force attention scores of masked positions to -inf (so they become 0 after softmax)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min / 2)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights=self.dropout(attention_weights)

        attention_output = attention_weights @ V

        # Combine all heads into one vector per token
        attention_output = attention_output.transpose(1,2).contiguous().view(B,T,C)

        # mixes information from all heads into one final representation
        return self.fc_out(attention_output)
    

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), #d_ff=dimension of feed forward layer
            nn.GELU(), #non-linear
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feedforward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        #pre-layer normalisation
        attention_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attention_output)

        feedforward_output = self.feedforward(self.norm2(x))
        x = x + self.dropout(feedforward_output)

        return x
    

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads,
                 num_layers, d_ff, max_len, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model

    
    def forward(self, x, pad_token_id=0):

        mask = create_padding_mask(x, pad_token_id)

        x = self.embedding(x) * math.sqrt(self.d_model) # token embedding + scaling
        x = self.pos_encoding(x)  
        x = self.dropout(x)       
        # randomly sets some elements to 0 during training
        # improves generalisation by preventing the model from depending too heavily on specific features

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
        

    






    