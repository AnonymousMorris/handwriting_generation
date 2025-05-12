import torch
import torch.nn as nn
import torch.nn.functional as F

class HandwritingTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=128,  # Character vocabulary size
        d_model=512,     # Model dimension
        nhead=8,         # Number of attention heads
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        style_dim=128,   # Style embedding dimension
        stroke_dim=3,    # x, y, pen-up (binary)
        max_seq_len=1000,
        dropout=0.1
    ):
        super().__init__()
        
        # Character embeddings
        self.char_embeddings = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Style encoder (processes example handwriting to extract style)
        self.style_encoder = StyleEncoder(stroke_dim, style_dim, d_model)
        
        # Style integration layer
        self.style_projection = nn.Linear(style_dim, d_model)
        
        # Transformer encoder-decoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output layer (predicts stroke parameters)
        self.output_layer = nn.Linear(d_model, stroke_dim)
        
        # MDN (Mixture Density Network) components for more realistic strokes
        self.mdn_layer = MDN(d_model, n_mixtures=20, stroke_dim=stroke_dim)
    
    def encode_style(self, style_sample):
        """Extract style embedding from handwriting sample"""
        return self.style_encoder(style_sample)
        
    def forward(self, src, tgt, style_embedding, src_mask=None, tgt_mask=None):
        """
        src: input text (batch, seq_len)
        tgt: target strokes until current step (batch, seq_len, stroke_dim)
        style_embedding: style representation (batch, style_dim)
        """
        # Embed characters and add positional encoding
        src_emb = self.positional_encoding(self.char_embeddings(src))
        
        # Project style embedding and add to source embeddings
        style_proj = self.style_projection(style_embedding).unsqueeze(1)
        src_emb = src_emb + style_proj
        
        # Transform through encoder
        memory = self.transformer_encoder(src_emb, src_mask)
        
        # Prepare autoregressive target sequence
        tgt_emb = self.positional_encoding(self.stroke_embedding(tgt))
        
        # Decoder
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask)
        
        # Generate stroke parameters through MDN
        return self.mdn_layer(output)


class StyleEncoder(nn.Module):
    """Encodes handwriting samples into style embeddings"""
    def __init__(self, stroke_dim, style_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(stroke_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.projection = nn.Linear(hidden_dim * 2, style_dim)  # *2 for bidirectional
    
    def forward(self, stroke_sample):
        _, (h_n, _) = self.lstm(stroke_sample)
        # Concatenate final forward and backward hidden states
        h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.projection(h_n)


class PositionalEncoding(nn.Module):
    """Standard transformer positional encoding"""
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)


class MDN(nn.Module):
    """Mixture Density Network for stroke prediction"""
    def __init__(self, input_size, n_mixtures, stroke_dim):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.stroke_dim = stroke_dim
        
        # Calculate output size: for each mixture, we need:
        # - 1 mixing coefficient (pi)
        # - stroke_dim means (mu)
        # - stroke_dim standard deviations (sigma)
        # - correlation coefficient for bivariate Gaussian (rho)
        output_size = n_mixtures * (1 + 2*stroke_dim + 1)
        
        self.output_layer = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        y = self.output_layer(x)
        
        # Split the output into mixture parameters
        pi, mu_x, mu_y, sigma_x, sigma_y, rho, pen_logits = self.get_mixture_params(y)
        
        return {
            'pi': pi,          # mixing coefficients
            'mu_x': mu_x,      # x-coordinate means
            'mu_y': mu_y,      # y-coordinate means
            'sigma_x': sigma_x,  # x-coordinate standard deviations
            'sigma_y': sigma_y,  # y-coordinate standard deviations
            'rho': rho,        # correlation coefficients
            'pen_logits': pen_logits  # pen-up probability logits
        }
        
    def get_mixture_params(self, y):
        # Implementation details for splitting output tensor into mixture parameters
        # This would extract all the parameters for the mixture model
        # (simplified for clarity)
        pass