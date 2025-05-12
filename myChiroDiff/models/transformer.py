import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConditionalTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.1, condition_dim=None):
        super().__init__()
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Condition processing
        self.condition_dim = condition_dim if condition_dim else d_model
        self.condition_projector = nn.Linear(self.condition_dim, d_model)
        
        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_decoder_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        
    def forward(self, tgt, condition, tgt_mask=None, tgt_key_padding_mask=None):
        """
        tgt: Target sequence [batch_size, tgt_len]
        condition: Conditioning input [batch_size, condition_dim]
        """
        # Get embeddings and add positional encoding
        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        
        # Process condition and expand to match sequence length
        condition_projected = self.condition_projector(condition).unsqueeze(1)  # [batch_size, 1, d_model]
        condition_expanded = condition_projected.expand(-1, tgt_emb.size(1), -1)  # [batch_size, tgt_len, d_model]
        
        # Add condition to input embeddings
        decoder_input = tgt_emb + condition_expanded
        
        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Pass through transformer decoder
        # For decoder-only model, we use the condition as memory
        decoder_output = self.transformer_decoder(
            decoder_input, 
            memory=condition_projected,  # Use condition as memory
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf')."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate(self, condition, start_token_id, max_length=50, temperature=1.0, top_k=None, top_p=None):
        """Generate a sequence conditioned on the input condition."""
        batch_size = condition.size(0)
        device = next(self.parameters()).device
        
        # Start with batch of start tokens
        current_output = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # Get model predictions
            with torch.no_grad():
                logits = self(current_output, condition)
                next_token_logits = logits[:, -1, :] / temperature
            
            # Apply sampling methods
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("inf")
                
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float("inf")
            
            # Sample from the filtered distribution
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)
            
            # Add the sampled token to the output
            current_output = torch.cat([current_output, next_token], dim=1)
            
        return current_output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

    

class ControlPointTransformer(nn.Module):
    def __init__(self, 
                 n_input=2,           # Input dimension (x,y coordinates)
                 n_cond=512,          # Conditioning dimension
                 n_internal=512,      # Internal transformer dimension
                 n_head=8,            # Number of attention heads
                 dropout=0.1,         # Dropout rate
                 max_points=100):      # Maximum number of control points
        super(ControlPointTransformer, self).__init__()
        
        # Input projection for control points
        self.input_projection = nn.Linear(n_input, n_internal)
        
        # Positional encoding for control points
        self.positional_encoding = PositionalEncoding(n_internal, dropout)
        
        # Condition processing
        self.condition_projector = nn.Linear(n_cond, n_internal)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_internal,
            nhead=n_head,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=6
        )
        
        # Output projection back to control point coordinates
        self.output_projection = nn.Linear(n_internal, n_input)
        
        self.d_model = n_internal
        self.max_points = max_points
        
    def forward(self, points, condition, points_mask=None, points_key_padding_mask=None):
        """
        points: Control points sequence [batch_size, seq_len, 2]
        condition: Conditioning input [batch_size, condition_dim]
        """
        # Project input points to internal dimension
        points_emb = self.input_projection(points) * math.sqrt(self.d_model)
        points_emb = self.positional_encoding(points_emb)
        
        # Process condition and expand to match sequence length
        condition_projected = self.condition_projector(condition).unsqueeze(1)
        condition_expanded = condition_projected.expand(-1, points_emb.size(1), -1)
        
        # Add condition to input embeddings
        decoder_input = points_emb + condition_expanded
        
        # Create causal mask if not provided
        if points_mask is None:
            points_mask = self.generate_square_subsequent_mask(points.size(1)).to(points.device)
        
        # Pass through transformer decoder
        decoder_output = self.transformer_decoder(
            decoder_input,
            memory=condition_projected,
            tgt_mask=points_mask,
            tgt_key_padding_mask=points_key_padding_mask
        )
        
        # Project back to control point coordinates
        output = self.output_projection(decoder_output)
        
        return output
    
    def generate(self, condition, start_point, max_length=None, temperature=1.0):
        """Generate a sequence of control points conditioned on the input condition."""
        batch_size = condition.size(0)
        device = next(self.parameters()).device
        
        # Use max_points if max_length not specified
        max_length = max_length if max_length is not None else self.max_points
        
        # Start with batch of start points
        current_output = start_point.unsqueeze(1)  # [batch_size, 1, 2]
        
        for _ in range(max_length - 1):
            # Get model predictions
            with torch.no_grad():
                logits = self(current_output, condition)
                next_point = logits[:, -1, :]  # [batch_size, 2]
                
                # Add some noise based on temperature
                if temperature > 0:
                    noise = torch.randn_like(next_point) * temperature
                    next_point = next_point + noise
            
            # Add the predicted point to the output
            current_output = torch.cat([current_output, next_point.unsqueeze(1)], dim=1)
            
        return current_output
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    
