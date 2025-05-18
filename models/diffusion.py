import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import Tensor

# output: (sqrt_alphas, sqrt_one_minus_alphas)
def ddpm_schedule(n_steps: int, beta_start: float, beta_end: float) -> tuple[Tensor, Tensor]:
    betas = torch.linspace(beta_start, beta_end, n_steps)
    sqrt_alphas = torch.sqrt(1.0 - betas)
    sqrt_one_minus_alphas = torch.sqrt(betas)
    return sqrt_alphas, sqrt_one_minus_alphas

class BiRNN_denoiser(nn.Module):
    def __init__(self, n_layers=5, hidden_size=96, dropout=0.0):
        super(BiRNN_denoiser, self).__init__()
        # BiGRU with multiple layers
        # input: (x, y, pen_state)
        # output: 2 * 96 hidden state (forward and backward are concatenated)
        self.rnn = nn.GRU(
            input_size=3,  # (x, y, pen_state)
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,  # dropout only between layers
            bidirectional=True
        )
        
        # Project concatenated forward and backward states back to hidden size
        self.out_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        # Final projection to output coordinates
        self.fc = nn.Linear(hidden_size, 3)  # (x, y, pen_state)

    def forward(self, x):
        # Check if input is a PackedSequence
        is_packed = isinstance(x, torch.nn.utils.rnn.PackedSequence)
        
        if is_packed:
            # Process through BiGRU with packed sequence
            x, _ = self.rnn(x)
            # Unpack the output
            x, _ = pad_packed_sequence(x, batch_first=True)
        else:
            # Process through BiGRU
            x, _ = self.rnn(x)
        
        # Project concatenated states back to hidden size
        x = self.out_proj(x)
        # Final projection to coordinates
        x = self.fc(x)
        return x

class point_cloud_ddpm_model(nn.Module):
    def __init__(self, model: nn.Module, n_steps: int, beta_start: float, beta_end: float):
        super(point_cloud_ddpm_model, self).__init__()
        self.model = model
        self.n_steps = n_steps
        self.sqrt_alphas, self.sqrt_one_minus_alphas = ddpm_schedule(n_steps, beta_start, beta_end)
        # Move schedule tensors to GPU
        self.sqrt_alphas = self.sqrt_alphas.cuda()
        self.sqrt_one_minus_alphas = self.sqrt_one_minus_alphas.cuda()

    def ddpm_forward(self, x_t, t, lengths=None):
        noise = torch.randn_like(x_t)
        # Reshape schedule tensors for broadcasting
        sqrt_alphas = self.sqrt_alphas[t].view(-1, 1, 1)
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas[t].view(-1, 1, 1)
        x_t = sqrt_alphas * x_t + sqrt_one_minus_alphas * noise
        return x_t

    def ddpm_backward(self, x_t, t, lengths=None):
        if lengths is not None:
            # Pack the sequence for efficient processing
            x_packed = pack_padded_sequence(x_t, lengths.cpu(), batch_first=True, enforce_sorted=False)
            # Unpack before passing to model
            x_unpacked, _ = pad_packed_sequence(x_packed, batch_first=True)
            # Process through model
            pred_noise = self.model(x_unpacked)
        else:
            pred_noise = self.model(x_t)
            
        # Reshape schedule tensors for broadcasting
        sqrt_alphas = self.sqrt_alphas[t].view(-1, 1, 1)
        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas[t].view(-1, 1, 1)
        x_t = (x_t - sqrt_one_minus_alphas * pred_noise) / sqrt_alphas
        return x_t

    def sample(self, x_t):
        """
        Generate samples through reverse diffusion process.
        
        Args:
            x_t: Initial noise tensor of shape (batch_size, seq_length, feature_dim)
            
        Returns:
            Generated samples of the same shape as input
        """
        self.eval()
        with torch.no_grad():
            # Start from pure noise
            x = x_t
            
            # Reverse diffusion process
            for t in reversed(range(self.n_steps)):
                # Create timestep tensor
                t_batch = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
                
                # Predict and remove noise
                x = self.ddpm_backward(x, t_batch)
                
                # Add small amount of noise for stability
                if t > 0:
                    noise = torch.randn_like(x) * 0.01
                    x = x + noise
                    
                # Clip values to reasonable range
                x = torch.clamp(x, -1, 1)
            
            return x
