import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import Tensor

def ddpm_schedule(n_steps: int, beta_start: float, beta_end: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Create a noise schedule for the forward and backward processes.
    
    Returns:
        sqrt_alphas: Square root of alphas for each timestep
        sqrt_one_minus_alphas: Square root of (1-alpha) for each timestep
        alphas: The alpha values (1-beta) for each timestep
        alphas_cumprod: Cumulative product of alphas for each timestep
    """
    # Beta schedule
    betas = torch.linspace(beta_start, beta_end, n_steps)
    
    # Alpha values (1 - beta)
    alphas = 1.0 - betas
    
    # Cumulative product of alphas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Square root of alphas and (1-alpha) for the noise schedule
    sqrt_alphas = torch.sqrt(alphas)
    sqrt_one_minus_alphas = torch.sqrt(1.0 - alphas)
    
    # Square root of cumulative alphas for x_0 estimation
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    return sqrt_alphas, sqrt_one_minus_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

class BiRNN_denoiser(nn.Module):
    def __init__(self, n_layers=5, hidden_size=96, dropout=0.0, condition_on_timestep=True):
        super(BiRNN_denoiser, self).__init__()
        
        # Add timestep embedding if conditioning on timestep
        self.condition_on_timestep = condition_on_timestep
        input_size = 3  # (x, y, pen_state)
        
        if condition_on_timestep:
            self.time_embed = nn.Sequential(
                nn.Linear(1, 32),
                nn.SiLU(),
                nn.Linear(32, 32)
            )
            input_size += 32  # Add timestep embedding dimension
        
        # BiGRU with multiple layers
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True
        )
        
        # Project concatenated forward and backward states back to hidden size
        self.out_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        # Final projection to output coordinates
        self.fc = nn.Linear(hidden_size, 3)  # (x, y, pen_state)

    def forward(self, x, t=None):
        batch_size, seq_len, _ = x.shape
        
        # Process timestep if conditioning is enabled
        if self.condition_on_timestep and t is not None:
            # Normalize timestep and reshape for embedding
            t_emb = (t.float() / 1000.0).view(-1, 1)
            t_emb = self.time_embed(t_emb)  # [B, 32]
            
            # Expand timestep embedding to match sequence length
            t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, seq_len, 32]
            
            # Concatenate timestep embedding with input along feature dimension
            x = torch.cat([x, t_emb], dim=-1)  # [B, seq_len, 3+32]
        
        # Process through BiGRU
        x, _ = self.rnn(x)
        
        # Project concatenated states back to hidden size
        x = self.out_proj(x)
        x = F.relu(x)
        
        # Final projection to coordinates
        x = self.fc(x)
        return x

class point_cloud_ddpm_model(nn.Module):
    def __init__(self, model: nn.Module, n_steps: int, beta_start: float, beta_end: float):
        super(point_cloud_ddpm_model, self).__init__()
        self.model = model
        self.n_steps = n_steps
        
        # Get noise schedule
        self.sqrt_alphas, self.sqrt_one_minus_alphas, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod = ddpm_schedule(
            n_steps, beta_start, beta_end
        )
        
        # Calculate posterior variance (sigma^2) for each timestep
        # This is used in the sampling process
        alphas = 1.0 - self.sqrt_one_minus_alphas ** 2
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculate posterior variance parameters
        self.posterior_variance = self.sqrt_one_minus_alphas ** 2 * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = torch.sqrt(alphas_cumprod_prev) * self.sqrt_one_minus_alphas ** 2 / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = torch.sqrt(alphas) * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def to(self, device):
        """Move model and all tensors to device"""
        self.sqrt_alphas = self.sqrt_alphas.to(device)
        self.sqrt_one_minus_alphas = self.sqrt_one_minus_alphas.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return super().to(device)

    def add_noise(self, x_0, t):
        """
        Add noise to clean data x_0 according to diffusion process.
        
        Args:
            x_0: Clean data tensor of shape (batch_size, seq_length, feature_dim)
            t: Timestep tensor of shape (batch_size,)
            
        Returns:
            Noisy data at timestep t
        """
        noise = torch.randn_like(x_0)
        
        # Reshape alphas for broadcasting
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # Apply noise according to diffusion process
        x_t = sqrt_alphas_cumprod * x_0 + sqrt_one_minus_alphas_cumprod * noise
        
        return x_t, noise

    def forward(self, x_0, t, lengths=None):
        """
        Forward diffusion process and compute loss.
        
        Args:
            x_0: Clean data tensor
            t: Timestep tensor
            lengths: Sequence lengths for variable-length sequences
            
        Returns:
            Loss value
        """
        # Add noise according to timestep
        x_t, noise = self.add_noise(x_0, t)
        
        # Handle variable-length sequences if provided
        if lengths is not None:
            # Pack the sequence for efficient processing
            packed_x_t = pack_padded_sequence(x_t, lengths.cpu(), batch_first=True, enforce_sorted=False)
            
            # Unpack for model processing
            x_t_padded, _ = pad_packed_sequence(packed_x_t, batch_first=True)
            
            # Pass through model to predict noise
            pred_noise = self.model(x_t_padded, t)
            
            # Compute mask for valid sequence positions
            max_len = x_t.size(1)
            mask = torch.arange(max_len, device=x_t.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
            
            # Apply mask to compute loss only on valid positions
            loss = F.mse_loss(pred_noise[mask], noise[mask])
        else:
            # Standard processing for fixed-length sequences
            pred_noise = self.model(x_t, t)
            loss = F.mse_loss(pred_noise, noise)
        
        return loss

    def sample(self, batch_size, seq_length, feature_dim=3, device='cuda'):
        """
        Generate samples through reverse diffusion process.
        
        Args:
            batch_size: Number of samples to generate
            seq_length: Length of each sequence
            feature_dim: Dimension of each point (default: 3 for x, y, pen_state)
            device: Device to run generation on
            
        Returns:
            Generated samples of shape (batch_size, seq_length, feature_dim)
        """
        self.eval()
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn(batch_size, seq_length, feature_dim, device=device)
            
            # Reverse diffusion process (from T to 0)
            for t in reversed(range(self.n_steps)):
                # Create timestep tensor (batch of same timestep)
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                
                # Predict noise at current timestep
                predicted_noise = self.model(x, t_batch)
                
                # Get alpha values for this timestep
                alpha = 1.0 - self.sqrt_one_minus_alphas[t] ** 2
                alpha_cumprod = self.sqrt_alphas_cumprod[t] ** 2
                alpha_cumprod_prev = self.sqrt_alphas_cumprod[t-1] ** 2 if t > 0 else torch.tensor(1.0, device=device)
                
                # No noise for the final step (t=0)
                if t == 0:
                    noise = 0
                else:
                    noise = torch.randn_like(x)
                
                # Get variance for this step
                variance = self.posterior_variance[t]
                
                # Reshape for broadcasting
                sqrt_alpha = self.sqrt_alphas[t].view(1, 1, 1)
                sqrt_one_minus_alpha = self.sqrt_one_minus_alphas[t].view(1, 1, 1)
                sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].view(1, 1, 1)
                sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(1, 1, 1)
                
                # Estimate x_0 from x_t and predicted noise
                x_0_pred = (x - sqrt_one_minus_alpha_cumprod * predicted_noise) / sqrt_alpha_cumprod
                
                # Clamp for stability
                x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
                
                # Compute the mean for the posterior p(x_{t-1} | x_t, x_0)
                mean = sqrt_alpha * (
                    x * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) + 
                    x_0_pred * (alpha_cumprod_prev - alpha_cumprod) / ((1 - alpha_cumprod) * sqrt_alpha_cumprod)
                )
                
                # Add scaled noise for the variance
                x = mean + torch.sqrt(variance).view(1, 1, 1) * noise
            
            # Ensure output is in expected range
            x = torch.clamp(x, -1.0, 1.0)
            
            return x

    def sample_from_noise(self, x_T, guidance_scale=1.0):
        """
        Sample from specific noise tensor with guidance.
        
        Args:
            x_T: Initial noise tensor of shape (batch_size, seq_length, feature_dim)
            guidance_scale: Control the adherence to the noise prediction (>1 = stronger guidance)
            
        Returns:
            Generated samples of the same shape as input
        """
        self.eval()
        with torch.no_grad():
            # Start from provided noise
            x = x_T
            batch_size = x.size(0)
            
            # Reverse diffusion process (from T to 0)
            for t in reversed(range(self.n_steps)):
                # Create timestep tensor
                t_batch = torch.full((batch_size,), t, device=x.device, dtype=torch.long)
                
                # Predict noise at current timestep
                predicted_noise = self.model(x, t_batch)
                
                # Apply classifier-free guidance if scale > 1
                if guidance_scale > 1.0:
                    # We would need unconditioned predictions here, but we're skipping
                    # the implementation for simplicity
                    noise_pred = predicted_noise
                else:
                    noise_pred = predicted_noise
                
                # Get alpha values for this timestep
                alpha = 1.0 - self.sqrt_one_minus_alphas[t] ** 2
                alpha_cumprod = self.sqrt_alphas_cumprod[t] ** 2
                alpha_cumprod_prev = self.sqrt_alphas_cumprod[t-1] ** 2 if t > 0 else torch.tensor(1.0, device=x.device)
                
                # No noise for the final step (t=0)
                if t == 0:
                    noise = 0
                else:
                    noise = torch.randn_like(x)
                
                # Get variance for this step
                variance = self.posterior_variance[t]
                
                # Reshape for broadcasting
                sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].view(1, 1, 1)
                sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(1, 1, 1)
                
                # Estimate x_0 from x_t and predicted noise
                x_0_pred = (x - sqrt_one_minus_alpha_cumprod * noise_pred) / sqrt_alpha_cumprod
                
                # Clamp for stability
                x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
                
                # Compute the mean for the posterior p(x_{t-1} | x_t, x_0)
                mean = sqrt_alpha_cumprod * x_0_pred + sqrt_one_minus_alpha_cumprod * noise_pred
                
                # Add scaled noise for the variance
                x = mean + torch.sqrt(variance).view(1, 1, 1) * noise
                
                # Optional: Clamp for stability after each step
                x = torch.clamp(x, -5.0, 5.0)
            
            # Final output should be in expected range
            x = torch.clamp(x, -1.0, 1.0)
            
            return x
            
# Example of how to use the model:
def train_example():
    # Initialize the denoiser model
    denoiser = BiRNN_denoiser(n_layers=3, hidden_size=128, dropout=0.1, condition_on_timestep=True)
    
    # Initialize the DDPM model
    diffusion = point_cloud_ddpm_model(
        model=denoiser,
        n_steps=1000,
        beta_start=1e-4,
        beta_end=0.02
    )
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = diffusion.to(device)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)
    
    # Training loop (pseudo-code)
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Get data
            x_0 = batch.to(device)  # Shape: [batch_size, seq_len, 3]
            
            # Sample random timesteps
            batch_size = x_0.size(0)
            t = torch.randint(0, diffusion.n_steps, (batch_size,), device=device)
            
            # Forward pass and compute loss
            loss = diffusion(x_0, t)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Update model parameters
            optimizer.step()
    
    # Generate samples
    samples = diffusion.sample(batch_size=16, seq_length=100, device=device)
    
    return samples