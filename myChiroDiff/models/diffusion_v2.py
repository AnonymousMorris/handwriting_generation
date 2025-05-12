import torch
import torch.nn as nn
from torch import Tensor, device, linspace
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

def ddpm_schedule(n_steps: int, beta_start: float, beta_end: float) -> tuple[Tensor, Tensor]:
    betas = torch.linspace(beta_start, beta_end, n_steps)
    sqrt_alphas = torch.sqrt(1.0 - betas)
    sqrt_one_minus_alphas = torch.sqrt(betas)
    return sqrt_alphas, sqrt_one_minus_alphas


class BiRNN_denoiser(nn.Module):
    def __init__(self, n_layers = 5, hidden_size=96, dropout=0.0):
        super(BiRNN_denoiser, self).__init__()
        self.rnn = nn.GRU(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
        )
        
        # Add time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # self.out_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(2 * hidden_size, 2)

    def forward(self, x, t):
        # Check if input is a PackedSequence
        is_packed = isinstance(x, torch.nn.utils.rnn.PackedSequence)
        
        # Process through RNN
        x, _ = self.rnn(x)
        if is_packed:
            x, _ = pad_packed_sequence(x, batch_first=True)
        
        # Project concatenated states back to hidden size
        # x = self.out_proj(x)
        # Final projection to coordinates
        x = self.fc(x)
        return x
    

class point_cloud_ddpm(nn.Module):
    def __init__(self, model: nn.Module, n_steps: int, beta_start: float, beta_end: float, device="cuda"):
        super(point_cloud_ddpm, self).__init__()
        self.model = model
        self.n_steps = n_steps

        # Linear noising schedule
        self.betas = linspace(beta_start, beta_end, n_steps, device=device)
        # precompute other values
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod) 
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        assert(self.betas.device == self.alphas.device == self.alphas_cumprod.device == self.alphas_cumprod_prev.device == self.sqrt_recip_alphas.device == self.sqrt_alphas_cumprod.device == self.sqrt_one_minus_alphas_cumprod.device == self.posterior_variance.device)

    # Gets multiple values at once and return it as a tensor
    def get_index_from_list(self, vals, t, x_shape):
        assert(t.device == vals.device)
        batch_size = t.shape[0]
        out = vals.gather(-1, t)
        # Reshape to [batch_size, 1, 1] to broadcast with [batch_size, seq_len, 3]
        return out.reshape(batch_size, 1, 1).to(t.device)

    def add_noise(self, x_0, t, mask=None, device="cuda"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it as well as the noise.
        If mask is provided, only adds noise to masked positions.
        
        Args:
            x_0: Input tensor
            t: Timestep tensor
            mask: Boolean mask tensor of same shape as x_0. True indicates positions to add noise to.
            device: Device to use for computation
        """
        # Move input to the specified device
        x_0 = x_0.to(device)
        t = t.to(device)
        
        noise = torch.randn_like(x_0, device=device)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        # If mask is provided, only add noise to masked positions
        if mask is not None:
            mask = mask.to(device)
            # Create masked noise where mask is True
            masked_noise = noise * mask
            # Combine masked noise and input
            x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * masked_noise
            return x_t, masked_noise

        # If no mask, add noise to all positions
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def forward(self, x_0, t, lengths=None):
        """
        Forward diffusion process and compute loss.
        
        Args:
            x_0: Clean data tensor
        """
        mask = torch.ones_like(x_0)
        mask[:,2,:] = 0
        x_t, noise = self.add_noise(x_0, t, mask=mask)
        if lengths is not None:
            x_t = pack_padded_sequence(x_t, lengths, batch_first=True, enforce_sorted=False)

        pred_noise = self.model(x_t, t)
        loss = F.mse_loss(noise[:, :, :2], pred_noise)
        return loss


    def sample_timestep(self, x, t):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        
        # Call model (current image - noise prediction)
        with torch.no_grad():
            pred_noise = self.model(x, t)
        x_denoised = sqrt_recip_alphas_t * (
            x - betas_t * torch.cat([pred_noise, torch.zeros((x.shape[0], x.shape[1], 1), device=x.device)], dim=2) * sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)
        
        # Check if any timestep is 0
        is_last_step = (t == 0).any()
        
        if is_last_step:
            return torch.clamp(x_denoised, -1, 1)
        else:
            noise = torch.randn_like(x)
            x_denoised = x_denoised + torch.sqrt(posterior_variance_t) * noise
            # Add small amount of noise for stability
            x_denoised = x_denoised + torch.randn_like(x) * 0.01
            return torch.clamp(x_denoised, -1, 1)

