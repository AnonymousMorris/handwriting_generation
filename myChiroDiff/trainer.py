import torch
from torch.optim import SGD
from dataloader.VMNIST_dataset import VectorMNISTDataset, collate_fn
import torch.nn.functional as F
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.diffusion_v2 import BiRNN_denoiser, point_cloud_ddpm

def sample_plot_image(save_dir='results', diffusion=None, denoiser=None, epoch=0):
    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Sample parameters
    batch_size = 1
    seq_length = 100  # Adjust based on your data
    feature_dim = 3   # x, y, pen_state
    
    # Start from pure noise
    x = torch.randn(batch_size, seq_length, feature_dim, device=device)
    
    # Create figure for visualization
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    
    # Number of timesteps to visualize
    num_images = 10
    stepsize = int(diffusion.n_steps / num_images)
    
    # Visualize the sampling process
    for i in range(diffusion.n_steps-1, -1, -1):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        
        # Denoise the sample at this timestep
        x = diffusion.sample_timestep(x, t)
        
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            
            # Convert to numpy and plot
            sample_np = x[0].detach().cpu().numpy()
            
            # Plot points
            plt.scatter(sample_np[:, 0], sample_np[:, 1], c='black', s=1)
            
            # Connect points with lines where pen is down
            pen_down = sample_np[:, 2] > 0.5
            for j in range(len(pen_down)-1):
                if pen_down[j] and pen_down[j+1]:
                    plt.plot([sample_np[j, 0], sample_np[j+1, 0]], 
                            [sample_np[j, 1], sample_np[j+1, 1]], 
                            'k-', alpha=0.5)
            
            plt.axis('equal')
            plt.title(f't={i}')
    
    # Save the figure
    save_path = os.path.join(save_dir, f'sample_{epoch}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def train():

    # Data
    train_dataset = VectorMNISTDataset(data_dir='data', target_digit=4)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size = 32,
                                  shuffle = True,
                                  collate_fn=collate_fn
                                  )

    # hyper parameters
    epochs = 1000
    learning_rate = 0.0001  # Reduced learning rate for stability
    batch_size = 32
    # denoiser hyper parameters
    n_layers = 3
    hidden_size = 128
    denoiser_dropout = 0.0
    # diffusion hyper parameter
    n_steps = 1000
    beta_start = 0.0001
    beta_end = 0.02

    # init values
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize models
    denoiser = BiRNN_denoiser(n_layers = n_layers, 
                              hidden_size=hidden_size, 
                              dropout=denoiser_dropout) 
    diffusion = point_cloud_ddpm(denoiser, 
                             n_steps = n_steps, 
                             beta_start = beta_start,
                             beta_end = beta_end)

    # move models to device
    denoiser.to(device)
    diffusion.to(device)
    
    # Use Adam optimizer instead of SGD
    optimizer = torch.optim.Adam(denoiser.parameters(),
                    lr = learning_rate,
                    betas=(0.9, 0.999)
                    )

    # Training 
    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            cur_batch_size = batch[0].shape[0]
            t = torch.randint(0, n_steps, (cur_batch_size,), device=device).long()
            loss = diffusion(batch[0], t, lengths=batch[2])
            loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
            optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item():.6f} ")  # Added more decimal places for better monitoring
            sample_plot_image(save_dir='results', diffusion=diffusion, denoiser=denoiser, epoch=epoch)


if __name__ == "__main__":
    train()