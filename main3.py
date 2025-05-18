import os
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from models.diffusion_v3 import point_cloud_ddpm, BiRNN_denoiser
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.collections import LineCollection

def collate_fn(batch):
    # batch is a list of (data, label) tuples
    data, labels = zip(*batch)
    
    # Find the maximum sequence length in this batch
    max_len = max(x.size(0) for x in data)
    
    # Get sequence lengths
    lengths = torch.tensor([x.size(0) for x in data])
    
    # Pad all sequences to max_len
    padded_data = []
    for seq in data:
        # Create a padded tensor of zeros with the same number of features
        padded = torch.zeros(max_len, seq.size(1))
        # Copy the actual data
        padded[:seq.size(0), :] = seq
        padded_data.append(padded)
    
    # Stack the padded sequences and labels
    return torch.stack(padded_data), torch.stack(labels), lengths


class DeltaVMNIST(Dataset):
    def __init__(self, data_dir, target_digit=None):
        self.data = []
        self.labels = []
        
        # Convert to absolute path if relative
        data_dir = os.path.abspath(data_dir)
        print(f"Loading data from: {data_dir}")
        
        # If target_digit is specified, only load that digit's data
        digits_to_load = [target_digit] if target_digit is not None else range(10)
        
        # Load data from each digit directory
        for digit in digits_to_load:
            digit_dir = os.path.join(data_dir, str(digit))
            if not os.path.exists(digit_dir):
                print(f"Warning: Directory {digit_dir} does not exist")
                continue
                
            for file in os.listdir(digit_dir):
                if file.endswith('.pkl'):
                    with open(os.path.join(digit_dir, file), 'rb') as f:
                        raw_data = pickle.load(f)
                        # Extract x, y, and pen_state from the drawing
                        drawing = raw_data['drawing']
                        # The drawing is a list of lists where:
                        # drawing[0] = x_coords
                        # drawing[1] = y_coords
                        # drawing[2] = pen_states
                        x_coords = np.array(drawing[0][0])
                        y_coords = np.array(drawing[0][1])
                        time_stamp = np.array(drawing[0][2])

                        # Normalize coordinates to [-1, 1] range using PyTorch's normalize
                        # First center the coordinates
                        x_coords = x_coords - x_coords.mean()
                        y_coords = y_coords - y_coords.mean()
                        
                        # Then scale to [-1, 1] range
                        x_scale = np.max(np.abs(x_coords))
                        y_scale = np.max(np.abs(y_coords))
                        box_max = max(x_scale, y_scale)

                        assert(box_max > 0)
                        x_coords = x_coords / box_max
                        y_coords = y_coords / box_max

                        dx = np.zeros_like(x_coords)
                        dy = np.zeros_like(y_coords)

                        dx[1:] = x_coords[1:] - x_coords[:-1]
                        dy[1:] = y_coords[1:] - y_coords[:-1]
                        
                        # Stack the coordinates and pen states
                        formatted_data = np.stack([x_coords, y_coords, time_stamp, dx, dy], axis=1)
                        self.data.append(formatted_data)
                        self.labels.append(digit)
        
        if len(self.data) == 0:
            raise ValueError(f"No data found in {data_dir}. Please ensure the data directory contains subdirectories 0-9 with .pkl files.")
            
        # Convert to tensors
        self.data = [torch.FloatTensor(d) for d in self.data]
        self.labels = torch.LongTensor(self.labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def reconstruct_repr(x):
    # Set first point's dx and dy to 0
    x[:,:,3:] = 0
    # Calculate dx and dy as differences between consecutive points
    x[:,1:,3] = x[:,1:,0] - x[:,:-1,0]  # dx = current_x - previous_x
    x[:,1:,4] = x[:,1:,1] - x[:,:-1,1]  # dy = current_y - previous_y
    return x

def verify_model(model, num_samples=5, device='cuda', epoch=None):
    """
    Verify the trained diffusion model by generating samples and saving them to results directory.
    
    Args:
        model: Trained diffusion model
        num_samples: Number of samples to generate
        device: Device to run inference on
        epoch: Current epoch number (for filename)
    """
    model.eval()
    with torch.no_grad():
        # Generate random noise
        batch_size = num_samples
        seq_length = 100  # Typical sequence length for MNIST digits
        feature_dim = 5   # x, y, pen_state
        
        # Create random noise
        x = torch.randn(batch_size, seq_length, feature_dim, device=device)
        x = reconstruct_repr(x)
        
        # Generate samples using the new sampling process
        for t in reversed(range(model.n_steps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = model.sample_timestep(x, t_batch)
            
        # Move to CPU for visualization
        samples = x.cpu().numpy()
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(results_dir, exist_ok=True)
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        epoch_str = f"_epoch{epoch}" if epoch is not None else ""
        
        # Create and save visualization
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        if num_samples == 1:
            axes = [axes]
            
        for i in range(num_samples):
            # Extract x, y coordinates and pen states
            x = samples[i, :, 0]
            y = samples[i, :, 1]
            pen_states = samples[i, :, 2]
            
            ax = axes[i]
            cmap = plt.cm.viridis
            
            # Prepare segments for LineCollection
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # Use the average pen state for each segment for color
            segment_colors = cmap((pen_states[:-1] + pen_states[1:]) / 2)
            
            # Create LineCollection for strokes
            lc = LineCollection(segments, colors=segment_colors, linewidths=2)
            ax.add_collection(lc)
            
            # Add scatter points colored by pen state
            sc = ax.scatter(x, y, c=pen_states, cmap=cmap, s=20, alpha=0.8)
            
            ax.set_title(f'Sample {i+1}')
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.axis('equal')
            ax.grid(True)
            
            # Add colorbar to the last subplot
            if i == num_samples - 1:
                plt.colorbar(sc, ax=ax, label='Pen State')
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(results_dir, f'visualization{epoch_str}_{timestamp}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

def train_vmnist(epochs: int, batch_size: int, learning_rate: float, n_steps: int, beta_start: float, beta_end: float, n_layers: int, hidden_size: int, dropout: float, target_digit=None):
    # Initialize dataset and dataloader
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    dataset = DeltaVMNIST(data_dir, target_digit=target_digit)
    print(f"\nDataset size: {len(dataset)} samples")
    if target_digit is not None:
        print(f"Training on digit {target_digit}")
    
    # Adjust batch size if it's too large for the dataset
    if batch_size > len(dataset):
        batch_size = max(1, len(dataset) // 2)
        print(f"Adjusted batch size to {batch_size} due to small dataset")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    denoiser = BiRNN_denoiser(n_layers, hidden_size, dropout)
    diffusion_model = point_cloud_ddpm(
        model=denoiser,
        n_steps=n_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device
    ).to(device)
    
    optimizer = Adam(diffusion_model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        diffusion_model.train()
        total_loss = 0
        
        for batch_idx, (data, _, lengths) in enumerate(dataloader):
            data = data.to(device)
            # Keep lengths on CPU as required by pack_padded_sequence
            lengths = lengths.cpu()
            batch_size = data.size(0)
            
            # Sample random timesteps
            t = torch.randint(0, n_steps, (batch_size,), device=device)
            
            # Forward pass with the new model
            loss = diffusion_model(data, t, lengths)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
        
        # Verify model after each epoch
        if epoch % 100 == 0:  # Verify every 100 epochs
            print(f"\nVerifying model after epoch {epoch}")
            verify_model(diffusion_model, num_samples=5, device=device, epoch=epoch)
            torch.save({
                'model_state_dict': diffusion_model.state_dict(),
            }, os.path.join("ckpt", f"model_epoch{epoch}.pt"))
    
    return diffusion_model
if __name__ == "__main__":
    # # Train the model
    # model = train_vmnist(
    #     epochs=10000,
    #     batch_size=32,  # Reduced from 128
    #     learning_rate=0.0005,  # Reduced from 0.001
    #     # input_size=3,
    #     n_steps=500,
    #     beta_start=0.0001,
    #     beta_end=0.02,
    #     n_layers=2,
    #     hidden_size=128,
    #     dropout=0.2,
    # )
    model = train_vmnist(
        epochs=10000,
        batch_size=32,  # Reduced from 128
        learning_rate=0.0001,  # Reduced from 0.001
        # input_size=3,
        n_steps=500,
        beta_start=0.0001,
        beta_end=0.02,
        n_layers=2,
        hidden_size=128,
        dropout=0.2,
        target_digit=6
    )
    # Save the final trained model
    print("\nSaving final model...")
    os.makedirs("ckpt", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
    }, os.path.join("ckpt", "final_model.pt"))

    # Final verification
    print("\nPerforming final verification...")
    verify_model(model, num_samples=5)