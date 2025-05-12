import os
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
from models.diffusion import point_cloud_ddpm_model, BiRNN_denoiser
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
from datetime import datetime

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

class VectorMNISTDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.labels = []
        
        # Convert to absolute path if relative
        data_dir = os.path.abspath(data_dir)
        print(f"Loading data from: {data_dir}")
        
        # Track min/max values for normalization
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        # Load data from each digit directory (0-9)
        for digit in range(5,6):
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
                        pen_states = np.array(drawing[0][2])
                        
                        # Update min/max values
                        min_x = min(min_x, np.min(x_coords))
                        min_y = min(min_y, np.min(y_coords))
                        max_x = max(max_x, np.max(x_coords))
                        max_y = max(max_y, np.max(y_coords))
                        
                        # Stack the coordinates and pen states (normalization happens later)
                        formatted_data = np.stack([x_coords, y_coords, pen_states], axis=1)
                        self.data.append(formatted_data)
                        self.labels.append(digit)
        
        if len(self.data) == 0:
            raise ValueError(f"No data found in {data_dir}. Please ensure the data directory contains subdirectories 0-9 with .pkl files.")
            
        # Normalize coordinates to [-1, 1] range for all sequences
        for i in range(len(self.data)):
            # Only normalize x,y coordinates, not pen states
            self.data[i][:, 0] = 2 * (self.data[i][:, 0] - min_x) / (max_x - min_x) - 1
            self.data[i][:, 1] = 2 * (self.data[i][:, 1] - min_y) / (max_y - min_y) - 1
            
        # Convert to tensors
        self.data = [torch.FloatTensor(d) for d in self.data]
        self.labels = torch.LongTensor(self.labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

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
        feature_dim = 3   # x, y, pen_state
        
        # Create random noise
        noise = torch.randn(batch_size, seq_length, feature_dim, device=device)
        
        # Generate samples
        samples = model.sample(noise)
        
        # Move to CPU for visualization
        samples = samples.cpu().numpy()
        
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
            
            # Plot the stroke
            ax = axes[i]
            # Plot points where pen is down
            mask = pen_states > 0.5
            ax.plot(x[mask], y[mask], 'b-', linewidth=2)
            ax.set_title(f'Sample {i+1}')
            ax.axis('equal')
            ax.grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(results_dir, f'visualization{epoch_str}_{timestamp}.png')
        plt.savefig(plot_file)
        plt.close()

def train_vmnist(epochs: int, batch_size: int, learning_rate: float, n_steps: int, beta_start: float, beta_end: float, n_layers: int, hidden_size: int, dropout: float):
    # Initialize dataset and dataloader
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    dataset = VectorMNISTDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    denoiser = BiRNN_denoiser(n_layers, hidden_size, dropout)
    diffusion_model = point_cloud_ddpm_model(
        model=denoiser,
        n_steps=n_steps,
        beta_start=beta_start,
        beta_end=beta_end,
    ).to(device)
    
    optimizer = Adam(diffusion_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        diffusion_model.train()
        total_loss = 0
        
        for batch_idx, (data, _, lengths) in enumerate(dataloader):
            data = data.to(device)
            lengths = lengths.to(device)
            batch_size = data.size(0)
            
            # Sample random timesteps
            t = torch.randint(0, n_steps, (batch_size,), device=device)
            
            # Add noise to the data
            noisy_data = diffusion_model.ddpm_forward(data, t)
            
            # Predict noise
            pred_noise = diffusion_model.ddpm_backward(noisy_data, t, lengths)
            
            # Calculate loss
            loss = criterion(pred_noise, data)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            #if batch_idx % 100 == 0:
                #print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
        
        # Verify model more frequently to monitor progress
        if epoch % 10 == 0:  # Verify every 10 epochs
            print(f"\nVerifying model after epoch {epoch}")
            verify_model(diffusion_model, num_samples=5, device=device, epoch=epoch)
    
    return diffusion_model

# Train the model with improved hyperparameters:
# - Reduced diffusion steps (500) for faster training while maintaining quality
# - Increased model capacity (4 layers, 256 hidden size) for better expressiveness
# - Reduced dropout (0.1) to prevent underfitting
# - Lower learning rate (0.0005) for more stable training
model = train_vmnist(
    epochs=1000,         # Keep epochs the same
    batch_size=64,       # Keep batch size the same
    learning_rate=0.0005,# Lower learning rate for stability
    n_steps=500,         # Fewer diffusion steps
    beta_start=0.0001,   # Keep beta schedule the same
    beta_end=0.02,
    n_layers=4,          # More layers for capacity
    hidden_size=256,     # Larger hidden size for capacity
    dropout=0.1          # Lower dropout
)

# Final verification
print("\nPerforming final verification...")
verify_model(model, num_samples=5)