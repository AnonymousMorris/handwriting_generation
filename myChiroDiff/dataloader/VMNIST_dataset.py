import os
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
from models.diffusion import point_cloud_ddpm_model, BiRNN_denoiser
import torch
from tqdm import tqdm
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
                        pen_states = np.array(drawing[0][2])
                        
                        # Stack the coordinates and pen states
                        formatted_data = np.stack([x_coords, y_coords, pen_states], axis=1)
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

