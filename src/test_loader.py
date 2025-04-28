import torch
from loader import IAMDataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

def test_dataloader():
    # Initialize the dataset
    dataset = IAMDataset(
        img_path='./data/lineImages',
        xml_path='./data/original',
        line_path='./data/lineStrokes'
    )
    
    # Get the first sample
    sample = dataset[0]
    
    # Get the first image and text from the sample
    img = sample['images'][0]  # Get first image from the list
    text = sample['texts'][0]  # Get first text from the list
    
    # Display the image directly
    plt.imshow(img)
    plt.title(f"Text: {text}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    test_dataloader() 