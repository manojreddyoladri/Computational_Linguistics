import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cairosvg

class ScriptDataset(Dataset):
    def __init__(self, data_dir, script, transform=None):
        self.data_dir = data_dir
        self.script = script
        self.transform = transform
        self.samples = self.load_data()

    def load_data(self):
        samples = []
        
        script_dir = os.path.join(self.data_dir, self.script)
        
        for filename in os.listdir(script_dir):
            image_path = os.path.join(script_dir, filename)
            if os.path.isfile(image_path):
              samples.append((image_path, self.script))
            
            
            
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, script = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, script

# Data directories
data_dir = 'Datasets'

# Define transformations for data preprocessing
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Create datasets for each script
old_hungarian_dataset = ScriptDataset(data_dir, 'old_hungarian', transform=data_transform)
old_turkic_dataset = ScriptDataset(data_dir, 'old_turkic', transform=data_transform)
#phoenician_dataset = ScriptDataset(os.path.join(data_dir, 'phoenician'), transform=data_transform)

# Example usage: Load data using DataLoader
batch_size = 32
old_hungarian_loader = DataLoader(old_hungarian_dataset, batch_size=batch_size, shuffle=True)
old_turkic_loader = DataLoader(old_turkic_dataset, batch_size=batch_size, shuffle=True)
#phoenician_loader = DataLoader(phoenician_dataset, batch_size=batch_size, shuffle=True)

# Iterate over data loaders
for images, scripts in old_hungarian_loader:
    # Perform operations with the batched data
    pass
