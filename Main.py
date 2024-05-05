import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

class ScriptDataset(Dataset):
    def __init__(self, data_dir, script, transform=None):
        self.data_dir = data_dir
        self.script = script;
        self.transform = transform
        self.samples = self.load_data()
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set([sample[1] for sample in self.samples])))}

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
        # Convert script (label) to tensor
        label_idx = self.label_to_idx[script]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        return image, label_tensor

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


# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create a combined dataset for both scripts
combined_dataset = torch.utils.data.ConcatDataset([old_hungarian_dataset, old_turkic_dataset])

# Calculate the lengths for splitting the combined dataset
total_length = len(combined_dataset)
train_length = int(0.8 * total_length)
val_length = total_length - train_length

# Define the train and validation datasets for the combined dataset
train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_length, val_length])

# Create data loaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Update the number of classes based on the combined dataset
num_classes = len(set(sample[1] for sample in combined_dataset))
model = SimpleCNN(num_classes).to(device)  # Assuming you have a 'device' variable for GPU/CPU

# Define the optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Evaluate the model on the validation set
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_loss += criterion(outputs, labels).item()

    val_accuracy = 100 * val_correct / val_total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {val_accuracy:.2f}%")