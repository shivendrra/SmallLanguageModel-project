import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

os.chdir('D:/Machine Learning/SLM-Project')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define CustomDataset
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'input': torch.FloatTensor(self.data.iloc[idx, :-1].values),
            'target': torch.FloatTensor([self.data.iloc[idx, -1]])
        }
        return sample


# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model=64, nhead=2, num_layers=2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.embedding(x)
        
        # Reshape x to have shape (sequence_length, batch_size, d_model)
        x = x.unsqueeze(0)  # Add a sequence dimension at the beginning
        
        # Apply transformer
        x = self.transformer(x, x)
        
        # Remove the sequence dimension
        x = x.squeeze(0)
        
        # Average over the sequence dimension
        x = torch.mean(x, dim=0)
        
        # Pass through fully connected layer
        x = self.fc(x)
        return x


# Initialize dataset and DataLoader
dataset = CustomDataset('Data/vector_data.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model
input_size = dataset[0]['input'].shape[0]
output_size = dataset[0]['target'].shape[0]
model = TransformerModel(input_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch['input'], batch['target']

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'transformer_model.pth')