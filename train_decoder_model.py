# -*- coding: utf-8 -*-
"""train_decoder_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RKqDi_rXHwdCtsytQhJhC8XkQFwlyt_c
"""

!pip install torch

import torch
print("GPU Available:", torch.cuda.is_available())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformer_decoder_model import TransformerDecoderModel, Config  # Import your model here

# Custom Dataset
import torch
from torch.utils.data import Dataset

class ChatDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.inputs = torch.tensor(data['input_ids'], dtype=torch.long)
        self.targets = torch.tensor(data['target_ids'], dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 3e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Data
dataset = ChatDataset('chat_data_sequences.npz')
train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize Model
config = Config()
model = TransformerDecoderModel(config).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training Loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"🎯 Epoch [{epoch+1}/{EPOCHS}] complete — Average Loss: {avg_loss:.4f}")

# Save Model
torch.save(model.state_dict(), "trained_decoder_model.pth")
print("Model training complete and saved as 'trained_decoder_model.pth'")