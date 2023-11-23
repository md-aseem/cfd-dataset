import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformer import TransformerEncoder
from PointNet import PointNet

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setting the hyperparameters
n_features = 128

# Loading Data
Data = np.load('Data.npy')
data_number = Data.shape[0]

print('Number of data is:')
print(data_number)

point_numbers = 1024
space_variable = 2
cfd_variable = 3

input_data = np.zeros([data_number, point_numbers, space_variable], dtype='f')
output_data = np.zeros([data_number, point_numbers, cfd_variable], dtype='f')

for i in range(data_number):
    input_data[i, :, 0] = Data[i, :, 0]
    input_data[i, :, 1] = Data[i, :, 1]
    output_data[i, :, 0] = Data[i, :, 3]
    output_data[i, :, 1] = Data[i, :, 4]
    output_data[i, :, 2] = Data[i, :, 2]

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, input_data, output_data, transform=None):
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        input = self.input_data[idx]
        output = self.output_data[idx]

        return input, output

# Move data to GPU
input_data = torch.tensor(input_data, dtype=torch.float32).to(device)
output_data = torch.tensor(output_data, dtype=torch.float32).to(device)

# PointNet to learn geometry features (B, P, D) -> (B, P, F)
point_net = PointNet(point_numbers=point_numbers, space_variable=2, n_features=n_features).to(device)

# Transformer to input those features and use attention (B, P, F) -> (B, P, F)
transformer = TransformerEncoder(num_layers=4, d_model=n_features, num_heads=8, d_ff=64, dropout=0.0).to(device)

# Linear Layer to convert the transformer output to u_x, u_y, p
LinearLayer = nn.Linear(n_features, 3, bias=False).to(device)

# Loss function
criterion = nn.MSELoss()

# Parameters to be optimized
parameters = list(point_net.parameters()) + list(transformer.parameters()) + list(LinearLayer.parameters())
total_parameters = sum(p.numel() for p in parameters)
print('Number of parameters to be optimized:', total_parameters)

# Optimizer
optimizer = torch.optim.Adam(parameters, lr=0.001)

# Create an instance of the custom dataset
custom_dataset = CustomDataset(input_data, output_data)

batch_size = 32
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Number of epochs
epochs = 1000
loss_list = []

# Set model to training mode
point_net.train()
transformer.train()
LinearLayer.train()

# Training loop
for epoch in range(epochs):
    for i, (input_data, output_data) in enumerate(data_loader):
        # Move data to GPU
        input_data = input_data.to(device)
        output_data = output_data.to(device)
        print(input_data.shape)
        print(output_data.shape)

        # Forward pass
        point_net_out = point_net(input_data)
        transformer_output = transformer(point_net_out)
        final_output = LinearLayer(transformer_output)

        # Compute loss
        loss = criterion(final_output, output_data)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss
    print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))
    loss_list.append(loss.item())

# Plot the loss
import matplotlib.pyplot as plt
plt.plot(loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()