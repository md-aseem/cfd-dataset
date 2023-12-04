import torch
import torch.nn as nn
from model_2 import Model
from dataloader import CFDDataset
from torch.utils.data import DataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 6
D_FF = 1024
DROPOUT = 0.1
N_INLETS = 500
N_OUTLETS = 500
N_WALLS = 10000
N_FLUID = 10000
N_FEATURES = 128
N_POINTS = 1000
SPACE_VARIABLE = 3
SEQ_LEN = (1 + 1 + 1)*16 + 1 + N_POINTS 
N_EPOCHS = 10
BATCH_SIZE = 1
meshing_dir = '/home/aseem/OpenFOAM/aseem-11/run/training_data/meshing_cases'
train_cases_list = ['random_channel_70', 'random_channel_31', 'holed_cylinder_7',  'holed_cylinder_15', 'holed_cylinder_33']
INLET_VELOCITY = torch.tensor([1, 1, 1], dtype=torch.float).unsqueeze(0).unsqueeze(0).repeat(BATCH_SIZE, 1, 1).to(device)
model = Model()
model.to(device)

# Loss function
fn_loss = nn.MSELoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load dataset
dataset = CFDDataset(meshing_cases_dir=meshing_dir , meshing_cases_list=train_cases_list, n_inlet=N_INLETS, n_outlet=N_OUTLETS, n_wall=N_WALLS, n_fluid=N_FLUID, n_points=N_POINTS)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
loss_list = []

# Training loop
for epoch in range(N_EPOCHS):
    model.train()
    total_loss = 0
    total_percentage_error = 0

    for i, data in enumerate(dataloader):
        # Data to feed to the PointNet
        inlet_point_net = data['input']['inlet']['points'].to(device)
        outlet_point_net = data['input']['outlet']['points'].to(device)
        wall_point_net = data['input']['wall']['points'].to(device)

        # Data to concantenate to the PointNet output to feed to the Transformer Encoder
        positions = data['inf_input']['positions'].to(device)

        # Inlet velocity
        current_batch_size = inlet_point_net.shape[0]
        current_inlet_velocity = torch.tensor([1, 1, 1], dtype=torch.float).unsqueeze(0).unsqueeze(0).repeat(current_batch_size, 1, 1).to(device)

        # Target data
        pressure = data['output']['pressure'].to(device)
        velocity = data['output']['velocity'].to(device)
        u = velocity[:, :, 0]
        v = velocity[:, :, 1]
        w = velocity[:, :, 2]

        # Run the model
        pressure_pred, u_pred, v_pred, w_pred = model(inlet_point_net, outlet_point_net, wall_point_net, positions, current_inlet_velocity)

        # Calculate loss
        loss_pressure = fn_loss(pressure_pred, pressure)
        loss_u = fn_loss(u_pred, u)
        loss_v = fn_loss(v_pred, v)
        loss_w = fn_loss(w_pred, w)
        loss = loss_pressure + loss_u + loss_v + loss_w

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # Calculate mean percentage error for each prediction
        mae_pressure = torch.mean(torch.abs((pressure - pressure_pred))) * 100
        mae_u = torch.mean(torch.abs((u - u_pred))) * 100
        mae_v = torch.mean(torch.abs((v - v_pred))) * 100
        mae_w = torch.mean(torch.abs((w - w_pred))) * 100

        # Calculate mean of MPEs
        mean_absolute_error = (mae_pressure + mae_u + mae_v + mae_w) / 4

        # Accumulate loss
        total_loss += loss.item()
        total_percentage_error += mean_absolute_error.item()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{N_EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    # Compute average loss and MPE for the epoch
    avg_loss = total_loss / len(dataloader)
    avg_mpe = total_percentage_error / len(dataloader)
    loss_list.append(avg_loss)
    print(f'Epoch [{epoch+1}/{N_EPOCHS}] Average Loss: {avg_loss:.4f}, Average Mean Absolute Error: {avg_mpe:.2f}')
