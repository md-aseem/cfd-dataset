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
N_EPOCHS = 1
BATCH_SIZE = 1
meshing_dir = '/home/aseem/OpenFOAM/aseem-11/run/training_data/meshing_cases'
train_cases_list = ['holed_cylinder_5', 'holed_cylinder_6', 'holed_cylinder_8', 'holed_cylinder_9', 'holed_cylinder_10', 'holed_cylinder_11', 'holed_cylinder_12', 'random_channel_5', 'random_channel_19', 'holed_cylinder_23', 'random_channel_45', 'random_channel_78']
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
        
        print(u[0])
        