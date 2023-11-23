# %%
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformer import TransformerEncoder
from PointNet import PointNet

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_features = 128
n_inlets = 500
n_outlets = 500
n_walls = 2000
inf_points = 100
space_variable = 3
seq_len = n_inlets + n_outlets + n_walls + inf_points*space_variable + space_variable

inlet_point_net = PointNet(point_numbers=n_inlets, space_variable=space_variable, n_features=n_features, scaling=1).to(device)
outlet_point_net = PointNet(point_numbers=n_outlets, space_variable=space_variable, n_features=n_features, scaling=1).to(device)
wall_point_net = PointNet(point_numbers=n_walls, space_variable=space_variable, n_features=n_features, scaling=1).to(device)

# %%
U = torch.tensor([1, 0, 1])
U = U.view(1, 3, 1).repeat(1, 1, 128).to(device)
P = torch.ones(inf_points, space_variable)

print(P.shape)
P = P.view(1, inf_points*space_variable, 1).repeat(1, 1, 128).to(device)
print(P.shape)
# %%
inlet_data = torch.rand(1, 500, 3).to(device)
outlet_data = torch.rand(1, 500, 3).to(device)
wall_data = torch.rand(1, 2000, 3).to(device)
fluid_zone_data = torch.rand(1, 5000, 3).to(device)

# %%
transformer_input = torch.cat((inlet_point_net(inlet_data), outlet_point_net(outlet_data), wall_point_net(wall_data), U, P), dim=1)

# %%

transformer = TransformerEncoder(seq_len= seq_len, d_model=n_features, num_heads=8, num_layers=6, d_ff=1024, dropout=0.1).to(device)

print(transformer(transformer_input).shape)
# %%

# %%

print((transformer(transformer_input)).shape)

class Model(nn.Module):
    def __init__(self, space_variable=3, d_model=128, num_heads=8, num_layers=6, d_ff=1024, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.inf_points = inf_points
        self.space_variable = space_variable
        self.seq_len = n_inlets + n_outlets + n_walls + inf_points*space_variable + space_variable
        
        self.dropout = nn.Dropout(p=dropout)
        self.inlet_point_net = PointNet(point_numbers=n_inlets, space_variable=space_variable, n_features=n_features, scaling=1)
        self.outlet_point_net = PointNet(point_numbers=n_outlets, space_variable=space_variable, n_features=n_features, scaling=1)
        self.wall_point_net = PointNet(point_numbers=n_walls, space_variable=space_variable, n_features=n_features, scaling=1)

        self.transformer = TransformerEncoder(seq_len= seq_len, d_model=d_model, num_heads=num_heads, num_layers=num_layers, d_ff=1024, dropout=0.1)
        self.fc = nn.Linear(d_model, 4)

    def forward(self, inlet_points, outlet_points, wall_points, positions, pressure, velocity):
        
        positions = positions.view(1, self.inf_points*self.space_variable, 1).repeat(1, 1, 128)

        inlet_features = self.inlet_point_net(inlet_points)
        outlet_features = self.outlet_point_net(outlet_points)
        wall_features = self.wall_point_net(wall_points)

        x = torch.cat((inlet_features, outlet_features, wall_features, positions), dim=1)
        x = self.transformer(x)
        x = self.fc(x)

        ### x = [p, u, v, w]
        p = x[:, 0]
        u = x[:, 1]
        v = x[:, 2]
        w = x[:, 3]

        ### Calculate loss if training(pressure and velocity are not None)
        if pressure is not None and velocity is not None:
            loss = torch.mean((p - pressure)**2 + (u - velocity[:, 0])**2 + (v - velocity[:, 1])**2 + (w - velocity[:, 2])**2)
            return p, u, v, w, loss
        else:
            return p, u, v, w