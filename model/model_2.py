import torch
import torch.nn as nn
from transformer_2 import CustomTransformerEncoder  
from PointNet import PointNet
from typing import Optional, Tuple

# Hyperparameters
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 4
D_FF = 1024
DROPOUT = 0.1
N_INLETS = 500
N_OUTLETS = 500
N_WALLS = 2000
N_FEATURES = 128
N_POINTS = 1000
SPACE_VARIABLE = 3
SEQ_LEN = 1 + 1 + 1 + 1 + N_POINTS 

class Model(nn.Module):
    """ 
    Model combining PointNet and Transformer Encoder.
    
    PointNet is used for feature extraction from point clouds representing inlets, outlets, and walls.
    Transformer Encoder is used for sequence processing of the spatial data.
    """
    
    def __init__(self):
        super(Model, self).__init__()
        
        self.pos_linear = nn.Linear(3, D_MODEL)
        self.inlet_vel_linear = nn.Linear(3, D_MODEL)

        # PointNet layers for different features
        self.inlet_point_net = PointNet(point_numbers=N_INLETS, space_variable=SPACE_VARIABLE, n_features=N_FEATURES, scaling=1)
        self.outlet_point_net = PointNet(point_numbers=N_OUTLETS, space_variable=SPACE_VARIABLE, n_features=N_FEATURES, scaling=1)
        self.wall_point_net = PointNet(point_numbers=N_WALLS, space_variable=SPACE_VARIABLE, n_features=N_FEATURES, scaling=1)

        # Transformer encoder layer
        self.transformer = CustomTransformerEncoder(long_seq_length=49, num_short_seqs=N_POINTS, d_model=D_MODEL, n_head=NUM_HEADS, num_layers=NUM_LAYERS, d_ff=D_FF, dropout=DROPOUT)
        
        # Final fully connected layer
        self.fc1 = nn.Linear(D_MODEL, 1024)
        self.fc2 = nn.Linear(1024, 4)
        

    def forward(self, 
                inlet_points: torch.Tensor, 
                outlet_points: torch.Tensor, 
                wall_points: torch.Tensor, 
                positions: torch.Tensor, 
                inlet_velocity: torch.Tensor):
        """
        Forward pass of the model.
        
        Parameters:
        - inlet_points: Point cloud for the inlets
        - outlet_points: Point cloud for the outlets
        - wall_points: Point cloud for the walls
        - positions: Spatial positions
        - inlet_velocity: Boundary condition for inlet velocity
        
        Returns:
        - p: Predicted pressure
        - u, v, w: Predicted velocity components
        """
    

        # Reshape and tile positions and inlet_velocity
        #positions = positions.view(positions.size(0), INF_POINTS * SPACE_VARIABLE, 1).expand(-1, -1, D_MODEL)
        #inlet_velocity = inlet_velocity.view(inlet_velocity.size(0), SPACE_VARIABLE, 1).expand(-1, -1, D_MODEL)
        
        # Pass through the linear layers
        positions = self.pos_linear(positions)
        inlet_velocity = self.inlet_vel_linear(inlet_velocity)

        # Generate features using PointNet
        inlet_features = self.inlet_point_net(inlet_points)
        outlet_features = self.outlet_point_net(outlet_points)
        wall_features = self.wall_point_net(wall_points)
        
        # Concatenate all features
        x = torch.cat((inlet_features, outlet_features, wall_features, positions, inlet_velocity), dim=1)
        
        # Pass through the transformer encoder
        x = self.transformer(x)
        
        # Removing the first four elements from the sequence
        x = x[:, 49:, :]

        # Pass through the final fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Extract individual components
        p, u, v, w = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]
        
        return p, u, v, w
    