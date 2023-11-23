# %%
import dataloader as dl

# Load the test data
test_data = dl.DirectoryReader('/home/aseem/OpenFOAM/aseem-11/run/training_data/meshing_cases/random_channel_7')

# %%

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np

class CFDDataset(Dataset):
    def __init__(self, meshing_cases_dir, meshing_cases_list, in_out_num = 500, wall_num = 2000, fluid_num = 5000, transform=None):
        """
        Args:
            data_list (list): List of your data.
            labels (list): Corresponding labels for your data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.meshing_cases_list = meshing_cases_list
        self.meshing_case_dir = meshing_cases_dir
        self.in_out_num = in_out_num
        self.wall_num = wall_num
        self.fluid_num = fluid_num

    def __len__(self):
        return len((self.meshing_cases_list))

    def __getitem__(self, idx):
        case = ((os.path.join(self.meshing_case_dir, self.meshing_cases_list[idx])))
        print(case)
        data = dl.DirectoryReader(case)
        
        inlet_points, inlet_pressure, inlet_velocity = data.inlet
        outlet_points, outlet_pressure, outlet_velocity = data.outlet
        wall_points, wall_pressure, wall_velocity = data.wall
        fluid_zone_points, fluid_zone_pressure, fluid_zone_velocity = data.fluid_zone


        rand_inlet_idxs = np.random.randint(0, inlet_points.shape[0], (self.in_out_num))
        rand_outlet_idxs = np.random.randint(0, outlet_points.shape[0], (self.in_out_num))
        rand_wall_idxs = np.random.randint(0, wall_points.shape[0], (self.wall_num))
        rand_fluid_zone_idxs = np.random.randint(0, fluid_zone_points.shape[0], (self.fluid_num))

        inlet_points = torch.from_numpy(inlet_points[rand_inlet_idxs]).float()
        inlet_pressures = torch.from_numpy(inlet_pressure[rand_inlet_idxs]).float()
        inlet_velocities = torch.from_numpy(inlet_velocity[rand_inlet_idxs]).float()

        outlet_points = torch.from_numpy(outlet_points[rand_outlet_idxs]).float()
        outlet_pressures = torch.from_numpy(outlet_pressure[rand_outlet_idxs]).float()
        outlet_velocities = torch.from_numpy(outlet_velocity[rand_outlet_idxs]).float()

        wall_points = torch.from_numpy(wall_points[rand_wall_idxs]).float()
        wall_pressures = torch.from_numpy(wall_pressure[rand_wall_idxs]).float()
        wall_velocities = torch.from_numpy(wall_velocity[rand_wall_idxs]).float()

        fluid_zone_points = torch.from_numpy(fluid_zone_points[rand_fluid_zone_idxs]).float()
        fluid_zone_pressures = torch.from_numpy(fluid_zone_pressure[rand_fluid_zone_idxs]).float()
        fluid_zone_velocities = torch.from_numpy(fluid_zone_velocity[rand_fluid_zone_idxs]).float()


        rand_inlet_idx = np.random.randint(0, inlet_points.shape[0], (1))
        rand_outlet_idx = np.random.randint(0, outlet_points.shape[0], (1))
        rand_wall_idx = np.random.randint(0, wall_points.shape[0], (1))
        rand_fluid_zone_idx = np.random.randint(0, fluid_zone_points.shape[0], (1))

        inlet_point = (inlet_points[rand_inlet_idx])
        inlet_pressure = (inlet_pressure[rand_inlet_idx])
        inlet_velocity = (inlet_velocity[rand_inlet_idx])

        outlet_point = (outlet_points[rand_outlet_idx])
        outlet_pressure = (outlet_pressure[rand_outlet_idx])
        outlet_velocity = (outlet_velocity[rand_outlet_idx])

        wall_point = (wall_points[rand_wall_idx])
        wall_pressure = (wall_pressure[rand_wall_idx])
        wall_velocity = (wall_velocity[rand_wall_idx])

        fluid_zone_point = (fluid_zone_points[rand_fluid_zone_idx])
        fluid_zone_pressure = (fluid_zone_pressure[rand_fluid_zone_idx])
        fluid_zone_velocity = (fluid_zone_velocity[rand_fluid_zone_idx])

        return inlet_points, inlet_point, inlet_pressures, inlet_pressure, inlet_velocities, inlet_velocity, outlet_points, outlet_point, outlet_pressures, outlet_pressure, outlet_velocities, outlet_velocity, wall_points, wall_point, wall_pressures, wall_pressure, wall_velocities, wall_velocity, fluid_zone_points, fluid_zone_point, fluid_zone_pressures, fluid_zone_pressure, fluid_zone_velocities, fluid_zone_velocity
        
# %%

# Example usage
meshing_dir = '/home/aseem/OpenFOAM/aseem-11/run/training_data/meshing_cases'
meshing_cases_list = ['holed_cylinder_5', 'holed_cylinder_6', 'holed_cylinder_8']
# Create a dataset object
dataset = CFDDataset(meshing_cases_dir=meshing_dir , meshing_cases_list=meshing_cases_list, in_out_num=500, wall_num=2000, fluid_num=5000)

# Create a dataloader
dataloader = DataLoader(dataset, batch_size=30, shuffle=True)

# Iterate through the batches

for i, data in enumerate(dataloader):

    inlet_points, inlet_point, inlet_pressures, inlet_pressure, inlet_velocities, inlet_velocity, outlet_points, outlet_point, outlet_pressures, outlet_pressure, outlet_velocities, outlet_velocity, wall_points, wall_point, wall_pressures, wall_pressure, wall_velocities, wall_velocity, fluid_zone_points, fluid_zone_point, fluid_zone_pressures, fluid_zone_pressure, fluid_zone_velocities, fluid_zone_velocity = data
    
    print(inlet_points.shape, inlet_point.shape, inlet_pressures.shape, inlet_pressure.shape, inlet_velocities.shape, inlet_velocity.shape, outlet_points.shape, outlet_point.shape, outlet_pressures.shape, outlet_pressure.shape, outlet_velocities.shape, outlet_velocity.shape, wall_points.shape, wall_point.shape, wall_pressures.shape, wall_pressure.shape, wall_velocities.shape, wall_velocity.shape, fluid_zone_points.shape, fluid_zone_point.shape, fluid_zone_pressures.shape, fluid_zone_pressure.shape, fluid_zone_velocities.shape, fluid_zone_velocity.shape) 
    break

