import pyvista as pv
import os
import torch

def get_four_random_numbers():
    """
    The function creates four random numbers that sum up to 1000.
    The last number is baised higher than other since it is the number of points in the fluid zone.
    """
    nums = torch.randint(0, 250, (3,))
    num = torch.tensor([1000]) - torch.sum(nums)
    nums = torch.cat((nums, num), dim=0)
    return nums


class DataReader():
    """
    A class to read data using PyVista.

    Attributes:
        file_path (str): Path to the input file.
        _dataset (pyvista.core.dataset): Cached PyVista dataset, loaded lazily.
    """

    def __init__(self, file_path):
        """
        Initializes the PyVistaDataReader with a given file path.

        Args:
            file_path (str): Path to the input file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        
        self.file_path = file_path
        self._dataset = None

    @property
    def dataset(self):
        """
        Lazily loads and returns the PyVista dataset.

        Returns:
            pyvista.core.dataset: Loaded PyVista dataset.
        """
        if self._dataset is None:
            try:
                self._dataset = pv.read(self.file_path)
            except Exception as e:
                raise RuntimeError(f"Failed to read file '{self.file_path}': {str(e)}")
        return self._dataset

    def fetch_data(self):
        """
        Retrieves points, pressure, and velocity data from the dataset.

        Returns:
            tuple: (points, pressure data, velocity data)
        """
        return self.dataset.points, self.dataset.point_data.get('p'), self.dataset.point_data.get('U')


### Class to read data from directory
class DirectoryReader():
    """
    This class reads data from a meshing_case directory.
    """
    def __init__(self, directory_path):
        self.directory_path = os.path.join(directory_path, 'VTK')
        self._readers = None
        files = os.listdir(self.directory_path)
        for file in files:
            if file.startswith('inlet'):
                self._inlet_dir = os.path.join(self.directory_path, file)
                self._inlet_file = os.listdir(self._inlet_dir)[-1]
            elif file.startswith('outlet'):
                self._outlet_dir = os.path.join(self.directory_path, file)
                self._outlet_file = os.listdir(self._outlet_dir)[-1]
            elif file.startswith('wall'):
                self._wall_dir = os.path.join(self.directory_path, file)
                self._wall_file = os.listdir(self._wall_dir)[-1]
            elif not file.endswith('_0.vtk') and file.endswith('.vtk'):
                self._fluid_zone_file = os.path.join(self.directory_path, file)
            elif file.endswith('_0.vtk'):
                continue
            else:
                raise RuntimeError(f"Unknown file found in directory '{self.directory_path}': {file}")

    @property
    def inlet(self):
        if self._inlet_file is None:
            raise RuntimeError(f"No inlet file found in directory '{self.directory_path}'")
        else:
            inlet = DataReader(os.path.join(self._inlet_dir, self._inlet_file))
            inlet_points, inlet_pressure, inlet_velocity = inlet.fetch_data()
        return inlet_points, inlet_pressure, inlet_velocity
    
    @property
    def outlet(self):
        if self._outlet_file is None:
            raise RuntimeError(f"No outlet file found in directory '{self.directory_path}'")
        else:
            outlet = DataReader(os.path.join(self._outlet_dir, self._outlet_file))
            outlet_points, outlet_pressure, outlet_velocity = outlet.fetch_data()
        return outlet_points, outlet_pressure, outlet_velocity
    
    @property
    def wall(self):
        if self._wall_file is None:
            raise RuntimeError(f"No wall file found in directory '{self.directory_path}'")
        else:
            wall = DataReader(os.path.join(self._wall_dir, self._wall_file))
            wall_points, wall_pressure, wall_velocity = wall.fetch_data()
        return wall_points, wall_pressure, wall_velocity
    
    @property
    def fluid_zone(self):
        if self._fluid_zone_file is None:
            raise RuntimeError(f"No fluid zone file found in directory '{self.directory_path}'")
        else:
            fluid_zone = DataReader(self._fluid_zone_file)
            fluid_zone_points, fluid_zone_pressure, fluid_zone_velocity = fluid_zone.fetch_data()
        return fluid_zone_points, fluid_zone_pressure, fluid_zone_velocity


from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class CFDDataset(Dataset):
    def __init__(self, meshing_cases_dir, meshing_cases_list, n_inlet = 500, n_outlet=500, n_wall = 2000, n_fluid = 5000, n_points = 1000):
        """
        Args:
            meshing_cases_dir (str): Path to the meshing_cases directory.
            meshing_cases_list (list): List of meshing cases.
            n_inlet (int): Number of inlet points to sample.
            n_outlet (int): Number of outlet points to sample.
            n_wall (int): Number of wall points to sample.
            n_fluid (int): Number of fluid zone points to sample.
        """
        self.meshing_cases_list = meshing_cases_list
        self.meshing_case_dir = meshing_cases_dir
        self.n_inlet = n_inlet
        self.n_outlet = n_outlet
        self.n_wall = n_wall
        self.n_fluid = n_fluid
        self.n_points = n_points

    def __len__(self):
        return len((self.meshing_cases_list))

    def __getitem__(self, idx):
        case = ((os.path.join(self.meshing_case_dir, self.meshing_cases_list[idx])))

        data = DirectoryReader(case)

        inlet_points, inlet_pressure, inlet_velocity = data.inlet
        outlet_points, outlet_pressure, outlet_velocity = data.outlet
        wall_points, wall_pressure, wall_velocity = data.wall
        fluid_zone_points, fluid_zone_pressure, fluid_zone_velocity = data.fluid_zone

        nums = get_four_random_numbers()

        inf_inlet_idxs = np.random.randint(0, inlet_points.shape[0], (nums[0],)) # The inlet points for inference.
        inf_outlet_idxs = np.random.randint(0, outlet_points.shape[0], (nums[1],)) # The outlet points for inference.
        inf_wall_idxs = np.random.randint(0, wall_points.shape[0], (nums[2],)) # The wall points for inference.
        inf_fluid_zone_idxs = np.random.randint(0, fluid_zone_points.shape[0], (nums[3],)) # The fluid zone points for inference.

        inf_inlet_points = torch.from_numpy(inlet_points[inf_inlet_idxs]).float()
        inf_inlet_pressures = torch.from_numpy(inlet_pressure[inf_inlet_idxs]).float()
        inf_inlet_velocities = torch.from_numpy(inlet_velocity[inf_inlet_idxs]).float()

        inf_outlet_points = torch.from_numpy(outlet_points[inf_outlet_idxs]).float()
        inf_outlet_pressures = torch.from_numpy(outlet_pressure[inf_outlet_idxs]).float()
        inf_outlet_velocities = torch.from_numpy(outlet_velocity[inf_outlet_idxs]).float()

        inf_wall_points = torch.from_numpy(wall_points[inf_wall_idxs]).float()
        inf_wall_pressures = torch.from_numpy(wall_pressure[inf_wall_idxs]).float()
        inf_wall_velocities = torch.from_numpy(wall_velocity[inf_wall_idxs]).float()

        inf_fluid_zone_points = torch.from_numpy(fluid_zone_points[inf_fluid_zone_idxs]).float()
        inf_fluid_zone_pressures = torch.from_numpy(fluid_zone_pressure[inf_fluid_zone_idxs]).float()
        inf_fluid_zone_velocities = torch.from_numpy(fluid_zone_velocity[inf_fluid_zone_idxs]).float()

        # Concatenate the inlet, outlet, and wall points to feed to the PointNet
        positions = torch.cat((inf_inlet_points, inf_outlet_points, inf_wall_points, inf_fluid_zone_points), dim=0)
        velocities = torch.cat((inf_inlet_velocities, inf_outlet_velocities, inf_wall_velocities, inf_fluid_zone_velocities), dim=0)
        pressures = torch.cat((inf_inlet_pressures, inf_outlet_pressures, inf_wall_pressures, inf_fluid_zone_pressures), dim=0)

        rand_inlet_idxs = np.random.randint(0, inlet_points.shape[0], (self.n_inlet)) # The inlet points for training.
        rand_outlet_idxs = np.random.randint(0, outlet_points.shape[0], (self.n_outlet)) # The outlet points for training.
        rand_wall_idxs = np.random.randint(0, wall_points.shape[0], (self.n_wall)) # The wall points for training.
        rand_fluid_zone_idxs = np.random.randint(0, fluid_zone_points.shape[0], (self.n_fluid)) # The fluid zone points for training.

        inlet_points = torch.from_numpy(inlet_points[rand_inlet_idxs]).float()
        inlet_pressures = torch.from_numpy(inlet_pressure[rand_inlet_idxs]).float()
        inlet_velocities = torch.from_numpy(inlet_velocity[rand_inlet_idxs]).float()

        outlet_points = torch.from_numpy(outlet_points[rand_outlet_idxs]).float()
        outlet_pressures = torch.from_numpy(outlet_pressure[rand_outlet_idxs]).float()
        outlet_velocities = torch.from_numpy(outlet_velocity[rand_outlet_idxs]).float()

        wall_points = torch.from_numpy(wall_points[rand_wall_idxs]).float()
        wall_pressures = torch.from_numpy(wall_pressure[rand_wall_idxs]).float()
        wall_velocities = torch.from_numpy(wall_velocity[rand_wall_idxs]).float()
        
        return {'input' : 
                    {"inlet": {"points": inlet_points}, 
                    "outlet" : {"points": outlet_points}, 
                    "wall" : {"points": wall_points}}, 
                'inf_input' : 
                    {"positions": positions,},
                'output' : 
                    {"pressure": pressures,
                    "velocity": velocities}
                    }