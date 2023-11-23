
import concurrent.futures
from multiprocessing import Process
import os
import numpy as np
import gmsh
from threading import Thread
from utils import generate_positions

n_geo = 1 # number of geometries to create
mesh_dir = "./meshes" # directory to save the meshes
mesh_size = 0.06 # decrease to increase the mesh resolution
inlet_marker, outlet_marker, wall_marker = 1, 2, 3 # markers for the inlet, outlet and walls

def mesh_generate_worker():
    gmsh.model.mesh.generate(3)

def get_start_number(name): # Function to get the last 
    meshes = os.listdir(mesh_dir)    
    files = [file for file in meshes if file.startswith(name)]
    if files != []:
        numbers = [int(file.split("_")[-1].split(".")[0]) for file in files]
        start_num = max(numbers)
    else:
        start_num = 0
    return start_num

def generate_mesh(geo, file_name):
    set_mesh_options()
    gmsh.model.mesh.generate(3)
    gmsh.write(os.path.join(mesh_dir, f'{file_name}.msh'))
    gmsh.finalize()


## Create mesh_dir if it does not exist
if not os.path.exists(mesh_dir):
    os.makedirs(mesh_dir)

def set_mesh_options():
    gmsh.option.setNumber("Mesh.Format", 1) 
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) 
    gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.3)
    gmsh.option.setNumber("Mesh.SaveAll", 0) 

def create_cubes(n_geo, start_num=0):
    start_num = get_start_number("cube_")
    for i in range(n_geo):
        gmsh.initialize()
        name = "cube_" + str(i+start_num)
        gmsh.model.add(name)

        scale = np.random.uniform(.01, 1, 3)
        # Create a new box
        box = gmsh.model.occ.addBox(0, 0, 0, scale[0], scale[1], scale[2])
        gmsh.model.occ.synchronize() 
        # Getting surfaces
        surfaces = gmsh.model.occ.getEntities(2) # get all the entities of dimension 3 (all the surfaces)
        
        walls = []
        for surface in surfaces:
            gmsh.model.occ.synchronize() 
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            print(com)
            if com[0] == 0:
                gmsh.model.add_physical_group(2, [surface[1]], inlet_marker, name= 'inlet') # add inlet
            elif com[0] == scale[0]:
                gmsh.model.add_physical_group(2, [surface[1]], outlet_marker, name= 'outlet')
            else:
                walls.append(surface[1])

        gmsh.model.add_physical_group(2, walls, wall_marker, name= 'walls') # add walls
        gmsh.model.add_physical_group(3, [box], 1, name="FluidZone") # add a physical group of dimension 3 (volume) with tag 1
        gmsh.model.occ.synchronize() 
        set_mesh_options()
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size/(scale.mean())/3)
        gmsh.model.mesh.generate(3) # generate 3D mesh

        # Write mesh file
        gmsh.write(os.path.join(mesh_dir, name+'.msh' ))
        gmsh.finalize()


  # %%                 
def create_var_cross_section_cylinders(n_geo):
    start_num = get_start_number("variable_cross_section_cylinder_") + 1
    for i in range(n_geo):
        gmsh.initialize()
        name = "variable_cross_section_cylinder_" + str(i+start_num)
        gmsh.model.add(name)
        rand_scale_1 = np.random.randint(2, 20) / 50
        rand_scale_2 = np.random.randint(2, 20) / 50
        rand_scale_3 = np.random.randint(2, 20) / 50
        rand_dis_1 = np.random.randint(1, 5) / 10
        rand_dis_2 = np.random.randint(5, 10) / 10
        circle_1 = gmsh.model.occ.add_circle(0, 0, 0, rand_scale_1)
        wire_1 = gmsh.model.occ.add_wire([circle_1])
        circle_2 = gmsh.model.occ.add_circle(0, 0, rand_dis_1, rand_scale_2)
        wire_2 = gmsh.model.occ.add_wire([circle_2])
        circle_3 = gmsh.model.occ.add_circle(0, 0, rand_dis_2, rand_scale_3)
        wire_3 = gmsh.model.occ.add_wire([circle_3])     

        # Creating a loft between the three circles
        gmsh.model.occ.addThruSections([wire_1, wire_2, wire_3], makeSolid=True, makeRuled=False, smoothing=True)
        
        # Naming the surfaces
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.getEntities(2)
        print(surfaces)
        
        gmsh.model.add_physical_group(2, [surfaces[2][1]], inlet_marker, name= 'inlet') # add inlet
        gmsh.model.add_physical_group(2, [surfaces[1][1]], outlet_marker, name= 'outlet') # add outlet
        gmsh.model.add_physical_group(2, [surfaces[0][1]], wall_marker, name= 'walls') # add walls
        gmsh.model.add_physical_group(3, [1], 1, name="FluidZone")

        # Generating the mesh
        gmsh.model.occ.synchronize()
        set_mesh_options()
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size*1)
        gmsh.model.mesh.generate(3)
        gmsh.write(os.path.join(mesh_dir, name+'.msh' ))
        gmsh.finalize()


# %%
def create_bent_cylinders(n_geo):
    start_num = get_start_number("bent_cylinder_")
    for i in range(n_geo):
        gmsh.initialize()
        name = "bent_cylinder_" + str(i+start_num + 1)
        gmsh.model.add(name)
        rand_dis_1 = np.random.uniform(1, int(5)) / 10
        rand_dis_2 = np.random.uniform(5, int(10)) / 10
        rand_scale_1 = np.random.uniform(1, 5) / 5
        rand_scale_2 = np.random.uniform(1, 5) / 10
        rand_scale_3 = np.random.uniform(1, 5) / 5
        rand_angle_1 = np.random.rand(3)/3
        rand_angle_1[2] = 1 - rand_angle_1[0] - rand_angle_1[1]
        rand_angle_2 = np.random.rand(3)/3
        rand_angle_2[2] = 1 - rand_angle_2[0] - rand_angle_2[1]
        rand_angle_3 = np.random.rand(3)/3
        rand_angle_3[2] = 1 - rand_angle_3[0] - rand_angle_3[1]
        rand_pos_1 = np.random.rand(3)
        rand_pos_1 = np.random.rand(3)/8 + rand_scale_1
        rand_pos_2 = np.random.rand(3)/8 + rand_pos_1 +  rand_scale_2
        rand_pos_3 = np.random.rand(3)/8 + + rand_pos_2 + rand_scale_3

        circle_1 = gmsh.model.occ.add_circle(0+rand_pos_1[0], 0+rand_pos_1[1], 0 + rand_pos_1[2], rand_scale_1, )
        wire_1 = gmsh.model.occ.add_wire([circle_1])
        circle_2 = gmsh.model.occ.add_circle(0+rand_pos_2[0], 0+rand_pos_2[1], rand_dis_1+rand_pos_2[2], rand_scale_2, )
        wire_2 = gmsh.model.occ.add_wire([circle_2])
        circle_3 = gmsh.model.occ.add_circle(0+rand_pos_3[0], 0 + rand_pos_3[1], rand_dis_2+ rand_pos_3[2], rand_scale_3, )
        wire_3 = gmsh.model.occ.add_wire([circle_3])

        # Creating a loft between the three circles
        gmsh.model.occ.addThruSections([wire_1, wire_2, wire_3], makeSolid=True, makeRuled=True, smoothing=True)

        # Naming the surfaces
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.getEntities(2)

        gmsh.model.add_physical_group(2, [surfaces[2][1]], inlet_marker, name= 'inlet') # add inlet
        gmsh.model.add_physical_group(2, [surfaces[3][1]], outlet_marker, name= 'outlet') # add outlet
        gmsh.model.add_physical_group(2, [surfaces[0][1], surfaces[1][1]], wall_marker, name= 'walls') # add walls
        gmsh.model.add_physical_group(3, [1], 1, name="FluidZone")


        # Generating the mesh
        gmsh.model.occ.synchronize()
        set_mesh_options()
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size)
        gmsh.model.mesh.generate(3)

        # Write mesh file
        gmsh.write(os.path.join(mesh_dir, name+'.msh' ))
        gmsh.finalize()


# %%

def create_random_channel(n_geo, n_circles):
    start_num = get_start_number("random_channel_")
    print("Starting from {start_num}".format(start_num=start_num))
    for j in range(n_geo):

        gmsh.initialize()
        name = "random_channel_" + str(j + int(start_num)  + 1)
        gmsh.model.add(name)
        rand_radii = np.random.uniform(.1, .5, n_circles)
        rand_dis = np.random.uniform(.1, .8, [n_circles, 3])
        rand_dis[:, 2] = np.zeros(n_circles) # Zeroing the Z-axis so that geometry is in x-y plane.
        rand_dis = np.cumsum(rand_dis, axis=0)
        rand_pos = rand_dis + np.tile(rand_radii, 3).reshape([n_circles, 3])
        #rand_angle = np.random.uniform(-.5, .5, [n_circles, 3])
        
        circle_tags = []
        wire_tags = []
        for i in range(n_circles):
            
            circle_tag = gmsh.model.occ.add_circle(rand_pos[i][0], rand_pos[i][1], rand_pos[i][2], rand_radii[i], zAxis=rand_pos[i])
            wire_tag = gmsh.model.occ.add_wire([circle_tag])
            circle_tags.append(circle_tag)
            wire_tags.append(wire_tag)
                    
        gmsh.model.occ.synchronize()
        gmsh.model.occ.addThruSections(wire_tags, makeSolid=True, makeRuled=True, smoothing=True)
        
        # Naming the surfaces
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.getEntities(2)

        gmsh.model.add_physical_group(2, [surfaces[-2][1]], inlet_marker, name="inlet")
        gmsh.model.add_physical_group(2, [surfaces[-1][1]], outlet_marker, name="outlet")
        gmsh.model.add_physical_group(2, [x[1] for x in surfaces[:-2]], wall_marker, name="walls")
        gmsh.model.add_physical_group(3, [1], 1, name="FluidZone")

        # Generating the mesh
        gmsh.model.occ.synchronize()
        set_mesh_options()
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size/(rand_radii.mean()*8))
        gmsh.model.mesh.generate(3)

        # Write mesh file
        gmsh.write(os.path.join(mesh_dir, name+'.msh' ))
        gmsh.finalize()

# %%

import numpy as np
import gmsh

def create_random_channel_2(n_geo, n_circles):
    start_num = get_start_number("random_channel_")
    print("Starting from {start_num}".format(start_num=start_num))
    for geo_index in range(n_geo):

        gmsh.initialize()
        name = "random_channel_" + str(geo_index + int(start_num) + 1)
        print("Creating {name} now".format(name=name))
        gmsh.model.add(name)

        pos, rad = generate_positions(n_circles, max_angle=120, zero_dim='x')
        rad = rad * 2

        circle_tags = []
        wire_tags = []
        for i in range(n_circles):
            
            circle_tag = gmsh.model.occ.add_circle(pos[i][0], pos[i][1], pos[i][2], rad[i], zAxis=pos[i])
            wire_tag = gmsh.model.occ.add_wire([circle_tag])
            circle_tags.append(circle_tag)
            wire_tags.append(wire_tag)

        gmsh.model.occ.synchronize()
        gmsh.model.occ.addThruSections(wire_tags, makeSolid=True, makeRuled=True, smoothing=True)
        
        # Naming the surfaces
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.getEntities(2)

        gmsh.model.add_physical_group(2, [surfaces[-2][1]], inlet_marker, name="inlet")
        gmsh.model.add_physical_group(2, [surfaces[-1][1]], outlet_marker, name="outlet")
        gmsh.model.add_physical_group(2, [x[1] for x in surfaces[:-2]], wall_marker, name="walls")
        gmsh.model.add_physical_group(3, [1], 1, name="FluidZone")

        # Generating the mesh
        gmsh.model.occ.synchronize()
        set_mesh_options()
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size/(rad.mean()*15))
        gmsh.model.mesh.generate(3)

        # Write mesh file
        gmsh.write(os.path.join(mesh_dir, name+'.msh' ))
        gmsh.finalize()

def create_holed_cylinder(n_geo, n_holes):
    start_num = get_start_number("holed_cylinder_")
    print("Starting from {start_num}".format(start_num=start_num))
    for i in range(n_geo):
        gmsh.initialize()
        name = "holed_cylinder_" + str(i + int(start_num) + 1)
        print("Creating {name} now".format(name=name))
        gmsh.model.add(name)
        rand_scale_cylinder = np.random.uniform(.4, 1, 4)
        rand_pos_cylinder = np.random.uniform(-1, 1, [3])

        rand_scale_holes = np.random.uniform(.05, .2, n_holes)
        rand_pos_holes_x = np.random.uniform(rand_pos_cylinder[0], rand_pos_cylinder[0] + rand_scale_cylinder[0], n_holes)
        rand_pos_holes_y = np.random.uniform(rand_pos_cylinder[1], rand_pos_cylinder[1] + rand_scale_cylinder[1], n_holes)
        rand_pos_holes_z = np.random.uniform(rand_pos_cylinder[2], rand_pos_cylinder[2] + rand_scale_cylinder[2], n_holes)

        # Create a new cyliner
        cylinder = gmsh.model.occ.add_cylinder(rand_pos_cylinder[0], rand_pos_cylinder[1], rand_pos_cylinder[2], rand_scale_cylinder[0], rand_scale_cylinder[1], rand_scale_cylinder[2], rand_scale_cylinder[3], tag=10)
        gmsh.model.occ.synchronize()
        for hole in range(n_holes):
            gmsh.model.occ.add_sphere(rand_pos_holes_x[hole], rand_pos_holes_y[hole], rand_pos_holes_z[hole], rand_scale_holes[hole], tag=hole)

        gmsh.model.occ.synchronize()

        # Converting the surface tags(holes and 10 for cylinder) to a pair of tags (dim, tag)

        holes_tags = [(3, x) for x in range(n_holes)]
        print(holes_tags)
        gmsh.model.occ.cut([(3, 10)], holes_tags, removeObject=True, removeTool=True, tag=7) # 7 is the tag of the final volume, its just a random number
        gmsh.model.occ.synchronize()
        # Naming the surfaces
        surfaces = gmsh.model.getEntities(2)
        print(surfaces)
        
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.getEntities(2)
        print(surfaces)
        walls = []
        for surface in surfaces:
            com = gmsh.model.occ.getCenterOfMass(2, surface[1])
            com = list(com)
            
            if np.allclose(com, rand_pos_cylinder):
                gmsh.model.add_physical_group(2, [surface[1]], inlet_marker, name= 'inlet')
                print(f"Found inlet at {surface}")
            elif np.allclose(com[0], rand_pos_cylinder[0] + rand_scale_cylinder[0]):
                gmsh.model.add_physical_group(2, [surface[1]], outlet_marker, name= 'outlet')
                print(f"Found outlet at {surface}")
            else:
                walls.append(surface[1])
                print(f"Found wall at {surface}")

        print(walls)
        gmsh.model.add_physical_group(2, walls, wall_marker, name= 'walls')
        gmsh.model.add_physical_group(3, [7], 1, name="FluidZone")
        
        # Generating the mesh
        gmsh.model.occ.synchronize()
        set_mesh_options()
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size/1.3)
        gmsh.model.mesh.generate(3)

        # Write mesh file
        gmsh.write(os.path.join(mesh_dir, name+'.msh' ))
        gmsh.finalize()

# %%

import numpy as np
import gmsh
import os

def create_backward_facing_steps(n_geo):
    start_num = get_start_number("forward_facing_step_")
    print("Starting from {start_num}".format(start_num=start_num))
    
    for i in range(n_geo):
        # Initialize Gmsh
        gmsh.initialize()
        name = "forward_facing_step_" + str(i + int(start_num) + 1)
        print("Creating {name} now".format(name=name))
        gmsh.model.add(name)
        
        # Random dimensions for the domain and step
        domain_length = np.random.uniform(1, 5) # Random domain length between 1 and 5 meters
        domain_height = np.random.uniform(1, 5)/2
        step_length = domain_length * np.random.uniform(.1, .9) 
        extrusion_length = np.random.uniform(1, 5)

        # Create the geometry
        gmsh.model.occ.addRectangle(0, 0, 0, domain_length, domain_height, tag=1)
        gmsh.model.occ.addRectangle(0, domain_height, 0, step_length, domain_height, tag=2)
        # add the two rectangles to create the step
        gmsh.model.occ.fuse([(2, 1)], [(2, 2)], tag=3, removeObject=True, removeTool=True)
        # create the volume
        gmsh.model.occ.extrude([(2, 3)], 0, 0, extrusion_length)

        # Synchronize the model
        gmsh.model.occ.synchronize()
        surface_tag = gmsh.model.getEntities(2)
        print(f"Surface tag is {surface_tag}")
        vol_tag = gmsh.model.getEntities(3)[0][1]
        print(f"Volume tag is {vol_tag}")
        gmsh.model.add_physical_group(2, [9], inlet_marker, name= 'outlet')
        gmsh.model.add_physical_group(2, [5], outlet_marker, name= 'inlet')
        gmsh.model.add_physical_group(2, [3, 4, 6, 7, 8, 10], wall_marker, name= 'wall')
        gmsh.model.add_physical_group(3, [vol_tag], 1, name="FluidZone")

        # Generating the mesh
        gmsh.model.occ.synchronize()
        set_mesh_options()
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size/1.3)
        gmsh.model.mesh.generate(3)
        # Write mesh file
        gmsh.write(os.path.join(mesh_dir, name+'.msh' ))

        # Finalize Gmsh for this geometry
        gmsh.finalize()

if __name__ == "__main__":
    create_backward_facing_steps(10)  # Create 50 geometries
