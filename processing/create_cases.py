import os
import shutil

base_dir = "/home/aseem/OpenFOAM/aseem-11/run/training_data/"
meshes_dir = os.path.join(base_dir, 'meshes')
case_dir_path = os.path.join(base_dir, 'case_dir')
meshing_cases_dir = os.path.join(base_dir, 'meshing_cases')

# Create 'meshing cases' directory if it doesn't exist
if not os.path.exists(meshing_cases_dir):
    os.makedirs(meshing_cases_dir)

# Iterate over each '.msh' file in the 'meshes' directory
for file in os.listdir(meshes_dir):
    if file.endswith('.msh'):
        file_path = os.path.join(meshes_dir, file)

        # Create a new directory for this '.msh' file inside 'meshing cases'
        new_dir_path = os.path.join(meshing_cases_dir, os.path.splitext(file)[0])
        if not os.path.exists(new_dir_path):
            os.makedirs(new_dir_path)
            print(f"Processing {file}")
        
            # Copy the contents of 'case_dir' into this new directory
            for item in os.listdir(case_dir_path):

                item_path = os.path.join(case_dir_path, item)
                if os.path.isdir(item_path):
                    shutil.copytree(item_path, os.path.join(new_dir_path, item))
                else:
                    shutil.copy2(item_path, new_dir_path)
        
            
            # Copy the '.msh' file into this new directory
            shutil.copy2(file_path, new_dir_path)
            

        else:
            print(f"Skipping {file} as it already exists.")
        


print("All done!")
