# %%
import subprocess
import os

solver_command = 'foamRun > log.foam'

# Get current working directory
training_dir = '/home/aseem/OpenFOAM/aseem-11/run/training_data/meshing_cases'
# Use os.listdir to get a list of all items in the directory
all_items = os.listdir(training_dir)
# Use a list comprehension to filter only the directories
cases = [item for item in all_items if os.path.isdir(os.path.join(training_dir, item))]

# %%

for case in cases:

    files = os.listdir(os.path.join(training_dir, case))

    if case + '.msh' not in files:
        print(f"{case}.msh not found in " + case)
        continue
    elif 'log.foam' in files:
        print("log.foam already exists in " + case)
        continue
    else:
        print("case.msh found in " + case)
        print("log.foam not found in " + case)
        os.chdir(os.path.join(training_dir, case))
        print("Running solver for case: " + case)
        print("In directory: " + os.getcwd())
        print("Changing directory to: " + os.path.join(training_dir, case))
    
    ### Change current working directory to the current case directory.
    subprocess.run("foamCleanCase", shell=True) ### Deletes all the timesteps and meshing and everything if exists.
    subprocess.run('gmshToFoam ' + case + '.msh', shell=True)

    ### Reading the boundary file and changing the walls B.C from 'patch' to 'wall'.
    with open('constant/polyMesh/boundary', 'r') as file:
        filedata = file.read()
        walls_index = filedata.find('walls')
        type_index = filedata.find('type', walls_index)
        physicalType_index = filedata.find('physicalType', walls_index)
        # Replace 'patch' with 'wall' for 'type' and 'physicalType'
        filedata = filedata[:type_index] + 'type            wall;\n' + '        physicalType    wall' + filedata[physicalType_index+21:]

        with open('constant/polyMesh/boundary', 'w') as file:
            file.write(filedata)
        
        try:
            subprocess.run('checkMesh', shell=True)
        except:
            print("checkMesh failed for case: " + case)
            continue

        subprocess.run('decomposePar', shell=True)
        print("Decomposed mesh for case: " + case + " successfully. " + "Now running solver...")

        try:
            subprocess.run("mpirun -np 6 foamRun -parallel > log.foam", shell=True)
            #subprocess.run("foamRun", shell=True)
        except :
            print("Solver failed for case: " + case)
            continue

        subprocess.run('reconstructPar', shell=True)
        subprocess.run('foamToVTK', shell=True)
