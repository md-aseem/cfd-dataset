import os
import shutil

# Get current working directory
training_dir = '/home/aseem/OpenFOAM/aseem-11/run/training_data/meshing_cases'
# Use os.listdir to get a list of all items in the directory
all_items = os.listdir(training_dir)
# Use a list comprehension to filter only the directories
cases = [item for item in all_items if os.path.isdir(os.path.join(training_dir, item))]

# %%

for case in cases:

    files = os.listdir(os.path.join(training_dir, case))
    if 'log.foam' not in files:
        ### Delete the file if log.foam doesn't exist.
        print("Deleting " + case)
        shutil.rmtree(os.path.join(training_dir, case))
    else:
        print("log.foam found in " + case)
        continue