import re
import os
import shutil
import numpy as np

"""
Renames the multiple file within the same directory with appending number
"""
input_dir = "/Users/p099947-dev/Desktop/Master2RD/Blend/Dataset__v3/normal"
output_dir ="/Users/p099947-dev/PycharmProjects/Vision/Vision/data/processed/Dataset_v3/bottle"

def copy_split (input_dir, output_dir):

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    filepaths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if re.search("[0-9]", f)]
    train_files = np.random.choice(filepaths, int(0.8 * len(filepaths)), replace=False).tolist()
    val_files = list(set(filepaths) - set(train_files))

    for p in train_files:
        fn = os.path.basename(p).replace("nor", "")
        path = os.path.join(train_dir, fn)
        shutil.copyfile(p, path)

    for p in val_files:
        fn = os.path.basename(p).replace("nor", "")
        path = os.path.join(val_dir, fn)
        shutil.copyfile(p, path)

copy_split(input_dir,output_dir)