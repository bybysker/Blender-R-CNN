import os
import re

#os.environ['ROOT_DIR'] = '/Users/p099947-dev/PycharmProjects/Vision/Vision'
#ROOT_DIR = os.environ['ROOT_DIR']

ROOT_DIR = os.path.abspath('../..')

def rename_imgs(folder_path):
    """
    Renames the multiple file within the same directory with appending number
    """
    files = os.listdir(folder_path)

    for file in files:
        if re.search("[0-9]",file):
            filename, file_extension = os.path.splitext(file)
            new_filename = re.findall("[0-9]+", filename)[0]
            os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_filename + file_extension))


rename_imgs(os.path.join(ROOT_DIR,'data/raw/Dataset_v2/bottle_train'))
rename_imgs(os.path.join(ROOT_DIR, 'data/raw/Dataset_v2/masks'))


