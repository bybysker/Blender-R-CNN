import cv2
import numpy as np
import os
import re


"""
couvercle = np.uint8([[[243,164,141 ]]])
bottle = np.uint8([[[253,242,254 ]]])
etiquette = np.uint8([[[228,254,255 ]]])

elems = [couvercle,bottle,etiquette]

hsv_elems = [0, 0, 0]
for i,elem in enumerate(elems):
    hsv_elems[i] = cv2.cvtColor(elem, cv2.COLOR_BGR2HSV)
np.array(hsv_elems)
lower_list =
#upper_list =np.array([cvt_hsv_couv[0, 0, 1]+10, 255, 255])

couvercle: `
lower_couv = np.array([14-10, 100, 100])
upper_couv = np.array([14+10, 255, 255])

bouteille : 
lower_bott = np.array([163-15, 10, 255])
upper_bott = np.array([163+15, 50, 255])

etiquette :
lower_etiq = np.array([80, 0, 0])
upper_etiq = np.array([100, 255, 255])

"""


# Convert color to hsv
#couvercle_col = np.uint8([[[141, 164, 243]]])
#cvt_hsv_couv = cv2.cvtColor(couvercle_col, cv2.COLOR_BGR2HSV)

# Image processing

"""
def make_masks(image):
    
#    Given an image, makes a grayscale mask of each of the instances of that image
    
    
"""

ann_folder_path = "/Users/p099947-dev/PycharmProjects/Vision/Vision/data/processed/Dataset_v2/annotations/train"
masks_folder_path = "/Users/p099947-dev/PycharmProjects/Vision/Vision/data/raw/Dataset_v2/masks/train"

files = os.listdir(masks_folder_path)

for file in files:
    if re.search("[0-9]", file):
        filename, file_extension = os.path.splitext(file)

        new_couv_name = "{}_cap".format(filename)
        new_bott_name = "{}_bottle".format(filename)
        new_etiq_name = "{}_label".format(filename)

        image = cv2.imread(os.path.join(masks_folder_path,file))
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

        # Interval of color to track

        # couvercle:
        lower_couv = np.array([14 - 10, 100, 100])
        upper_couv = np.array([14 + 10, 255, 255])

        # bouteille :
        lower_bott = np.array([163 - 15, 10, 255])
        upper_bott = np.array([163 + 15, 50, 255])

        # etiquette :
        lower_etiq = np.array([80, 0, 0])
        upper_etiq = np.array([100, 255, 255])

        # Mask obtention
        mask_couv = cv2.inRange(hsv_image, lower_couv, upper_couv)
        mask_bott = cv2.inRange(hsv_image, lower_bott, upper_bott)
        mask_etiq = cv2.inRange(hsv_image, lower_etiq, upper_etiq)



        cv2.imwrite(os.path.join(ann_folder_path,new_couv_name + file_extension),
                    mask_couv)
        cv2.imwrite(os.path.join(ann_folder_path,new_bott_name + file_extension),
                    mask_bott)
        cv2.imwrite(os.path.join(ann_folder_path,new_etiq_name + file_extension),
                    mask_etiq)


