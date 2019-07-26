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
ROOT_DIR = os.path.abspath('../..')


ann_folder_path = os.path.join(ROOT_DIR, 'data/processed/Dataset_v4/annotations/val')

masks_folder_path = os.path.join(ROOT_DIR, 'data/raw/Dataset_v3/masks/val')


files = os.listdir(masks_folder_path)

for file in files:
    if re.search("[0-9]", file):
        filename, file_extension = os.path.splitext(file)

        new_couv_name = "{}_cap".format(filename)
        new_bott_name = "{}_bottle".format(filename)
        new_etiq_name = "{}_label".format(filename)
        new_bg_name = "{}_bg".format(filename)

        image = cv2.imread(os.path.join(masks_folder_path,file))
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

        # Interval of color to track

        # couvercle
        lower_couv = np.array([0, 100, 100])
        upper_couv = np.array([0, 255, 255])

        # bouteille :
        lower_bott = np.array([60, 100, 255])
        upper_bott = np.array([60, 255, 255])

        # etiquette :
        lower_etiq = np.array([120, 100, 100])
        upper_etiq = np.array([120, 255, 255])

        # Mask obtention + Mask processing
        mask_couv = cv2.inRange(hsv_image, lower_couv, upper_couv)
        mask_couv = cv2.morphologyEx(mask_couv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask_couv = cv2.morphologyEx(mask_couv, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        mask_bott = cv2.inRange(hsv_image, lower_bott, upper_bott)
        mask_bott = cv2.morphologyEx(mask_bott, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask_bott = cv2.morphologyEx(mask_bott, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        mask_etiq = cv2.inRange(hsv_image, lower_etiq, upper_etiq)
        mask_etiq = cv2.morphologyEx(mask_etiq, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask_etiq = cv2.morphologyEx(mask_etiq, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        mask_tot = mask_bott + mask_couv + mask_etiq

        mask_bg = 255 - mask_tot


        cv2.imwrite(os.path.join(ann_folder_path,new_couv_name + file_extension),
                    mask_couv)
        cv2.imwrite(os.path.join(ann_folder_path,new_bott_name + file_extension),
                    mask_bott)
        cv2.imwrite(os.path.join(ann_folder_path,new_etiq_name + file_extension),
                    mask_etiq)
        cv2.imwrite(os.path.join(ann_folder_path, new_bg_name + file_extension),
                    mask_bg)

