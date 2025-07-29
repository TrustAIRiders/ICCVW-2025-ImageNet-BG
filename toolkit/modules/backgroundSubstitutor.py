import cv2 as cv
import numpy as np
import math
from modules.util import helpers
from modules.util import classes_s
import pandas as pd
import signal

class GracefulExiter():

    def __init__(self):
        self.state = False
        signal.signal(signal.SIGINT, self.change_state)

    def change_state(self, signum, frame):
        print("exit flag set to True (repeat to exit now)")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.state = True

    def exit(self):
        return self.state


flag = GracefulExiter()


WORKING_RESOLUTION = (896, 896)
FINAL_RESOLUTION = (224, 224)
backgrounds_dir = "./background_selection/"  
dest_dir = "./output/" 

HAQA_path = "./modules/util/HAQA.csv"
df = pd.read_csv(HAQA_path)
quality_dict = pd.Series(df["Overall Quality"].values ,index=df["Class Label"]).to_dict()

def get_image(path_to_file):
    img = cv.imread(path_to_file)
    img = cv.resize(img, WORKING_RESOLUTION)
    return img
    
def overlay_image(background, image, mask):
    result = background.copy()
    result[mask] = image[mask]
    return result

def cat_num_from_code(code):
    return classes_s.CLASSES.index(code)

def get_mask(path_to_mask, class_label , newdims = WORKING_RESOLUTION):
    mask = cv.imread(path_to_mask, flags = cv.IMREAD_UNCHANGED)
    mask = cv.cvtColor(mask,cv.COLOR_BGR2RGB)
    mask = cv.resize(mask, newdims)
    col_val = cat_num_from_code(class_label)+1
    col_val = [col_val % 256, math.floor(col_val/256), 0]
    mask = np.all(mask == col_val, -1)
    return mask

def generate_images(images_path, masks_path, dest_path = dest_dir, backgrounds_path = backgrounds_dir):
    bg_pathing = []
    bgs = []
    for background_file in helpers.universal_path_iterator(backgrounds_dir):
        background_file = background_file.replace("\\","/")
        try:
            bg_pathing.append(background_file.replace(backgrounds_path, "").split(".")[-2])
            bgs.append(get_image(background_file))
        except: 
            print("Files " + background_file + " could not be processed")

    
    for img_file, mask_file in zip(helpers.universal_path_iterator(images_path), helpers.universal_path_iterator(masks_path)):
        img_file = img_file.replace("\\","/")
        mask_file = mask_file.replace("\\","/")
        img_class = img_file.split("/")[-2]
        img_name = img_file.split("/")[-1]
        try:
            img = get_image(img_file)
            msk = get_mask(mask_file, img_class)

            for bg, bgp in zip(bgs, bg_pathing):
                output = overlay_image(bg, img, msk)
                output = cv.resize(output, FINAL_RESOLUTION)
                dest = dest_path +  quality_dict[img_class] + "/" +  bgp + "/" + img_class + "/" + img_name
                helpers.save_imge_file(output, dest)
        except: 
            print("Files " + img_file + " and/or " + mask_file + " could not be processed")
        if flag.exit():
            print('\n')
            print("exited due to user interrupt")
            quit()