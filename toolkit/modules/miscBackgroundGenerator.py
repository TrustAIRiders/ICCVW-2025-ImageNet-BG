import numpy as np
import cv2 as cv
from modules.util import helpers

IMG_SHAPE = (244,244,3)
dest_path_def = "./background_selection/"

def gen_noise(colored = True):
    noise = np.full(IMG_SHAPE, 0, np.float32)
    noise2 = np.full((244,244,1), 0, np.float32)
    cv.randn(noise2, 0, 0.05)
    noise3 = noise2
    noise2 = np.full((244,244,1), 0, np.float32)
    cv.randn(noise2, 0, 0.05)
    noise3 = np.dstack((noise3, noise2))
    noise2 = np.full((244,244,1), 0, np.float32)
    cv.randn(noise2, 0, 0.05)
    noise3 = np.dstack((noise3, noise2))
    noise = noise + noise3
    noise = noise - np.min(noise)
    noise = noise / np.max(noise)
    noise = noise*225
    noise = noise.astype(np.uint8)
    if not colored:
        noise = cv.cvtColor(noise, cv.COLOR_BGR2GRAY)
    return noise
    
def gen_full_color(color):
    return np.dstack(
        (np.full(IMG_SHAPE[:2], color[0], np.float32), 
         np.full(IMG_SHAPE[:2], color[1], np.float32), 
         np.full(IMG_SHAPE[:2], color[2], np.float32))).astype(np.uint8)

def gen_and_save_suite(dest_path = dest_path_def):
    helpers.save_imge_file(gen_full_color((0,0,0)), dest_path+"method4_black.png")
    helpers.save_imge_file(gen_full_color((255,255,255)), dest_path+"method4_white.png")
    helpers.save_imge_file(gen_noise(False), dest_path+"method4_gray_noise.png")
    helpers.save_imge_file(gen_noise(True), dest_path+"method4_colored_noise.png")