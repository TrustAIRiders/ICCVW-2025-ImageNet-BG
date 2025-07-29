import os
import shutil
import cv2 as cv

def universal_path_iterator(path): #prosty irerator do wszystkfich plikow w danym folderze
    if os.path.isfile(path):
       yield path
    else:
        subfolders = os.listdir(path)
        for subfolder in subfolders:
            newpath = os.path.join(path, subfolder)
            yield from universal_path_iterator(newpath)


def move_file(src_fpath, dest_fpath):
    try:
        shutil.copy(src_fpath, dest_fpath)
    except IOError as io_err:
        os.makedirs(os.path.dirname(dest_fpath))
        shutil.copy(src_fpath, dest_fpath)


def save_imge_file(image, dest_fpath):
    if not os.path.isdir(os.path.dirname(dest_fpath)):\
        os.makedirs(os.path.dirname(dest_fpath))        
    cv.imwrite(dest_fpath, image)