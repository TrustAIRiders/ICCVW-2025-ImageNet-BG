import os
from modules.util import helpers 
from modules.util import modelProvider as model
from modules.util import analysisFunctions as analysis
import cv2 as cv
import numpy as np

def_dest_path = "./background_selection/"   

def method1(amount = 0):
    print("Method 1 is fully manual, you'll need to choose the images yourself")

def method2(dest_path, path, classes):
    if not classes:
        print("Please specify desired classes in config.json (background_classes)")
        return
    candidates = {}
    candidateScores = {}
    for label in classes:
        candidateScores[label]=100
        candidates[label] = ""

    for image_path in helpers.universal_path_iterator(path):
        image_path = image_path.replace("\\","/")
        current_class = image_path.split('/')[-2]
        if current_class not in classes:
            continue
        try:
            image = cv.imread(image_path)
            image = cv.resize(image, (224,224))
            current_score = analysis.get_neutrailty_score(model.eval(image))
            if current_score < candidateScores[current_class]:
                candidateScores[current_class] = current_score
                candidates[current_class] = image_path
        except: 
            print("File " + image_path + " couldn't be read")

    for label in classes:
        src_fpath = candidates[label]
        dest_fpath = dest_path + label + '_method2_' + candidates[label].split('/')[-1]
        helpers.move_file(src_fpath, dest_fpath)


def method3(dest_path, path, number = 10):
    candidates = {}
    candidateVectors = {}
    
    for image_path in helpers.universal_path_iterator(path):
        image_path = image_path.replace("\\","/")
        current_class = image_path.split('/')[-2]
        try:
            image = cv.imread(image_path)
            image = cv.resize(image, (224,224))
            current_vector = model.eval(image)
            if current_class not in candidates or analysis.get_neutrailty_score(current_vector) < analysis.get_neutrailty_score(candidateVectors[current_class]):
                candidateVectors[current_class] = current_vector
                candidates[current_class] = image_path
        except: 
            print("File " + image_path + " couldn't be read")
        
    classes =  list(candidates.keys())
    vectors = np.squeeze(np.array(list(candidateVectors[label] for label in classes)))
    vectors = analysis.reduce_dims(vectors, min(1000, number))
    closest = analysis.cluster_dimensions(vectors, number)
    classes = np.array(classes)[closest]

    for label in classes:
        src_fpath = candidates[label]
        dest_fpath = dest_path + label + '_method3_' + candidates[label].split('/')[-1]
        helpers.move_file(src_fpath, dest_fpath)