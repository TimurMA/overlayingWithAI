import cv2
from propotions import scale_image
from cutting_clothes import cut_top_clothes
import numpy as np
import sys
import os, random
from propotions import scale_image


sys.path.append('D:\\Python37\\openpose\\build\\python\\openpose\\Release')
import pyopenpose as op

params = {
    'model_folder': 'D:\\Python37\\openpose\\models',
    'model_pose': 'BODY_25',
    'number_people_max': 1,
    'net_resolution': '256x128',
    'disable_blending': True,
}

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

def get_keypoints(image):
    datum = op.Datum()
    datum.cvInputData = image
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    return datum.poseKeypoints[0][1:9]


def overlay(person_path):

    person = cv2.imread(person_path)
    person_BGR = cv2.cvtColor(person, cv2.COLOR_RGB2BGR)
    dir = 'clothes\\'
    clothes = cv2.imread(dir + random.choice(os.listdir(dir)))
    clothes_BGR = cv2.cvtColor(clothes, cv2.COLOR_RGB2BGR)

    person_keypoints = get_keypoints(person_BGR)
    clothes_keypoints = get_keypoints(clothes_BGR)

    target_distance_ox = person_keypoints[1][0] - person_keypoints[4][0]
    target_distance_oy = person_keypoints[0][1] - person_keypoints[-1][1]
    scaled_image, scaled_keypoints = scale_image(clothes, clothes_keypoints, target_distance_ox, target_distance_oy)

    cutted_image = cut_top_clothes(scaled_image, scaled_keypoints, person_keypoints,
                 'temp\colored_mask.png', person.shape[1], person.shape[0])
    alpha = cutted_image[:,:, 3] / 255.0
    cutted_image = cutted_image[:,:,:3]
    cutted_image = cv2.cvtColor(cutted_image, cv2.COLOR_RGBA2RGB)
    
    composite = cv2.bitwise_and(cutted_image, cutted_image, mask = (alpha > 0).astype(np.uint8))
    backround_masked = cv2.bitwise_and(person, person, mask = (1 - alpha >0).astype(np.uint8))

    return cv2.add(composite, backround_masked)
