import cv2
from propotions import scale_clothes
from cutting_clothes import cut_clothes
import numpy as np
import sys
from urllib.request import urlopen
import os, random


sys.path.append('D:\\Python37\\openpose\\build\\python\\openpose\\Release')
import pyopenpose as op

params = {
    'model_folder': 'D:\Python37\openpose\models',
    'model_pose': 'BODY_25',
    'number_people_max': 1,
    'net_resolution': '256x128',
    'disable_blending': True,
}

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

colors_to_extract_top = [(47, 167, 239), (56, 140, 255), (75, 115, 255), (87, 245, 135), (91, 240, 175), 
                     (99, 94, 255), (125, 78, 255), (149, 67, 238), (167, 62, 210), (178, 60, 178), (169, 219, 28)]

colors_to_extract_pants = [(96, 247, 96), (115, 243 , 64), (141, 234, 40), (169, 219, 28), 
                           (194, 199, 26), (213, 176, 33), (224, 125, 65), (224, 150 ,47)]

def get_keypoints(image):
    datum = op.Datum()
    datum.cvInputData = image
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    return datum.poseKeypoints[0]




def overlay():

    # req = urlopen(person_URL)
    # per_array = np.asarray(bytearray(req.read()), dtype = 'uint8')
    # person = cv2.imdecode(per_array, cv2.IMREAD_COLOR)
    person = cv2.imread('input\input.jpg')
    person_BGR = cv2.cvtColor(person, cv2.COLOR_RGB2BGR)
    dir_tops = 'tops\\'
    dir_pants = 'pants\\'
    top = cv2.imread(dir_tops + random.choice(os.listdir(dir_tops)))
    pants = cv2.imread(dir_pants + random.choice(os.listdir(dir_pants)))
    top_BGR = cv2.cvtColor(top, cv2.COLOR_RGB2BGR)
    pants_BGR = cv2.cvtColor(pants, cv2.COLOR_RGB2BGR)

    person_keypoints = get_keypoints(person_BGR)
    person_keypoints = [person_keypoints[2], person_keypoints[5], person_keypoints[1], person_keypoints[8], 
                        person_keypoints[9], person_keypoints[10], person_keypoints[12]]
    top_keypoints = get_keypoints(top_BGR)
    top_keypoints = [top_keypoints[2], top_keypoints[5], top_keypoints[1], top_keypoints[8]]
    pants_keypoints = get_keypoints(pants_BGR)
    pants_keypoints = [pants_keypoints[9], pants_keypoints[10], pants_keypoints[12]]
    

    scaled_clothes, scaled_clothes_keypoints = scale_clothes(top, pants, top_keypoints, pants_keypoints, person_keypoints)

    cutted_top = cut_clothes(scaled_clothes[0], scaled_clothes_keypoints[0][0], person_keypoints[0],
                 'temp\colored_mask_top.png', person.shape[1], person.shape[0], colors_to_extract_top)
    cutted_pants = cut_clothes(scaled_clothes[1], scaled_clothes_keypoints[1][0], person_keypoints[4],
                 'temp\colored_mask_pants.png', person.shape[1], person.shape[0], colors_to_extract_pants)
    
    alpha_pants = cutted_pants[:, :, 3] / 255.0
    alpha_top = cutted_top[:,:, 3] / 255.0

    cutted_pants = cutted_pants[:,:,:3]
    cutted_pants = cv2.cvtColor(cutted_pants, cv2.COLOR_RGBA2RGB)
    cutted_top = cutted_top[:,:,:3]
    cutted_top = cv2.cvtColor(cutted_top, cv2.COLOR_RGBA2RGB)
    
    composite_pants = cv2.bitwise_and(cutted_pants, cutted_pants, mask=(alpha_pants > 0).astype(np.uint8))
    composite_top = cv2.bitwise_and(cutted_top, cutted_top, mask=(alpha_top > 0).astype(np.uint8))
    
    backround_masked_pants = cv2.bitwise_and(person, person, mask=(1 - alpha_pants > 0).astype(np.uint8))
    pasted_pants_image = cv2.add(composite_pants, backround_masked_pants)
    backround_masked_top = cv2.bitwise_and(pasted_pants_image, pasted_pants_image, mask=(1 - alpha_top > 0).astype(np.uint8))

    return cv2.add(composite_top, backround_masked_top)


cv2.imwrite('output\output.jpg', overlay())