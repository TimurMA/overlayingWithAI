import cv2
import numpy as np
from predict import *
import sys
#include 

sys.path.append('D:\\Python37\\openpose\\build\\python\\openpose\\Release')
import pyopenpose as op

def create_mask(body_parts, image):
    points = np.array([body_parts["mid_hip"], body_parts["neck"], 
                       body_parts["r_shoulder"], body_parts["r_elbow"], 
                       body_parts["r_wrist"], body_parts["l_shoulder"], 
                       body_parts["l_elbow"], body_parts["l_wrist"]])
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, points, (255, 255, 255))
    
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    image_masked = cv2.bitwise_and(image, image, mask=mask)
    return image_masked

def apply_clothes(body_parts, image, clothes):
    image_masked = create_mask(body_parts, image)

    scale = 1.5
    clothes_resized = cv2.resize(clothes, (int(scale * image_masked.shape[1]), int(scale * image_masked.shape[0])))

    x, y, w, h = cv2.boundingRect(np.array([body_parts["mid_hip"], body_parts["neck"], body_parts["r_shoulder"], body_parts["l_shoulder"]]))

    clothes_overlay = clothes_resized[y:y+h, x:x+w]
    image_masked[y:y+h, x:x+w] = cv2.addWeighted(clothes_overlay, 0.5, image_masked[y:y+h, x:x+w], 0.5, 0)

    return image_masked

params = {
    'model_folder': 'D:\\Python37\\openpose\\models',
    'model_pose': 'BODY_25',
    'number_people_max': 1,
    'net_resolution': '256x256',
    'disable_blending': True,
}

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

person1 = cv2.imread('D:\Python37\Overlaying\input\input.jpg', cv2.COLOR_RGB2BGR)
# person = cv2.equalizeHist(person)
person2 =  cv2.imread('D:\Python37\Overlaying\input\person1.jpg', cv2.COLOR_RGB2BGR)


datum = op.Datum()
datum.cvInputData = person2
opWrapper.emplaceAndPop(op.VectorDatum([datum]))
person2_keypoints = datum.poseKeypoints[0][1:9]
datum = op.Datum()
datum.cvInputData = person1
opWrapper.emplaceAndPop(op.VectorDatum([datum]))
person1_keypoints = datum.poseKeypoints[0][1:9]

def scale_image(image, keypoints, target_distance_ox, target_distance_oy):

    current_distance_ox = abs(keypoints[1][0]-keypoints[4][0])
    current_distance_oy = abs(keypoints[0][1]-keypoints[-1][1])

    scale_ratio_ox = target_distance_ox / current_distance_ox
    scale_ratio_oy = target_distance_oy / current_distance_oy

    scaled_image = cv2.resize(image, None, fx=scale_ratio_ox, fy=scale_ratio_oy)
    scaled_keypoints = [(int(keypoint[0] * scale_ratio_ox), int(keypoint[1] * scale_ratio_oy)) for keypoint in keypoints]
    
    return scaled_image, scaled_keypoints


target_distance_ox = abs(person1_keypoints[1][0]-person1_keypoints[4][0])
target_distance_oy = abs(person1_keypoints[0][1]-person1_keypoints[-1][1])

scaled_image_person2, scaled_keypoints_person2 = scale_image(person2, person2_keypoints, target_distance_ox, target_distance_oy)
                         



cv2.imwrite("D:\Python37\Overlaying\output\\result.jpg", scaled_image_person2)
