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

p1_hands = person1_keypoints[1:7][:2]
p1_neck = person1_keypoints[0][:2]
p1_hip = person1_keypoints[7][:2]

p2_hands = person2_keypoints[1:7][:2]
p2_neck = person2_keypoints[0][:2]
p2_hip = person2_keypoints[7][:2]

propotions_neck_hip = np.linalg.norm(p1_neck - p2_hip) / np.linalg.norm(p2_neck - p2_hip)
propotions_hands = np.linalg.norm(p1_hands)/np.linalg.norm(p2_hands)

scaled_image = cv2.resize(person2, None, fx = propotions_hands, fy= propotions_hands)

new_neck_position = (int(p2_neck[0] * propotions_neck_hip), int(p2_neck[1] * propotions_neck_hip))
new_hip_position = (int(p2_hip[0] * propotions_neck_hip), int(p2_hip[1] * propotions_neck_hip))
offset_x = new_neck_position[0] - p2_neck[0]
offset_y = new_neck_position[1] - p2_neck[0]

M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
translated_image = cv2.warpAffine(scaled_image, M, (person2.shape[1], person2.shape[0]))

cv2.imshow("result", translated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()