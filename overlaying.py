import cv2
from propotions import scale_image
from cutting_clothes import cut_top_clothes
import numpy as np
import sys
from propotions import scale_image
#include 

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

person1 = cv2.imread('D:\Python37\Overlaying\input\input.jpg', cv2.COLOR_RGB2BGR)
# person = cv2.equalizeHist(person)
person2 =  cv2.imread('D:\Python37\Overlaying\input\\6802df34.jpg', cv2.COLOR_RGB2BGR)


datum = op.Datum()
datum.cvInputData = person2
opWrapper.emplaceAndPop(op.VectorDatum([datum]))
person2_keypoints = datum.poseKeypoints[0][1:9]
datum = op.Datum()
datum.cvInputData = person1
opWrapper.emplaceAndPop(op.VectorDatum([datum]))
person1_keypoints = datum.poseKeypoints[0][1:9]


target_distance_ox = abs(person1_keypoints[1][0]-person1_keypoints[4][0])
target_distance_oy = abs(person1_keypoints[0][1]-person1_keypoints[-1][1])

scaled_image = scale_image(person2, person2_keypoints, target_distance_ox, target_distance_oy)
# cv2.imwrite('D:\Python37\Overlaying\output\\result.jpg', scaled_image)
# cut_top_clothes('D:\Python37\Overlaying\output\\result.jpg', 'D:\Python37\Overlaying\output\colored_mask.png', 
#                 'D:\Python37\Overlaying\output\cutted_clothes.png')

datum = op.Datum()
datum.cvInputData = scaled_image
opWrapper.emplaceAndPop(op.VectorDatum([datum]))
scaled_keypoints = datum.poseKeypoints[0][1:9]

dx = scaled_keypoints[4][0] - scaled_keypoints[1][0]
dy = scaled_keypoints[4][1] - scaled_keypoints[1][1]
scaled_neck = scaled_keypoints[0][:2]
person1_neck = person1_keypoints[0][:2]

x = int(scaled_neck[0] - person1_neck[0] + (person1.shape[1] - dx)/2)
y = int(scaled_neck[1] - person1_neck[1] + (person1.shape[0] - dy)/2)

person1[y:y+scaled_image.shape[0],x:x+scaled_image.shape[1]] = scaled_image
cv2.imshow('result', person1)



