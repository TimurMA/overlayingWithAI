import cv2
from propotions import scale_image
from cutting_clothes import cut_top_clothes
import numpy as np
import sys
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

person1 = cv2.imread('D:\Python37\Overlaying\input\MAGAZHIEST.jpg', cv2.COLOR_RGB2BGR)
# person = cv2.equalizeHist(person)
person2 =  cv2.imread('D:\Python37\Overlaying\input\osuzhdai.jpg', cv2.COLOR_RGB2BGR)


datum = op.Datum()
datum.cvInputData = person2
opWrapper.emplaceAndPop(op.VectorDatum([datum]))
person2_keypoints = datum.poseKeypoints[0][1:9]
datum = op.Datum()
datum.cvInputData = person1
opWrapper.emplaceAndPop(op.VectorDatum([datum]))
person1_keypoints = datum.poseKeypoints[0][1:9]


target_distance_ox = person1_keypoints[1][0]-person1_keypoints[4][0]
target_distance_oy = person1_keypoints[0][1]-person1_keypoints[-1][1]

scaled_image, scaled_keypoints = scale_image(person2, person2_keypoints, target_distance_ox, target_distance_oy)
cutted_image = cut_top_clothes(scaled_image, scaled_keypoints, person1_keypoints,
                 'D:\Python37\Overlaying\output\colored_mask.png', person1.shape[1], person1.shape[0])
cv2.imwrite('D:\Python37\Overlaying\output\cutted_clothes_chekist.png', cutted_image)
alpha = cutted_image[:,:, 3] / 255.0
cutted_image = cutted_image[:,:,:3]
cutted_image = cv2.cvtColor(cutted_image, cv2.COLOR_RGBA2RGB)

person = cv2.imread('D:\Python37\Overlaying\input\MAGAZHIEST.jpg')


composite = cv2.bitwise_and(cutted_image, cutted_image, mask = (alpha > 0).astype(np.uint8))
backround_masked = cv2.bitwise_and(person, person, mask = (1 - alpha >0).astype(np.uint8))

result = cv2.add(composite, backround_masked)
result_path = 'D:\Python37\Overlaying\output\outputMAGA.jpg'
cv2.imwrite(result_path, result)