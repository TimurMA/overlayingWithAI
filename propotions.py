import cv2

def scale_image(image, keypoints, target_distance_ox, target_distance_oy):

    current_distance_ox = keypoints[1][0]-keypoints[4][0]
    current_distance_oy = keypoints[0][1]-keypoints[-1][1]

    scale_ratio_ox = target_distance_ox / current_distance_ox
    scale_ratio_oy = target_distance_oy / current_distance_oy

    scaled_image = cv2.resize(image, None, fx=scale_ratio_ox+0.05, fy=scale_ratio_oy+0.05)
    scaled_keypoints = [(keypoint[0]*(scale_ratio_ox+0.05), keypoint[1]*(scale_ratio_oy+0.05), keypoint[2]) for keypoint in keypoints]
    
    return scaled_image, scaled_keypoints
