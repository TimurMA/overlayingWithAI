import cv2

def scale_clothes(top, pants, top_keypoints, pants_keypoints, person_keypoints):

    # вычисление коэффициента
    scale_ratio_top = [abs((person_keypoints[1][0] - person_keypoints[0][0]) / (top_keypoints[1][0] - top_keypoints[0][0])),
                       abs((person_keypoints[3][1] - person_keypoints[2][1]) / (top_keypoints[3][1] - top_keypoints[2][1]))]
    scale_ratio_pants = [abs((person_keypoints[6][0] - person_keypoints[4][0]) / (pants_keypoints[2][0] - pants_keypoints[0][0])),
                       abs((person_keypoints[5][1] - person_keypoints[4][1]) / (pants_keypoints[1][1] - pants_keypoints[0][1]))]
    # изменение масштаба
    top = cv2.resize(top, None, fx=scale_ratio_top[0]+0.05, fy=scale_ratio_top[1]+0.05)
    pants = cv2.resize(pants, None, fx=scale_ratio_pants[0]+0.05, fy=scale_ratio_pants[1]+0.05)
    # изменение значений ключевых точек
    top_keypoints = [(top_keypoint[0]*(scale_ratio_top[0]+0.05), top_keypoint[1]*(scale_ratio_top[1]+0.05), top_keypoint[2])
                      for top_keypoint in top_keypoints]
    pants_keypoints = [(pants_keypoint[0]*(scale_ratio_pants[0]+0.05), pants_keypoint[1]*(scale_ratio_pants[1]+0.05), pants_keypoint[2])
                      for pants_keypoint in pants_keypoints]
    return [top, pants], [top_keypoints, pants_keypoints]
