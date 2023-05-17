import tensorflow as tf
import tf_bodypix.api as tfbd
import cv2
from tf_bodypix.draw import draw_poses
import numpy as np

def predictColoredMask(image):

    bodypix_model = tfbd.load_model(tfbd.download_model(
    tfbd.BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
    ))
    image_np = tf.keras.preprocessing.image.img_to_array(image)
    result = bodypix_model.predict_single(image_np)
    mask = result.get_mask(threshold=0.75)
    return result.get_colored_part_mask(mask)




colors_to_extract_top = [(47, 167, 239), (56, 140, 255), (75, 115, 255), (87, 245, 135), (91, 240, 175), 
                     (99, 94, 255), (125, 78, 255), (149, 67, 238), (167, 62, 210), (178, 60, 178), (169, 219, 28)]


def cut_top_clothes(image, image_keypoints, person_keypoints, output_mask_path:str, width, height):
    colored_mask = predictColoredMask(image)
    tf.keras.preprocessing.image.save_img(output_mask_path, colored_mask)
    color_mask = cv2.imread(output_mask_path)

    new_mask = np.zeros(color_mask.shape[:2], dtype=np.uint8)
    for color in colors_to_extract_top:
        mask = cv2.inRange(color_mask, color, color)
        new_mask = cv2.bitwise_or(new_mask, mask)

    alpha_channel = new_mask.copy()
    alpha_channel[alpha_channel>0] = 255
    image = image.astype(np.uint8)
    result_cut = cv2.merge((image[:,:,0], image[:,:,1], image[:,:,2], alpha_channel))

    
    scaled_neck = image_keypoints[0][:2]
    person1_neck = person_keypoints[0][:2]
    x = int(person1_neck[0] - scaled_neck[0])
    y = int(person1_neck[1] - scaled_neck[1])
    bg_width = width
    bg_height = height
    if (y+result_cut.shape[0] > bg_height):
        bg_height = y+result_cut.shape[0]
    if (x+result_cut.shape[1] > bg_width):
        bg_width = x+result_cut.shape[1]
    transparent_bg = np.zeros((bg_height, bg_width, 4), dtype=np.uint8)
    transparent_bg[:,:, 3] = 0
    transparent_bg[y:y+result_cut.shape[0], x:x+result_cut.shape[1]] = result_cut
    return transparent_bg[:height, :width]