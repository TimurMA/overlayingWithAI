import tensorflow as tf
import tf_bodypix.api as tfbd
import cv2
from tf_bodypix.draw import draw_poses
import numpy as np

def predictColoredMask(image):

    bodypix_model = tfbd.load_model(tfbd.download_model(
    tfbd.BodyPixModelPaths.RESNET50_FLOAT_STRIDE_16
    ))
    image_np = tf.keras.preprocessing.image.img_to_array(image)
    result = bodypix_model.predict_single(image_np)
    mask = result.get_mask(threshold=0.75)
    return result.get_colored_part_mask(mask)


def cut_clothes(clothes, clothes_keypoint, person_keypoint, output_mask_path:str, width, height, colors_to_extract):
    colored_mask = predictColoredMask(clothes)
    tf.keras.preprocessing.image.save_img(output_mask_path, colored_mask)
    color_mask = cv2.imread(output_mask_path)

    new_mask = np.zeros(color_mask.shape[:2], dtype=np.uint8)
    for color in colors_to_extract:
        mask = cv2.inRange(color_mask, color, color)
        new_mask = cv2.bitwise_or(new_mask, mask)

    alpha_channel = new_mask.copy()
    alpha_channel[alpha_channel>0] = 255
    clothes = clothes.astype(np.uint8)
    result_cut = cv2.merge((clothes[:,:,0], clothes[:,:,1], clothes[:,:,2], alpha_channel))

    x = int(person_keypoint[0] - clothes_keypoint[0])
    y = int(person_keypoint[1] - clothes_keypoint[1])
    
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