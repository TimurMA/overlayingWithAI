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
                     (99, 94, 255), (125, 78, 255), (149, 67, 238), (167, 62, 210), (178, 60, 178)]


def cut_top_clothes(image_path:str):
    image = cv2.imread(image_path)
    colored_mask = predictColoredMask(image)
    tf.keras.preprocessing.image.save_img("D:\Python37\Overlaying\output\colored_mask.png", colored_mask)
    color_mask = cv2.imread("D:\Python37\Overlaying\output\colored_mask.png")
    new_mask = np.zeros(color_mask.shape[:2], dtype=np.uint8)
    for color in colors_to_extract_top:
        mask = cv2.inRange(color_mask, color, color)
        new_mask = cv2.bitwise_or(new_mask, mask)
    result_cut = cv2.bitwise_and(image, image, mask = new_mask)
    cv2.imwrite("D:\Python37\Overlaying\output\cutted_clothes.png", result_cut)

cut_top_clothes("D:\Python37\Overlaying\output\\result.jpg")

