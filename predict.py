import tensorflow as tf
import tf_bodypix.api as tfbd
from tf_bodypix.draw import draw_poses
import numpy as np

def predictColoredMask(image, name):

    bodypix_model = tfbd.load_model(tfbd.download_model(
    tfbd.BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
    ))
    image_np = np.array(image)
    result = bodypix_model.predict_single(image_np)
    mask = result.get_mask(threshold=0.75)
    return mask