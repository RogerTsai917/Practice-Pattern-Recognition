import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf


def readEachImageInFolder(path):
    images_list = os.listdir(path)
    result = []
    for image_name in images_list:
        im = Image.open(os.path.join(path, image_name))
        result.append(np.array(im))
    return result

def readMoneyData(base_folder):
    list_100 = readEachImageInFolder(os.path.join(base_folder, '100'))
    list_500 = readEachImageInFolder(os.path.join(base_folder, '500'))
    list_1000 = readEachImageInFolder(os.path.join(base_folder, '1000'))
    array_100 = np.asarray(list_100)
    array_500 = np.asarray(list_500)
    array_1000 = np.asarray(list_1000)
    return array_100, array_500, array_1000

def augmentData(save_dir, images_array):
    count = 1
    for image_array in images_array:
        tf.keras.preprocessing.image.save_img(os.path.join(save_dir,f'{count}.jpg'), image_array)
        # 水平翻轉並存檔
        flipped = tf.image.flip_left_right(image_array)
        tf.keras.preprocessing.image.save_img(os.path.join(save_dir, f'{count}_flipped.jpg'), flipped)
        # 旋轉90度並存檔
        rotated = tf.image.rot90(image_array)
        tf.keras.preprocessing.image.save_img(os.path.join(save_dir, f'{count}_rotated.jpg'), rotated)
        # 改變亮度
        bright = tf.image.adjust_brightness(image_array, 0.2)
        tf.keras.preprocessing.image.save_img(os.path.join(save_dir, f'{count}_bright.jpg'), bright)
        count += 1

def visualize(original, augmented):
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1,2,2)
    plt.title('Augmented image')
    plt.imshow(augmented)
    plt.show(block=False)
    plt.pause(3)
    plt.close("all")

if __name__ == "__main__":
    resource_folder = 'origin_data'
    target_folder = 'augmentation_data'

    array_100, array_500, array_1000 = readMoneyData(resource_folder)

    augmentData(os.path.join(target_folder, '100'), array_100)
    augmentData(os.path.join(target_folder, '500'), array_500)
    augmentData(os.path.join(target_folder, '1000'), array_1000)
