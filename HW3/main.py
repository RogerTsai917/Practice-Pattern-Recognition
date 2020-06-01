import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, SpatialDropout2D

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
    label_100 = np.zeros(array_100.shape[0], dtype=np.int)
    label_500 = np.ones(array_500.shape[0], dtype=np.int)*1
    label_1000 = np.ones(array_1000.shape[0], dtype=np.int)*2
    # 把 array 和 label 串在一起
    image_array = np.concatenate((array_100, array_500, array_1000), axis=0)
    image_label = np.concatenate((label_100, label_500, label_1000), axis=0)
    return image_array, image_label

def makeBasicModel(img_height, img_width):
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
        MaxPooling2D(2, 2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(3)
    ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model

def makeDropoutModel(img_height, img_width):
    model = Sequential([
        Conv2D(32, 5, padding='same', activation='relu', input_shape=(img_height, img_width ,3)),
        MaxPooling2D(2, 2),
        SpatialDropout2D(0.25),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        SpatialDropout2D(0.25),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        SpatialDropout2D(0.25),
        Flatten(),
        Dropout(0.25),
        Dense(128, activation='relu'),
        Dense(3)
    ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model

def drawHistoryAccAndLoss(epochs, history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    
    epochs_range = range(epochs)
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

if __name__ == "__main__":
    origin_folder = 'origin_data'
    augmentation_folder = 'augmentation_data'
    EPOCHS = 25
    IMG_HEIGHT = 128
    IMG_WIDTH = 128

    # 讀取原始資料
    original_images, original_labels = readMoneyData(origin_folder)
    # 把原始資料分割成訓練資料和測試資料
    original_train_images, original_test_images, original_train_labels, original_test_labels = train_test_split(original_images, original_labels, train_size=0.5, random_state=42)
    # 讀取增量資料
    augmentation_images, augmentation_labels = readMoneyData(augmentation_folder)

    # 最基本的模型並使用原始資料訓練
    basic_model = makeBasicModel(img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    basic_history = basic_model.fit(original_train_images, original_train_labels, epochs=EPOCHS, validation_data=(original_test_images, original_test_labels))
    drawHistoryAccAndLoss(epochs=EPOCHS, history=basic_history)
    
    # 加上 Dropout的模型並使用增量資料訓練
    dropout_model = makeDropoutModel(img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    dropout_aug_history = dropout_model.fit(augmentation_images, augmentation_labels, epochs=EPOCHS, validation_data=(original_test_images, original_test_labels))
    drawHistoryAccAndLoss(epochs=EPOCHS, history=dropout_aug_history)