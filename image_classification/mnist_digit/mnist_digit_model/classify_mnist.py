import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
# from tf.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import tempfile, os
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# model = tf.keras.models.load_model('./model/1')

# def main():
#     img_name = ""

#     while(img_name != "sair"):
#         img_name = input("=> ")
#         classify_mnist("mnist_"+img_name+".jpg")

def classify_mnist(img_path, model_path):
    model = tf.keras.models.load_model(model_path)

    print(img_path)
    
    image = tf.keras.preprocessing.image.load_img(img_path, color_mode="grayscale")
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    input_arr /= 255.
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    prediction = probability_model.predict(input_arr)

    prediction_result = np.argmax(prediction[0])

    print(prediction)
    print("[ RESULT ] => "+str(prediction_result))

    print("## DONE ##")

    return prediction_result

def parse_result(res):
    max_pos = 0

    for i in range(0, len(res)):
        if res[i] > res[max_pos]:
            max_pos = i

    return max_pos

# main()