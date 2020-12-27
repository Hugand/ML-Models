
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist
from PIL import Image
import numpy as np


# def main():
#     classify_mnist("../media/test_fyDmldj.png", "./model/1")

def classify_mnist(img_path, model_path):
    model = keras.models.load_model(model_path)
    print(img_path)

    image = keras.preprocessing.image.load_img(
        img_path,
        color_mode="grayscale",
        target_size=(28, 28))

    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = input_arr.reshape((1,784))
    input_arr = input_arr / 255.
    probability_model = keras.Sequential([model, keras.layers.Softmax()])
    prediction = probability_model.predict(input_arr)

    prediction_result = np.argmax(prediction[0])

    print(prediction)
    print("[ RESULT ] => "+str(prediction_result))

    return prediction_result

# main()