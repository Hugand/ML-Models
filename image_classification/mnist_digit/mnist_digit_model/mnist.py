import tempfile, os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist
from PIL import Image
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

#
#   MNIST handwritten digits recon
#

def main():
    mnist_train()

def mnist_train():
    # Load the data 28x28 imgs
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    num_classes = 10

    print(x_train.shape, y_train.shape)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Normalize data
    x_train = x_train / 255.
    x_test = x_test / 255.

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    model = Sequential()

    model.add(Dense(units=128, input_shape=(784,), activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # model = keras.models.load_model('./model/1')

    model.summary()

    # Train
    batch_size = 512
    epochs = 10
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test Loss: {}, Test Accuracy: {}".format(test_loss, test_acc))

    version = 1
    export_path = os.path.join("./model", str(version))
    print('export_path = {}\n'.format(export_path))
    
    keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )

    print("done")

main()
