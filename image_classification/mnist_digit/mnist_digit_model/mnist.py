import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
# from tf.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import tempfile, os
import matplotlib.pyplot as plt

#
#   MNIST handwritten digits recon
#

def main():
    # Load the data 28x28 imgs
    mnist_dataset, mnist_info = tfds.load(name='mnist', as_supervised=True, with_info=True)

    print(mnist_dataset)
    print(mnist_info)

    # for ex in mnist_dataset.take(1):
    #     print(ex["label"])

    # for example in ds:  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`
    #     print(list(example.keys()))
    #     image = example["image"]
    #     label = example["label"]
    #     print(image.shape, label)


    # Unpack the data
    mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

    num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
    num_validation_samples = tf.cast(num_validation_samples, tf.int64)
    num_test_sample = mnist_info.splits['test'].num_examples
    num_test_sample = tf.cast(num_test_sample, tf.int64)

    # Scale image
    scaled_train_and_validation_data = mnist_train.map(scale)
    test_data = mnist_test.map(scale)

    # Shuffle data
    BUFFER_SIZE = 10000
    shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)
    validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
    train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

    # Batch the datasets
    BATCH_SIZE = 100
    train_data = train_data.batch(BATCH_SIZE)
    validation_data = validation_data.batch(num_validation_samples)
    test_data = test_data.batch(num_test_sample)

    # Define model
    output_size = 10
    hidden_layer_size = 50

    model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape = (28, 28, 1)),
                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                                tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                                tf.keras.layers.Dense(output_size, activation='softmax')])

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, amsgrad=True)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    NUM_EPOCHS = 20
    BATCH_SIZE = 100
    EARLY_STOP = tf.keras.callbacks.EarlyStopping(patience=2)

    model.fit(train_data,
            epochs=NUM_EPOCHS,
            callbacks=[EARLY_STOP],
            validation_data=validation_data,
            verbose=2)

    test_loss, test_accuracy = model.evaluate(test_data)

    print('Test loss: {0:.2f}'.format(test_loss))
    print('Test accuracy: {0:.2f}'.format(test_accuracy*100.))

    # img_w = img_h = 28
    # img = tf.keras.preprocessing.image.load_img('./mnist_2.png')
    # x = tf.keras.preprocessing.image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x /= 255

    # images = np.vstack([x])
    # classes = model.predict(img)
    # print(classes)
    
    MODEL_DIR = tempfile.gettempdir()
    print(MODEL_DIR)
    version = 1
    export_path = os.path.join("./model", str(version))
    print('export_path = {}\n'.format(export_path))
    

    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )

    print('\nSaved model:')


    print("done")

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label

main()
