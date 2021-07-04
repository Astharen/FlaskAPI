from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model, save_model, Model
from sklearn import svm
import numpy as np
import joblib
# to retrieve and send back data

def get_model():
    num_classes = 10
    input_shape = (28, 28, 1)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = x_train.reshape((x_train.shape[0], input_shape[0], input_shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], input_shape[0], input_shape[1], 1))
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")


    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dense(10, activation='softmax'))
	# compile model
    opt = keras.optimizers.Adam(learning_rate=0.01)
    print(x_train.shape)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    new_order = np.random.permutation(range(x_train.shape[0]))
    x_train = x_train[new_order]
    y_train = y_train[new_order]

    model.fit(x_train, y_train, epochs=15, batch_size=256, validation_data=(x_test, y_test))

    return model


# model = get_model()

# model.save('model.h5')


def get_model2():
    num_classes = 10
    input_shape = (28, 28, 1)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = x_train.reshape((x_train.shape[0], input_shape[0]*input_shape[1]))
    x_test = x_test.reshape((x_test.shape[0], input_shape[0]*input_shape[1]))
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")


    clf = svm.SVC()
    clf.fit(x_train, y_train)

    return clf


model2 = get_model2()

joblib.dump(model2,'model.pkl')