import tensorflow as tf
from tensorflow import keras as ks
import numpy as np
from matplotlib import pyplot as plt

from dataLoader import load_data

import time

time_start = time.time()

# choose data set
useMNIST = False

if useMNIST is True:
    fashion_mnist = ks.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    classNum = 10
    loss_func = "sparse_categorical_crossentropy"
else:
    x_train, y_train, x_test, y_test = load_data()
    classNum = 1
    loss_func = "binary_crossentropy"


# normalize to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# construct model
# sequential way
useCNN = True

if not useCNN:
    input_shape = (x_train[0].shape[0], x_train[0].shape[1])

    model = ks.Sequential([
        ks.layers.Flatten(input_shape=input_shape, name='input'),
        ks.layers.Dense(128, activation='tanh', kernel_initializer='random_uniform', name='hidden1'),
        ks.layers.Dense(20, activation='tanh', kernel_initializer='random_uniform', name='hidden2'),
        ks.layers.Dense(classNum, activation='sigmoid', kernel_initializer='random_uniform', name='output')
    ])
else:
    input_shape = (x_train[0].shape[0], x_train[0].shape[1], 1)

    x_train = x_train[:, :, :, np.newaxis]
    x_test = x_test[:, :, :, np.newaxis]

    model = ks.Sequential()
    model.add(ks.layers.Conv2D(25, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(ks.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(ks.layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(ks.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(1000, activation='relu'))
    model.add(ks.layers.Dense(classNum + 1, activation='softmax'))
    # 2 class, 0 for not cat, 1 for is a cat, out put is one hot

print(x_train.shape)

# functional API way
# nn_input = ks.layers.Input(shape=(x_train[0].shape[0], x_train[0].shape[1]), name='input')
# h1 = ks.layers.Dense(100, activation='relu', name='hidden1')(nn_input)
# output = ks.layers.Dense(2, activation='softmax', name='output')(h1)
#
# model = ks.Model(inputs=nn_input, outputs=output)

opt_adam = ks.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
opt_sgd = ks.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

# mi
if not useCNN:
    model.compile(optimizer='adam', loss=loss_func, metrics=['accuracy'])
else:
    model.compile(loss=ks.losses.sparse_categorical_crossentropy,
                  optimizer=ks.optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])
# ks.utils.plot_model(model, show_shapes=True, show_layer_names=True)
print(model.summary())

# first predict
# predict = model.predict(x_test[0].reshape(1, 100, 56))

# print("predict: ", predict)

# try call bakcs
# mcb = ks.callbacks.ModelCheckpoint()

# start training
train_results = model.fit(x_train, y_train, batch_size=30, epochs=40, validation_split=0.2, shuffle=True)


# print('train history dict:', train_results.history)
# print time
time_end = time.time()
print('training total time cost', time_end - time_start, 's')

# plot learning curv

loss = train_results.history['loss']
val_loss = train_results.history['val_loss']
acc = train_results.history['accuracy']
val_acc = train_results.history['val_accuracy']

epochs = np.arange(len(loss))

plt.figure()
plt.plot(epochs, acc, label='acc')
plt.plot(epochs, val_acc, label='val_acc')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#test model
print("*** testing the model ***")
test_result = model.evaluate(x_test, y_test)

print('test loss, test acc:', test_result)

# save model
model.save("pqModel.hdf5")
print("pqModel.hdf5 model saved!")
