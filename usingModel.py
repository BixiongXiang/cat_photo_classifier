import tensorflow as tf
from tensorflow import keras as ks
import h5py
import dataLoader
import numpy as np
from matplotlib import pyplot as plt

x_train, y_train, x_test, y_test = dataLoader.load_data()

plt.imshow(x_train[8])
plt.show()

x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]

model = ks.models.load_model("pqModel.hdf5")

# predict = model.predict(x_test)
# predict = model.predict(x_train[10].reshape(1, 200, 200))
# predict = model.predict(x_train[8].reshape(1, 200, 200, 1))
predict = model.predict(x_test)

print("predict: ", predict)

