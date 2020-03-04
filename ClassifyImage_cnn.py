import tensorflow as tf
from tensorflow import keras as ks
import h5py
import imageTransfer
import dataLoader
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

use_TestSet = True

model = ks.models.load_model("pqModel.hdf5")

if use_TestSet:
    x_train_ori, y_train, x_test_ori, y_test = dataLoader.load_data()

    x_train = x_train_ori[:, :, :, np.newaxis]
    x_test = x_test_ori[:, :, :, np.newaxis]

    predict = model.predict(x_test)

else:
    source = ""
    dest = ""
    imageTransfer.image_Transfer(source, dest)

    source2 = ""
    img_ori = dataLoader.image_loader(source2)

    img = img_ori[:, :, :, np.newaxis]

    predict = model.predict(img)


predict = np.argmax(predict, axis=1)

true_path = "testModel/result/true/"
false_path = "testModel/result/false/"

for i in range(len(predict)):
    pred = predict[i]

    if pred == 0:
        print(str(i) + " not cat")
        plt.imsave(false_path + str(i) + ".png", x_test_ori[i])

    else:
        print(str(i) + " is cat")
        plt.imsave(true_path + str(i) + ".png", x_test_ori[i])
