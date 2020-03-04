from PIL import Image as img
import h5py
import glob
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    path = "data/transfered/PQ_image.hdf5"
    with h5py.File(path, "r") as hd:
        x_train = hd["x_train"][:]
        y_train = hd["y_train"][:]

        x_test = hd["x_test"][:]
        y_test = hd["y_test"][:]

    print("load data finished!")
    hd.close()
    return x_train, y_train, x_test, y_test

def transfer_iamge2hdf5():
    path = "data/transfered/PQ_image.hdf5"
    trans_path_true = "data/transfered/true/"
    trans_path_false = "data/transfered/false/"

    x_true = image_loader(trans_path_true)
    y_true = np.ones(len(x_true))

    x_false = image_loader(trans_path_false)
    y_false = 0 * np.ones(len(x_false))

    x = np.concatenate((x_true, x_false), axis=0)
    y = np.concatenate((y_true, y_false), axis=0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    hd = h5py.File(path, 'w')

    hd.create_dataset("x_train", data=x_train)
    hd.create_dataset("y_train", data=y_train)
    hd.create_dataset("x_test", data=x_test)
    hd.create_dataset("y_test", data=y_test)

    hd.close()

    print("hdf5 file saved!")
    print("transfer image to data set finished")

# only load jpg files
def image_loader(source):
    result = None

    counter = 0
    for file in glob.glob(source + "*.jpg"):
        im = img.open(file)

        if im.size[0] != 200 or im.size[0] != im.size[1]:
            print("picture ", counter + 1, " load failed")

        np_img = np.array(im.getdata()).reshape(1, im.size[0], im.size[1])
        if result is not None:
            result = np.concatenate((result, np_img), axis=0)
        else:
            result = np_img
        counter += 1

        if counter % 50 == 0:
            print("transfered: ", counter)

    print("loaded")
    return result


# test script

if __name__ == "__main__":
    transfer_iamge2hdf5()

# x_train, y_train, x_test, y_test = load_data()


