from PIL import Image as img
import glob
import os

ori_path_true = "data/ori/true/"
ori_path_false = "data/ori/false/"

trans_path_true = "data/transfered/true/"
trans_path_false = "data/transfered/false/"


def image_Transfer(source, dest):

    count = 0
    for file in glob.glob(source + "*.jpg"):

        # im = img.open(ori_path_true + "vertical.jpg", 'r')
        im = img.open(file)

        if im.size[0] < im.size[1]:
            im = im.rotate(270, expand=True)

        # img size 100 x 100
        im.thumbnail((400, 200))
        im = im.convert('L')
        im = im.crop((100, 0, 300, 200))

        save_path = os.path.join(dest, os.path.split(file)[1])
        im.save(save_path, "JPEG")
        count += 1

        if count % 50 == 0:
            print("transfered: ", count)

    print("total transfered: ", count, ", path: ", dest)

if __name__ == "__main__":
    print("transfer true")
    image_Transfer(ori_path_true, trans_path_true)

    print("transfer false")
    image_Transfer(ori_path_false, trans_path_false)

