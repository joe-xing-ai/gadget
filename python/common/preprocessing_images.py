import os
import cv2 as cv
import numpy as np
import logging
from efficientnet.preprocessing import center_crop_and_resize


def preprocessing(src_folder, des_folder):
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.endswith(".jpg"):
                image_file = os.path.join(root, file)
                frame = cv.imread(image_file)
                image = center_crop_and_resize(frame, 224).astype(np.uint8)
                logging.info(image.shape)
                path_write = os.path.join(des_folder, file)
                logging.info(path_write)
                cv.imwrite(path_write, image)


def main():
    preprocessing("../data/cars196", "../data/cars196_new/")


if __name__ == "__main__":
    main()
