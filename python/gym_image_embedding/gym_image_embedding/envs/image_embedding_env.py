import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import cv2 as cv
import numpy as np
import logging
import random
import time
import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB0
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import pandas as pd

from utility.embedding_search_utility import read_embeddings, build_index, search
from common.constants import *

BLACK = (0, 0, 0)
BLUE_BGR = (255, 0, 0)


class Viewer(object):
    def __init__(self, env):
        self.env = env

        self.top_n = 5
        self.n_rows = 2
        self.width_unit_image = 300
        self.height_unit_image = 200
        self.width_hybrid_image = self.width_unit_image * self.top_n
        self.height_hybrid_image = self.height_unit_image * self.n_rows
        self.window_name = "RL in Image Embedding Space"
        self.window_w = self.width_hybrid_image
        self.window_h = self.height_hybrid_image

        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        w_start = 200
        h_start = 200
        cv.moveWindow(self.window_name, w_start, h_start)
        cv.resizeWindow(self.window_name, self.window_w, self.window_h)

        self.index_column_for_query = 2
        self.x_corner_1, self.y_corner_1 = 20, 40
        self.x_corner_2, self.y_corner_2 = 20, self.height_unit_image - 20

    def render(self):
        frame_aggregated = np.full((self.height_hybrid_image, self.width_hybrid_image, 3), BLACK, dtype=np.uint8)

        if self.env.test_frame is not None:
            frame = self.env.test_frame
            frame_query = cv.resize(frame, (self.width_unit_image, self.height_unit_image))
            w_s = self.index_column_for_query * self.width_unit_image
            w_e = w_s + self.width_unit_image
            r_s = 0
            r_e = self.height_unit_image
            frame_aggregated[r_s:r_e, w_s:w_e] = frame_query

            cv.putText(frame_aggregated, "Probe Image: %s" %
                       self.env.test_image_file_name,
                       (self.x_corner_1, self.y_corner_1), cv.FONT_HERSHEY_SIMPLEX, 0.8, BLUE_BGR, 2)

            cv.putText(frame_aggregated, "Ground Truth Car Type: %s" %
                       self.env.test_image_car_type,
                       (self.x_corner_2, self.y_corner_2), cv.FONT_HERSHEY_SIMPLEX, 0.8, BLUE_BGR, 2)

        cv.imshow(self.window_name, frame_aggregated)

    def close(self):
        cv.destroyAllWindows()


class ImageEmbeddingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.num_behaviors = 2
        self.action_space = spaces.Discrete(self.num_behaviors)

        self.num_input_feature = 3
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_input_feature,), dtype=np.float32)

        self.embedding_folder = "./data/embeddings"

        self.image_folder_train = IMAGE_FOLDER_TRAIN
        self.image_folder_test = IMAGE_FOLDER_TEST
        self.label_file_path_train = LABEL_FILE_PATH_TRAIN
        self.label_file_path_test = LABEL_FILE_PATH_TEST
        self.class_name_path = CLASS_NAME_PATH

        self.df_train = None
        self.df_test = None
        self.df_names = None
        self.parse_label_files()
        self.num_train_images = len(self.df_train.index)
        self.num_test_images = len(self.df_test.index)

        self.id_to_name, self.name_to_id, self.embeddings = read_embeddings(self.embedding_folder)
        self.index_image = build_index(self.embeddings)
        self.num_images = len(self.id_to_name)
        logging.info("there are total %s images that have embedding vectors that has shape %s" %
                     (self.num_images, self.embeddings.shape))

        self.images_for_gif = []
        self.model = EfficientNetB0(weights="imagenet", include_top=True, pooling="avg")
        self.viewer = Viewer(self)
        self.test_frame = None
        self.test_image_file_name = None
        self.test_image_car_type = None

    def parse_label_files(self):
        list_names = LIST_CARS196_DATA_SCHEMA
        self.df_train = pd.read_csv(self.label_file_path_train, names=list_names)
        self.df_test = pd.read_csv(self.label_file_path_test, names=list_names)
        self.df_names = pd.read_csv(self.class_name_path, names=["car_type"])

    def step(self, action):
        p = random.randint(0, self.num_test_images - 1)
        selected = self.df_test.loc[self.df_test.index == p]
        image_file_name = selected["file_name"].values[0]
        selected_class = selected["class"].values[0]
        selected_class_name = self.df_names.loc[self.df_names.index == selected_class]
        selected_class_name = selected_class_name["car_type"].values[0]
        logging.info("Testing image: %s with ground truth class name %s" % (image_file_name, selected_class_name))
        image_path = os.path.join(self.image_folder_test, image_file_name)

        if os.path.exists(image_path):
            frame = cv.imread(image_path)
            self.test_frame = frame
            self.test_image_file_name = image_file_name
            self.test_image_car_type = selected_class_name
            x = center_crop_and_resize(frame, image_size=EFFICIENTNET_IMAGE_SIZE)
            x = preprocess_input(x)
            frame_4d = np.expand_dims(x, axis=0).astype(np.float32)

            output = self.model.predict(frame_4d, verbose=1)

            classes = decode_predictions(output, top=5)

            print("joe classes", classes)

        observation_next = np.zeros((self.num_input_feature,), dtype=np.float32)
        reward = 1.
        done = True
        info = {}
        return observation_next, reward, done, info

    def reset(self):
        return np.zeros((self.num_input_feature,), dtype=np.float32)

    def render(self, mode='human'):
        self.viewer.render()

    def close(self):
        self.viewer.close()
