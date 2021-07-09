import os
import cv2 as cv
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utility.embedding_search_utility import read_embeddings, build_index, search

BLACK = (0, 0, 0)
BLUE_BGR = (255, 0, 0)


def visualize(embedding_folder, image_folder):
    dataset_names_downloaded = ["cars196", "food101", "stanford_dogs"]
    colors = ["green", "red", "blue"]
    emb_dimen_reduction = [[] for _ in range(len(dataset_names_downloaded))]
    id_to_name, name_to_id, embeddings = read_embeddings(embedding_folder)
    id_to_category = {}
    embeddings_after_dimen_reduction = TSNE(n_components=3).fit_transform(embeddings)

    for file_name, index in name_to_id.items():
        for index_dataset, dataset in enumerate(dataset_names_downloaded):
            image_path = os.path.join(os.path.join(image_folder, dataset), file_name)
            image_path = "%s.jpg" % image_path
            if os.path.exists(image_path):
                id_to_category[index] = index_dataset
                emb_dimen_reduction[index_dataset].append(embeddings_after_dimen_reduction[index])

    num_images = len(id_to_name)
    logging.info("there are total %s images that have embedding vectors that has shape %s" %
                 (num_images, embeddings.shape))

    f = plt.figure(figsize=(9, 6))
    ax1 = plt.axes(projection='3d')

    for i, emd in enumerate(emb_dimen_reduction):
        emb_matrix = np.array(emd)
        ax1.scatter3D(emb_matrix[:, 0], emb_matrix[:, 1], emb_matrix[:, 2], c=colors[i],
                      label=dataset_names_downloaded[i])

    ax1.set_xlabel("Image Embedding X", fontsize=10)
    ax1.set_ylabel("Image Embedding Y", fontsize=10)
    ax1.set_zlabel("Image Embedding Z", fontsize=10)
    plt.legend(loc="upper left")
    plt.show()
