import os
import cv2 as cv
import random
import logging
import numpy as np
import imageio
import time

from utility.embedding_search_utility import read_embeddings, build_index, search

BLACK = (0, 0, 0)
BLUE_BGR = (255, 0, 0)


def render(embedding_folder, image_folder, gif, output_path):
    top_n = 5
    n_rows = 2
    width_unit_image = 300
    height_unit_image = 200
    width_hybrid_image = width_unit_image * top_n
    height_hybrid_image = height_unit_image * n_rows
    window_name = "Image Embedding Demo"
    window_w = width_hybrid_image
    window_h = height_hybrid_image

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    w_start = 200
    h_start = 200
    cv.moveWindow(window_name, w_start, h_start)
    cv.resizeWindow(window_name, window_w, window_h)

    index_column_for_query = 2
    x_corner_1, y_corner_1 = 20, 40
    x_corner_2, y_corner_2 = 20, height_unit_image - 20

    id_to_name, name_to_id, embeddings = read_embeddings(embedding_folder)
    index_image = build_index(embeddings)
    num_images = len(id_to_name)
    logging.info("there are total %s images that have embedding vectors that has shape %s" %
                 (num_images, embeddings.shape))

    images_for_gif = []

    while True:
        frame_aggregated = np.full((height_hybrid_image, width_hybrid_image, 3), BLACK, dtype=np.uint8)

        p = random.randint(0,  num_images - 1)
        image_file_name = id_to_name[p]
        logging.info("Query with key: %s" % image_file_name)

        image_path = os.path.join(image_folder, image_file_name)
        image_path = "%s.%s" % (image_path, "jpg")

        if os.path.exists(image_path):
            frame = cv.imread(image_path)
            frame_query = cv.resize(frame, (width_unit_image, height_unit_image))
            w_s = index_column_for_query * width_unit_image
            w_e = w_s + width_unit_image
            r_s = 0
            r_e = height_unit_image
            frame_aggregated[r_s:r_e, w_s:w_e] = frame_query

            cv.putText(frame_aggregated, "Query with Key: %s" % image_file_name,
                        (x_corner_1, y_corner_1), cv.FONT_HERSHEY_SIMPLEX, 0.8, BLUE_BGR, 2)

        start_time = time.time()
        results = search(index_image, id_to_name, embeddings[p], rank=top_n)
        query_latency = time.time() - start_time
        query_latency *= 1000.  # in units of ms

        for index, e in enumerate(results):
            dist, image_name = e
            image_path_result = os.path.join(image_folder, image_name)
            image_path_result = "%s.%s" % (image_path_result, "jpg")

            if os.path.exists(image_path_result):
                w_s = index * width_unit_image
                w_e = w_s + width_unit_image
                r_s = height_unit_image
                r_e = n_rows * height_unit_image
                frame = cv.imread(image_path_result)
                frame_result = cv.resize(frame, (width_unit_image, height_unit_image))
                frame_aggregated[r_s:r_e, w_s:w_e] = frame_result
                cv.putText(frame_aggregated, "%s: %.1f" % (image_name, dist),
                           (x_corner_2 + index * width_unit_image,
                            y_corner_2 + height_unit_image // 2), cv.FONT_HERSHEY_SIMPLEX, 0.6, BLUE_BGR, 2)

        cv.putText(frame_aggregated, "Query results:", (x_corner_2, y_corner_2), cv.FONT_HERSHEY_SIMPLEX,
                   0.8, BLUE_BGR, 2)

        cv.putText(frame_aggregated, "Query latency: %.1f [ms]" % query_latency,
                   (width_hybrid_image - 2 * width_unit_image, y_corner_2),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, BLUE_BGR, 2)

        cv.imshow(window_name, frame_aggregated)

        if gif:
            images_for_gif.append(cv.cvtColor(frame_aggregated, cv.COLOR_BGR2RGB))

        key = cv.waitKey(0)

        if key & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break

        elif key & 0xFF == ord('n'):
            continue

    if gif:
        imageio.mimsave(os.path.join(output_path, 'image_search.gif'),
                        images_for_gif, duration=1.0)
