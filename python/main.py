import optparse
import logging

from utility.download_data import save_examples_to_folder
from utility.inference_utility import generate_tfrecord, generate_embeddings
from UI.ui_controller import render
from UI.ui_controller_embedding_distribution import visualize


def main():
    """
    this is the main driver for several main use cases:
    - (a) Visualizing image embedding vector distributions after dimensionality reduction, this is for sanity check
        of the generated image embedding vectors

        python main.py --demo_visualize --image_folder ./data

    - (b) Image search using similarity metrics of image embedding vectors, this serves as another sanity check

        python main.py --demo_search

    - (c) Image algebra, TBD

    - (d) Reinforcement Learning / Imitation Leaning based exploration of DNN performance, TBD

    :return:
    """
    parser = optparse.OptionParser()
    parser.add_option("--image_folder", action="store", default="./data/images", help="Folder for tfrecords")
    parser.add_option("--tfrecord_folder", action="store", default="./data/tfrecords", help="Folder for tfrecords")
    parser.add_option('--embedding_folder', action="store", default="./data/embeddings", help="Folder for embeddings")
    parser.add_option('--numpy_data_folder', action="store", default="./data/numpy_data/", help=None)
    parser.add_option('--download_data', action="store_true", default=False,
                      help="this helps you to download the sample dataset provided by tensorflow")
    parser.add_option('--dataset_name', action="store", default="food101", help="cars196, food101, stanford_dogs etc.")
    parser.add_option('--dataset_num_images', action="store", default=1000, help=None)
    parser.add_option('--preprocessing', action="store_true", default=False, help=None)
    parser.add_option('--demo_search', action="store_true", default=False, help=None)
    parser.add_option('--gif', action="store_true", default=False, help=None)
    parser.add_option('--gif_folder', action="store", default="./artifacts/", help=None)
    parser.add_option('--gif_name', action="store", default="image_search", help=None)
    parser.add_option('--demo_visualize', action="store_true", default=False, help=None)

    options, args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

    if options.download_data:
        save_examples_to_folder(options.image_folder, options.dataset_name, int(options.dataset_num_images))
    elif options.preprocessing:
        generate_tfrecord(options.image_folder, options.tfrecord_folder)
        generate_embeddings(options.tfrecord_folder, options.embedding_folder)
    elif options.demo_visualize:
        visualize(options.embedding_folder, options.image_folder)
    elif options.demo_search:
        render(options.embedding_folder, options.image_folder, options.gif, options.gif_folder, options.gif_name)


if __name__ == '__main__':
    main()



