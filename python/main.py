import optparse
import logging

from utility.download_data import save_examples_to_folder
from utility.inference_utility import generate_tfrecord, generate_embeddings
from UI.ui_controller import render
from UI.ui_controller_embedding_distribution import visualize
from train_rl import train
from train_perception import train_perception


def main():
    """
    this is the main driver for several main use cases:
    - (a) Visualizing image embedding vector distributions after dimensionality reduction, this is for sanity check
        of the generated image embedding vectors

        python main.py --demo_visualize --image_folder ../data/images

    - (b) Image search using similarity metrics of image embedding vectors, this serves as another sanity check

        python main.py --demo_search

    - (c) Image algebra, TBD

    - (d) Reinforcement Learning / Imitation Leaning based exploration of DNN performance, TBD

    :return:
    """
    parser = optparse.OptionParser()
    parser.add_option("--image_folder", action="store", default="../data/images", help="folder for raw images, each"
                                                                                       "type of image is a sub-folder")
    parser.add_option("--tfrecord_folder", action="store", default="../data/preprocessed/tfrecords",
                      help="folder for tfrecords")
    parser.add_option('--embedding_folder', action="store", default="../data/preprocessed/embeddings",
                      help="Folder for embeddings")
    parser.add_option('--download_data', action="store_true", default=False,
                      help="this helps you to download the sample dataset provided by tensorflow")
    parser.add_option('--dataset_name', action="store", default="food101", help="cars196, food101, stanford_dogs etc.")
    parser.add_option('--dataset_num_images', action="store", default=1000, help=None)
    parser.add_option('--preprocessing', action="store_true", default=False, help=None)
    parser.add_option('--demo_search', action="store_true", default=False, help="Search demo, sanity check")
    parser.add_option('--gif', action="store_true", default=False, help=None)
    parser.add_option('--gif_folder', action="store", default="./artifacts/", help=None)
    parser.add_option('--gif_name', action="store", default="image_search", help=None)
    parser.add_option('--demo_visualize', action="store_true", default=False, help="Visualize embedding vectors")
    parser.add_option('--train_rl', action="store_true", default=False,
                      help="Train RL agent to select MID (Most Informative Data)")
    parser.add_option('--env_name', action="store", default="gym_image_embedding:image_embedding-v0",
                      help="CartPole-v0 for unit-test, gym_image_embedding:image_embedding-v0 image exploration")
    parser.add_option('--train_visualize_fps', action="store", default=10., help=None)
    parser.add_option('--train_perception', action="store_true", default=False,
                      help="Train the baseline perception / CNN system (model agnostic in terms of CNN)")
    parser.add_option('--epochs_train_perception', action="store", default=10, help=None)

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
    elif options.train_rl:
        train(options.env_name, float(options.train_visualize_fps))
    elif options.train_perception:
        train_perception(int(options.epochs_train_perception))


if __name__ == '__main__':
    main()



