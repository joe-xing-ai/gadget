import optparse
import logging


def main():
    parser = optparse.OptionParser()
    parser.add_option("--image_folder", action="store", default="./data/images", help="Folder for tfrecords")
    parser.add_option("--tfrecord_folder", action="store", default="./data/tfrecords", help="Folder for tfrecords")
    parser.add_option('--embedding_folder', action="store", default="./data/embeddings", help="Folder for embeddings")
    parser.add_option('--numpy_data_folder', action="store", default="./data/numpy_data/", help=None)
    parser.add_option('--search', action="store_true", default=False, help=None)
    parser.add_option('--initialize', action="store_true", default=False, help=None)

    options, args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

    if options.initialize:
        from utility.download_data import save_examples_to_folder
        save_examples_to_folder(options.image_folder)

    elif options.search:
        from utility.embedding_search_utility import random_search, embeddings_to_numpy
        random_search(options.embedding_folder)
        embeddings_to_numpy(options.embedding_folder, options.numpy_data_folder)
    else:
        from utility.inference_utility import run_inference
        run_inference(options.tfrecord_folder, options.embedding_folder)


if __name__ == '__main__':
    main()



