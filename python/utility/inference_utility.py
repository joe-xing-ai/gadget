import os
import logging
import platform
import tensorflow as tf
import time
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from efficientnet.tfkeras import EfficientNetB0


IMAGE_KEY = "image"
IMAGE_NAME_KEY = "image_name"
IMAGE_SUB_DIR_KEY = "image_sub_dir"


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(image, image_name, image_sub_dir):
    feature = {IMAGE_NAME_KEY: _bytes_feature(image_name),
               IMAGE_KEY: _bytes_feature(image),
               IMAGE_SUB_DIR_KEY: _bytes_feature(image_sub_dir)
               }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(image, image_name, image_sub_dir):
    """
    all tensor operators need to be used here
    :param image:
    :param image_name:
    :param image_sub_dir:
    :return:
    """
    tf_string = tf.py_function(serialize_example, (image, image_name, image_sub_dir), tf.string)
    return tf.reshape(tf_string, ())


def map_image_file_name(file_path):
    """
    all tensor operators need to used here
    :param file_path:
    :return:
    """
    # TODO for now only assume Linux and Windows are supported...
    os_name = platform.system()
    if "Windows" in os_name:
        parts = tf.strings.split(file_path, "\\")
    else:
        parts = tf.strings.split(file_path, "/")

    image_name = tf.strings.split(parts[-1], ".")[0]

    image_sub_directory = parts[-2]

    raw = tf.io.read_file(file_path)
    return raw, image_name, image_sub_directory


def write_to_tfrecord_file(shard_dataset, output_filename):
    image_ds = shard_dataset.map(map_image_file_name, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    serialized_features_dataset = image_ds.map(tf_serialize_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    writer = tf.data.experimental.TFRecordWriter(output_filename)
    writer.write(serialized_features_dataset)


def image_files_to_tfrecords(dataset, output_folder, num_shard):
    start_time = time.time()
    for shard_id in range(0, num_shard):
        shard_dataset = dataset.shard(num_shards=num_shard, index=shard_id)
        tfrecord_file_path = os.path.join(output_folder, "part-%03d.tfrecord" % shard_id)
        write_to_tfrecord_file(shard_dataset, tfrecord_file_path)
        logging.info("Shard %s processing takes %.1f s" % (shard_id, time.time() - start_time))
        start_time = time.time()


feature_description = {
    IMAGE_NAME_KEY: tf.io.FixedLenFeature([], tf.string),
    IMAGE_KEY: tf.io.FixedLenFeature([], tf.string),
    IMAGE_SUB_DIR_KEY: tf.io.FixedLenFeature([], tf.string)
}


def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


def preprocess_image(d):
    image_name = d[IMAGE_NAME_KEY]
    raw = d[IMAGE_KEY]
    image_sub_dir = d[IMAGE_SUB_DIR_KEY]

    image = tf.image.decode_jpeg(raw)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image, image_name, image_sub_dir


def read_tfrecord(filename):
    filenames = [filename]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    return (
        raw_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .apply(tf.data.experimental.ignore_errors())
    )


def tfrecords_to_embeddings(tfrecords_folder, output_folder, model, batch_size):
    tfrecords = [f.numpy().decode("utf-8")
                 for f in tf.data.Dataset.list_files(os.path.join(tfrecords_folder, "*.tfrecord"), shuffle=False)]

    start_time = time.time()
    for shard_id, tfrecord in enumerate(tfrecords):
        shard = read_tfrecord(tfrecord)
        embeddings = images_to_embeddings(model, shard, batch_size)

        logging.info("Shard %s computing image embedding takes %.1f s" % (shard_id, time.time() - start_time))

        parquet_file_path = os.path.join(output_folder, "part-%03d.parquet" % shard_id)
        save_embeddings_ds_to_parquet(embeddings, shard, parquet_file_path)

        logging.info("Shard %s saving embedding takes %.1f s" % (shard_id, time.time() - start_time))
        start_time = time.time()


def list_files(images_path):
    return tf.data.Dataset.list_files(images_path + "/*", shuffle=False).cache()


def images_to_embeddings(model, dataset, batch_size):
    return model.predict(dataset.batch(batch_size).map(lambda image_raw, image_name, image_sub_dir: image_raw),
                         verbose=1)


def save_embeddings_ds_to_parquet(embeddings, dataset, path):
    embeddings = pa.array(embeddings.tolist(), type=pa.list_(pa.float32()))

    image_names = pa.array(dataset.map(lambda image_raw, image_name, image_sub_dir: image_name).
                           as_numpy_iterator())

    image_sub_dirs = pa.array(dataset.map(lambda image_raw, image_name, image_sub_dir: image_sub_dir).
                              as_numpy_iterator())

    table = pa.Table.from_arrays([image_names, embeddings, image_sub_dirs],
                                 [IMAGE_NAME_KEY, "embedding", IMAGE_SUB_DIR_KEY])

    pq.write_table(table, path)


def generate_tfrecord(image_folder, output_folder, num_shards=10):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Note: the assumption is that each class of images will have a sub-directory under the image_folder
    combined_dataset = None
    for rootdir, dirs, files in os.walk(image_folder):
        for subdir in dirs:
            image_dir = os.path.join(rootdir, subdir)
            cached_dataset = list_files(image_dir)

            if combined_dataset is None:
                combined_dataset = cached_dataset
            else:
                combined_dataset = combined_dataset.concatenate(cached_dataset)

    image_files_to_tfrecords(combined_dataset, output_folder, num_shards)


def generate_embeddings(tfrecords_folder, output_folder, batch_size=1000):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
    tfrecords_to_embeddings(tfrecords_folder, output_folder, model, batch_size)


def main():
    generate_tfrecord("../data/images/", "../data/tfrecords/")


if __name__ == "__main__":
    main()
