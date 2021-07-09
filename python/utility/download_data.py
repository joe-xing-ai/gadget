from pathlib import Path

from tensorflow_datasets.core import dataset_utils
from tensorflow_datasets.core import features as features_lib
from PIL import Image
from efficientnet.preprocessing import center_crop_and_resize
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
import logging


def save_examples(ds_info, ds, num_examples=10, folder=".", image_key=None):
    if not image_key:
        # Infer the image and label keys
        image_keys = [k for k, feature in ds_info.features.items() if isinstance(feature, features_lib.Image)]

        if not image_keys:
            raise ValueError(
                "Visualisation not supported for dataset `{}`. Was not able to "
                "auto-infer image.".format(ds_info.name)
            )

        if len(image_keys) > 1:
            raise ValueError(
                "Multiple image features detected in the dataset. Using the first one. You can "
                "use `image_key` argument to override. Images detected: %s" % (",".join(image_keys))
            )

        image_key = image_keys[0]

    label_keys = [k for k, feature in ds_info.features.items() if isinstance(feature, features_lib.ClassLabel)]

    label_key = label_keys[0] if len(label_keys) == 1 else None
    if not label_key:
        logging.info("Was not able to auto-infer label.")

    examples = list(dataset_utils.as_numpy(ds.take(num_examples)))

    for i, ex in enumerate(examples):
        if not isinstance(ex, dict):
            raise ValueError(
                "tfds.show_examples requires examples as `dict`, with the same "
                "structure as `ds_info.features`. It is currently not compatible "
                "with `as_supervised=True`. Received: {}".format(type(ex))
            )

        # Plot the image
        image = ex[image_key]
        if len(image.shape) != 3:
            raise ValueError(
                "Image dimension should be 3. tfds.show_examples does not support " "batched examples or video."
            )
        _, _, c = image.shape
        if c == 1:
            image = image.reshape(image.shape[:2])
        image = center_crop_and_resize(image, 224).astype(np.uint8)
        im = Image.fromarray(image)
        if label_key:
            label = ex[label_key]
            label_str = ds_info.features[label_key].int2str(label).replace("/", "_")
        else:
            label_str = ""
        im.save(f"{folder}/image_{label_str}_{i}.jpg")


def save_examples_to_folder(output_folder, dataset, images_count):
    ds, ds_info = tfds.load(dataset, split="train", with_info=True)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    save_examples(ds_info, ds, images_count, output_folder)


def main():
    save_examples_to_folder("../data/images/", "cars196", 1000)


if __name__ == "__main__":
    main()
