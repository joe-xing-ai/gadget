import os
import glob
import logging
import pandas as pd
import matplotlib.pyplot as plt

from efficientnet.tfkeras import EfficientNetB0
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input

from tensorflow.keras import optimizers
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from common.constants import *


def train_perception():
    batch_size = 48
    width = EFFICIENTNET_IMAGE_SIZE
    height = EFFICIENTNET_IMAGE_SIZE
    epochs = 5
    dropout_rate = 0.2
    input_shape = (height, width, 3)
    list_names = LIST_CARS196_DATA_SCHEMA

    image_folder_train = IMAGE_FOLDER_TRAIN
    image_folder_test = IMAGE_FOLDER_TEST
    label_file_path_train = LABEL_FILE_PATH_TRAIN
    label_file_path_test = LABEL_FILE_PATH_TEST
    class_name_path = CLASS_NAME_PATH

    train_images = glob.glob(os.path.join(image_folder_train, '*.jpg'))
    test_images = glob.glob(os.path.join(image_folder_test, '*.jpg'))
    num_train = len(train_images)
    num_test = len(test_images)
    logging.info("number of training images: %s and number of test images: %s" % (num_train, num_test))

    # conv_base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    conv_base = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")

    df_train = pd.read_csv(label_file_path_train, names=list_names)
    df_test = pd.read_csv(label_file_path_test, names=list_names)

    df_names = pd.read_csv(class_name_path, names=["car_type"])
    df_names.insert(1, "class_id", df_names.index)
    num_classes = len(df_names)
    class_names = df_names["car_type"].tolist()

    df_train = pd.merge(df_train, df_names, left_on="class", right_on="class_id")
    df_test = pd.merge(df_test, df_names, left_on="class", right_on="class_id")

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_dataframe(dataframe=df_train, directory=image_folder_train,
                                                        target_size=(height, width), batch_size=batch_size,
                                                        class_mode="categorical", x_col="file_name", y_col="car_type",
                                                        classes=class_names)
    test_generator = test_datagen.flow_from_dataframe(dataframe=df_test, directory=image_folder_test,
                                                        target_size=(height, width), batch_size=batch_size,
                                                        class_mode="categorical", x_col="file_name", y_col="car_type",
                                                      classes=class_names)

    model = models.Sequential()
    model.add(conv_base)
    # model.add(layers.GlobalMaxPooling2D(name="gap"))
    # model.add(layers.Flatten(name="flatten"))

    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate, name="dropout_out"))

    # model.add(layers.Dense(256, activation='relu', name="fc1"))

    model.add(layers.Dense(num_classes, activation='softmax', name="fc_out"))
    model.summary()

    logging.info('Number of trainable layers before freezing the conv base: %s' % len(model.trainable_weights))

    conv_base.trainable = False

    logging.info('Number of trainable layers after freezing the conv base: %s' % len(model.trainable_weights))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=['acc'])

    history = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=num_test // batch_size,
        verbose=1,
        use_multiprocessing=False,
        workers=4)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_x = range(len(acc))

    plt.plot(epochs_x, acc, 'bo', label='Training acc')
    plt.plot(epochs_x, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs_x, loss, 'bo', label='Training loss')
    plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
