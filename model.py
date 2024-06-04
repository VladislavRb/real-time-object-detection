import keras.layers as layers
import keras.models as models
from constants import constants


def _modify_with_batchnorm(model_layers):
    for i in range(len(model_layers) - 1, -1, -1):
        if 'conv2d' in model_layers[i].name.lower():
            model_layers.insert(i + 1, layers.BatchNormalization())


def create_full_model(add_batch_norm=True):
    input_shape = (constants.image_resolution, constants.image_resolution, 3)
    model_layers = [
        layers.Conv2D(64, kernel_size=7, strides=2, padding='same', input_shape=input_shape),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        layers.Conv2D(192, kernel_size=3, padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        layers.Conv2D(128, kernel_size=1),
        layers.LeakyReLU(alpha=0.1),
        layers.Conv2D(256, kernel_size=3, padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.Conv2D(256, kernel_size=1),
        layers.LeakyReLU(alpha=0.1),
        layers.Conv2D(512, kernel_size=3, padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
    ]

    for i in range(4):
        model_layers += [
            layers.Conv2D(256, kernel_size=1),
            layers.Conv2D(512, kernel_size=3, padding='same'),
            layers.LeakyReLU(alpha=0.1)
        ]
    model_layers += [
        layers.Conv2D(512, kernel_size=1),
        layers.Conv2D(1024, kernel_size=3, padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
    ]

    for i in range(2):
        model_layers += [
            layers.Conv2D(512, kernel_size=1),
            layers.Conv2D(1024, kernel_size=3, padding='same'),
            layers.LeakyReLU(alpha=0.1)
        ]
    model_layers += [
        layers.Conv2D(1024, kernel_size=3, padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.Conv2D(1024, kernel_size=3, strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.1),
    ]

    for _ in range(2):
        model_layers += [
            layers.Conv2D(1024, kernel_size=3, padding='same'),
            layers.LeakyReLU(alpha=0.1)
        ]

    model_layers += [
        layers.Flatten(),
        layers.Dense(4096),
        layers.Dropout(0.3),
        layers.LeakyReLU(alpha=0.1),
        layers.Dense(constants.s * constants.s * constants.cell_predictions_amount),
        layers.Reshape((constants.s, constants.s, constants.cell_predictions_amount))
    ]

    if add_batch_norm:
        _modify_with_batchnorm(model_layers)

    return models.Sequential(model_layers)


def create_light_model(add_batch_norm=True):
    input_shape = (constants.image_resolution, constants.image_resolution, 3)
    model_layers = [
        layers.Conv2D(32, kernel_size=7, strides=2, padding='same', input_shape=input_shape),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        layers.Conv2D(96, kernel_size=3, padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

        layers.Conv2D(64, kernel_size=1),
        layers.LeakyReLU(alpha=0.1),
        layers.Conv2D(128, kernel_size=3, padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.Conv2D(128, kernel_size=1),
        layers.LeakyReLU(alpha=0.1),
        layers.Conv2D(256, kernel_size=3, padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
    ]

    for i in range(2):
        model_layers += [
            layers.Conv2D(32, kernel_size=1),
            layers.Conv2D(64, kernel_size=3, padding='same'),
            layers.LeakyReLU(alpha=0.1)
        ]
    model_layers += [
        layers.Conv2D(32, kernel_size=1),
        layers.Conv2D(64, kernel_size=3, padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
    ]

    for i in range(2):
        model_layers += [
            layers.Conv2D(32, kernel_size=1),
            layers.Conv2D(64, kernel_size=3, padding='same'),
            layers.LeakyReLU(alpha=0.1)
        ]
    model_layers += [
        layers.Conv2D(64, kernel_size=3, padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.Conv2D(64, kernel_size=3, strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.1),
    ]

    for _ in range(2):
        model_layers += [
            layers.Conv2D(128, kernel_size=3, padding='same'),
            layers.LeakyReLU(alpha=0.1)
        ]

    model_layers += [
        layers.Flatten(),
        layers.Dense(512),
        layers.Dropout(0.3),
        layers.LeakyReLU(alpha=0.1),
        layers.Dense(constants.s * constants.s * constants.cell_predictions_amount),
        layers.Reshape((constants.s, constants.s, constants.cell_predictions_amount))
    ]

    if add_batch_norm:
        _modify_with_batchnorm(model_layers)

    return models.Sequential(model_layers)


def create_light_model_v2(add_batch_norm=True):
    def default_conv(filters, kernel_size):
        return layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)

    input_shape = (constants.image_resolution, constants.image_resolution, 3)
    model_layers = [
        layers.Conv2D(32, kernel_size=3, padding='same', use_bias=False, input_shape=input_shape),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPool2D(pool_size=(2, 2)),

        default_conv(64, 3),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPool2D(pool_size=(2, 2)),

        default_conv(128, 3),
        layers.LeakyReLU(alpha=0.1),
        default_conv(64, 1),
        layers.LeakyReLU(alpha=0.1),
        default_conv(128, 3),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        default_conv(256, 3),
        layers.LeakyReLU(alpha=0.1),
        default_conv(128, 1),
        layers.LeakyReLU(alpha=0.1),
        default_conv(256, 3),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPool2D(pool_size=(2, 2)),
        default_conv(512, 3),
        layers.LeakyReLU(alpha=0.1),
        default_conv(256, 1),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPool2D(pool_size=(2, 2)),
        default_conv(1024, 3),
        layers.LeakyReLU(alpha=0.1),
        default_conv(512, 1),
        layers.LeakyReLU(alpha=0.1),
        default_conv(1024, 3),
        layers.LeakyReLU(alpha=0.1),
        default_conv(512, 1),
        layers.LeakyReLU(alpha=0.1),
        default_conv(1024, 3),
        layers.LeakyReLU(alpha=0.1),
        default_conv(512, 3),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPool2D(pool_size=(2, 2)),
    ]

    model_layers += [
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(constants.s * constants.s * constants.cell_predictions_amount),
        layers.Reshape((constants.s, constants.s, constants.cell_predictions_amount))
    ]

    if add_batch_norm:
        _modify_with_batchnorm(model_layers)

    return models.Sequential(layers=model_layers)
