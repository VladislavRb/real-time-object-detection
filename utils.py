import tensorflow as tf
from constants import constants


def clip(x, min_value=0, max_value=1):
    return min(max(x, min_value), max_value)


def bbox_attr(data: tf.Tensor, i) -> tf.Tensor:
    attr_start = constants.c + i
    return data[..., attr_start::5]


def scale_bbox_coord(coord, center, zoom):
    return ((coord - center) * zoom) + center


def image_shape(images, i_shape, target_type):
    if i_shape is None:
        if not isinstance(images, tf.RaggedTensor):
            i_shape = tf.shape(images)
            height, width = i_shape[1], i_shape[2]
        else:
            height = tf.reshape(images.row_lengths(), (-1, 1))
            width = tf.reshape(tf.maximum(images.row_lengths(axis=2), 1), (-1, 1))
            height = tf.expand_dims(height, axis=-1)
            width = tf.expand_dims(width, axis=-1)
    else:
        height, width = i_shape[0], i_shape[1]
    return tf.cast(height, target_type), tf.cast(width, target_type)
