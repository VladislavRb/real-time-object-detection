import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.ops.numpy_ops import np_config

from data_preprocessing import PreprocessingUnit
from utils import image_shape
from constants import constants


def _rel_yxyx_to_xyxy(boxes, images=None, i_shape=None):
    image_height, image_width = image_shape(images, i_shape, boxes.dtype)
    top, left, bottom, right = tf.split(boxes, 4, axis=-1)

    return tf.concat(
        [
            (left + right) * image_width * 0.5,
            (top + bottom) * image_height * 0.5,
            (right - left) * image_width,
            (bottom - top) * image_height
        ],
        axis=-1,
    )


def _format_inputs(boxes, images):
    boxes_rank = len(boxes.shape)
    if boxes_rank > 3:
        raise ValueError(
            "Expected len(boxes.shape)=2, or len(boxes.shape)=3, got "
            f"len(boxes.shape)={boxes_rank}"
        )
    boxes_includes_batch = boxes_rank == 3
    # Determine if images needs an expand_dims() call
    if images is not None:
        images_rank = len(images.shape)
        if images_rank > 4:
            raise ValueError(
                "Expected len(images.shape)=2, or len(images.shape)=3, got "
                f"len(images.shape)={images_rank}"
            )
        images_include_batch = images_rank == 4
        if boxes_includes_batch != images_include_batch:
            raise ValueError(
                "convert_format() expects both boxes and images to be batched, "
                "or both boxes and images to be unbatched. Received "
                f"len(boxes.shape)={boxes_rank}, "
                f"len(images.shape)={images_rank}. Expected either "
                "len(boxes.shape)=2 AND len(images.shape)=3, or "
                "len(boxes.shape)=3 AND len(images.shape)=4."
            )
        if not images_include_batch:
            images = tf.expand_dims(images, axis=0)

    if not boxes_includes_batch:
        return tf.expand_dims(boxes, axis=0), images, True
    return boxes, images, False


def _format_outputs(boxes, squeeze):
    if squeeze:
        return tf.squeeze(boxes, axis=0)
    return boxes


def _validate_image_shape(image_shape):
    # Escape early if image_shape is None and skip validation.
    if image_shape is None:
        return
    # tuple/list
    if isinstance(image_shape, (tuple, list)):
        if len(image_shape) != 3:
            raise ValueError(
                "image_shape should be of length 3, but got "
                f"image_shape={image_shape}"
            )
        return

    # tensor
    if tf.is_tensor(image_shape):
        if len(image_shape.shape) > 1:
            raise ValueError(
                "image_shape.shape should be (3), but got "
                f"image_shape.shape={image_shape.shape}"
            )
        if image_shape.shape[0] != 3:
            raise ValueError(
                "image_shape.shape should be (3), but got "
                f"image_shape.shape={image_shape.shape}"
            )
        return

    # Warn about failure cases
    raise ValueError(
        "Expected image_shape to be either a tuple, list, Tensor. "
        f"Received image_shape={image_shape}"
    )


def _convert_format(boxes, images=None, image_shape=None, dtype="float32"):
    if isinstance(boxes, dict):
        converted_boxes = boxes.copy()
        converted_boxes["boxes"] = _convert_format(
            boxes["boxes"],
            images=images,
            image_shape=image_shape,
            dtype=dtype,
        )
        return converted_boxes

    if boxes.shape[-1] is not None and boxes.shape[-1] != 4:
        raise ValueError(
            "Expected `boxes` to be a Tensor with a final dimension of "
            f"`4`. Instead, got `boxes.shape={boxes.shape}`."
        )
    if images is not None and image_shape is not None:
        raise ValueError(
            "convert_format() expects either `images` or `image_shape`, but "
            f"not both. Received images={images} image_shape={image_shape}"
        )

    _validate_image_shape(image_shape)

    boxes = tf.cast(boxes, dtype)
    boxes, images, squeeze = _format_inputs(boxes, images)

    result = _rel_yxyx_to_xyxy(boxes, images=images, i_shape=image_shape)
    return _format_outputs(result, squeeze)


def _unpackage_raw_tfds_inputs(inputs):
    image = inputs["image"]
    boxes = _convert_format(inputs["objects"]["bbox"], images=image)
    bounding_boxes = {
        "classes": inputs["objects"]["label"],
        "boxes": boxes,
    }
    return image, bounding_boxes


def load_voc(split, dataset, data_dir, repeat=True, batchify=True, take=None):
    np_config.enable_numpy_behavior()
    ds: tf.data.Dataset = tfds.load(dataset, data_dir=data_dir, split=split, with_info=False, shuffle_files=False)
    ds = ds.map(
        lambda x: _unpackage_raw_tfds_inputs(x),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.map(
        lambda x, y: PreprocessingUnit().preprocess(x, y),
        num_parallel_calls=1
    )

    ds = ds.filter(
        lambda x, y: tf.math.count_nonzero(y) > 0
    )

    if take is not None:
        ds = ds.take(take)

    ds = ds.shuffle(buffer_size=10*constants.BATCH_SIZE)
    if repeat:
        ds = ds.repeat(constants.EPOCHS)
    if batchify:
        ds = ds.batch(constants.BATCH_SIZE)

    return ds.prefetch(buffer_size=tf.data.AUTOTUNE).__iter__()


def load_voc_single(split, dataset, data_dir):
    sample = load_voc(split, dataset, data_dir, repeat=False, batchify=True, take=1)
    for x, y in sample:
        return x, y
