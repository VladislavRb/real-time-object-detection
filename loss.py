import tensorflow as tf
import tensorflow.python.ops.clip_ops as ops
from typing import Tuple
import numpy as np
from constants import constants
from utils import bbox_attr


def general_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
    iou = get_iou(y_pred, y_true)
    max_iou = tf.reduce_max(iou, axis=-1)

    base_obj_i = tf.where(tf.reduce_max(bbox_attr(y_true, 4), axis=-1) > 0.0, 1.0, 0.0)
    base_obj_i_shape = tf.shape(base_obj_i)

    zeros = tf.zeros_like(bbox_attr(y_pred, 4))
    responsible = responsibility_tensor(zeros, tf.math.top_k(max_iou)[1])
    obj_ij = broadcast(base_obj_i, -1, (base_obj_i_shape[0], base_obj_i_shape[1], base_obj_i_shape[2], constants.b)) * responsible
    noobj_ij = tf.abs(obj_ij - 1)

    number_of_class_ij_boxes = tf.reduce_sum(obj_ij)
    x_losses = mse_loss(
        obj_ij * bbox_attr(y_pred, 0),
        obj_ij * bbox_attr(y_true, 0)
    )
    y_losses = mse_loss(
        obj_ij * bbox_attr(y_pred, 1),
        obj_ij * bbox_attr(y_true, 1)
    )
    pos_losses = (x_losses + y_losses) / (number_of_class_ij_boxes + constants.EPSILON)

    p_width = bbox_attr(y_pred, 2)
    a_width = bbox_attr(y_true, 2)
    width_losses = mse_loss(
        obj_ij * tf.sqrt(tf.abs(p_width)),
        obj_ij * tf.sqrt(tf.abs(a_width))
    )
    p_height = bbox_attr(y_pred, 3)
    a_height = bbox_attr(y_true, 3)
    height_losses = mse_loss(
        obj_ij * tf.sqrt(tf.abs(p_height)),
        obj_ij * tf.sqrt(tf.abs(a_height))
    )
    dim_losses = (width_losses + height_losses) / (number_of_class_ij_boxes + constants.EPSILON)

    obj_confidence_losses = mse_loss(
        obj_ij * bbox_attr(y_pred, 4),
        obj_ij * tf.ones_like(max_iou)
    ) / (number_of_class_ij_boxes + constants.EPSILON)
    noobj_confidence_losses = mse_loss(
        noobj_ij * bbox_attr(y_pred, 4),
        tf.zeros_like(max_iou)
    ) / (tf.reduce_sum(noobj_ij) + constants.EPSILON)

    number_of_class_boxes = tf.reduce_sum(base_obj_i)
    class_losses = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred[..., :constants.c], labels=y_true[..., :constants.c], axis=-1)
    class_losses = tf.reduce_sum(class_losses * base_obj_i) / (number_of_class_boxes + constants.EPSILON)

    total = constants.LCOORD * (pos_losses + dim_losses) \
            + obj_confidence_losses \
            + constants.LNOOBJ * noobj_confidence_losses \
            + class_losses
    return total


def responsibility_tensor(t, indices) -> tf.Tensor:
    t_rank = 4
    mesh_grid = tf.meshgrid(*[tf.range(tf.shape(t)[dim_i]) for dim_i in range(t_rank - 1)], indexing='ij')
    ij = tf.stack(mesh_grid, axis=-1)

    gathered_indices = tf.concat([ij, indices], axis=-1)
    indices_shape = tf.shape(gathered_indices)

    values_shape = tf.math.reduce_prod(indices_shape[:-1])
    gathered_indices = tf.reshape(gathered_indices, (values_shape, indices_shape[-1]))
    values = tf.ones(values_shape)
    output = tf.tensor_scatter_nd_update(t, gathered_indices, values)

    return output


def mse_loss(y_pred: tf.Tensor, y_true: tf.Tensor):
    return tf.keras.losses.mse(tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1]))


def get_iou(p, a) -> tf.Tensor:
    p_tl, p_br = bbox_to_coords(p)
    a_tl, a_br = bbox_to_coords(a)

    p_shape = tf.shape(p)
    coords_join_size = (p_shape[0], p_shape[1], p_shape[2], constants.b, constants.b, 2)
    tl: tf.Tensor = tf.maximum(
        broadcast(p_tl, 4, coords_join_size),
        broadcast(a_tl, 3, coords_join_size)
    )
    br: tf.Tensor = tf.minimum(
        broadcast(p_br, 4, coords_join_size),
        broadcast(a_br, 3, coords_join_size)
    )

    intersection_sides = ops.clip_by_value(br - tl, clip_value_min=0.0, clip_value_max=np.inf)
    intersection = tf.multiply(intersection_sides[..., 0], intersection_sides[..., 1])
    intersection_shape = tf.shape(intersection)

    p_area = tf.multiply(bbox_attr(p, 2), bbox_attr(p, 3))
    p_area = broadcast(p_area, 4, intersection_shape)

    a_area = tf.multiply(bbox_attr(a, 2), bbox_attr(a, 3))
    a_area = broadcast(a_area, 3, intersection_shape)

    union = p_area + a_area - intersection

    zero_unions = (union == 0.0)
    tf.where(zero_unions, constants.EPSILON, union)
    tf.where(zero_unions, 0.0, intersection)

    return tf.divide(intersection, union)


def broadcast(t: tf.Tensor, broadcast_axis: int, broadcast_shape: Tuple):
    return tf.broadcast_to(tf.expand_dims(t, axis=broadcast_axis), broadcast_shape)


def bbox_to_coords(t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    width = bbox_attr(t, 2)
    x = bbox_attr(t, 0)
    x1 = x - tf.divide(width, 2.0)
    x2 = x + tf.divide(width, 2.0)

    height = bbox_attr(t, 3)
    y = bbox_attr(t, 1)
    y1 = y - tf.divide(height, 2.0)
    y2 = y + tf.divide(height, 2.0)

    return tf.stack((x1, y1), axis=4), tf.stack((x2, y2), axis=4)
