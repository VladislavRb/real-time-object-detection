import random

import tensorflow as tf
import numpy as np

from constants import constants
from utils import scale_bbox_coord
from augment_data import AugmentData


class PreprocessingUnit:
    def __init__(self, augment):
        self.augment = augment and (random.random() < constants.AUGMENT_PROB)
        self.augment_data = AugmentData.create_random()

    def preprocess(self, x, y):
        p_image = tf.py_function(self._preprocess_image, [x], Tout=tf.float32)
        p_ground_truth = tf.py_function(self._preprocess_ground_truth, [x[tf.newaxis, ...], y['classes'], y['boxes']], Tout=tf.float32)

        return p_image, p_ground_truth

    def _preprocess_image(self, image):
        preprocessed = tf.image.resize(images=image,
                                       size=[constants.image_resolution, constants.image_resolution]) / 255
        if self.augment:
            preprocessed = tf.keras.preprocessing.image.apply_affine_transform(preprocessed,
                                                                               theta=0,
                                                                               tx=self.augment_data.x_shift,
                                                                               ty=self.augment_data.y_shift,
                                                                               shear=0,
                                                                               zx=self.augment_data.zoom,
                                                                               zy=self.augment_data.zoom,
                                                                               row_axis=0,
                                                                               col_axis=1,
                                                                               channel_axis=2)
            preprocessed = tf.image.adjust_hue(preprocessed, delta=self.augment_data.hue_shift)
            preprocessed = tf.image.adjust_saturation(preprocessed, saturation_factor=self.augment_data.saturation_shift)
        return preprocessed

    def _preprocess_ground_truth(self, original_image, classes, boxes):
        try:
            ground_truth_tensor = np.zeros((constants.s, constants.s, constants.cell_predictions_amount))
            i_shape = tf.shape(original_image)
            image_height, image_width = int(i_shape[1]), int(i_shape[2])
            grid_y, grid_x = image_height / constants.s, image_width / constants.s
            assigned_classes = {}
            assigned_boxes = {}

            box_i = 0
            for box in tf.unstack(boxes):
                center_x, center_y, width, height = self._get_box_coordinates(box, image_height, image_width,
                                                                              self.augment, self.augment_data)
                cell_col, cell_row = int(center_x // grid_x), int(center_y // grid_y)
                if 0 <= cell_col < constants.s and 0 <= cell_row < constants.s:
                    cell = (cell_col, cell_row)
                    box_class = int(classes[box_i])
                    if cell not in assigned_classes or assigned_classes[cell] == box_class:
                        gt_classes = np.array([1 if gt_i == box_class else 0 for gt_i in range(constants.c)],
                                              dtype='float32')
                        ground_truth_tensor[cell_row, cell_col, :constants.c] = gt_classes
                        assigned_classes[cell] = box_class

                        bbox_index = assigned_boxes.get(cell, 0)
                        if bbox_index < constants.b:
                            bbox_truth = (
                                (center_x - cell_col * grid_x) / grid_x,  # X coord relative to grid square
                                (center_y - cell_row * grid_y) / grid_y,  # Y coord relative to grid square
                                width / image_width,  # Width
                                height / image_height,  # Height
                                1.0  # Confidence
                            )
                            bbox_start = 5 * bbox_index + constants.c
                            ground_truth_tensor[cell_row, cell_col, bbox_start:] = np.tile(bbox_truth,
                                                                                           (constants.b - bbox_index))
                            assigned_boxes[cell] = bbox_index + 1
                box_i += 1

            return tf.constant(ground_truth_tensor, dtype='float32')
        except Exception:
            print(f'A dataset record was not processed because of an error')
            return tf.zeros(shape=(constants.s, constants.s, constants.cell_predictions_amount))

    def _get_box_coordinates(self, box, image_height, image_width, augment, augment_data):
        center_x = int(box[0])
        center_y = int(box[1])
        width = int(box[2])
        height = int(box[3])

        if augment:
            half_width = image_width * 0.5
            half_height = image_height * 0.5

            center_x = scale_bbox_coord(center_x - augment_data.x_shift, half_width, augment_data.zoom)
            center_y = scale_bbox_coord(center_y - augment_data.y_shift, half_height, augment_data.zoom)
            width = width / augment_data.zoom
            height = height / augment_data.zoom

        return center_x, center_y, width, height
