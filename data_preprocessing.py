import tensorflow as tf
import numpy as np

from constants import constants


class PreprocessingUnit:

    def preprocess(self, x, y):
        p_image = tf.py_function(self._preprocess_image, [x], Tout=tf.float32)
        p_ground_truth = tf.py_function(self._preprocess_ground_truth, [x[tf.newaxis, ...], y['classes'], y['boxes']], Tout=tf.float32)

        return p_image, p_ground_truth

    def _preprocess_image(self, image):
        preprocessed = tf.image.resize(images=image,
                                       size=[constants.image_resolution, constants.image_resolution]) / 255
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
                center_x, center_y, width, height = self._get_box_coordinates(box, image_height, image_width)
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

    def _get_box_coordinates(self, box, image_height, image_width):
        center_x = int(box[0])
        center_y = int(box[1])
        width = int(box[2])
        height = int(box[3])

        return center_x, center_y, width, height
