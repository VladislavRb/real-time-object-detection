import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import tensorflow.python.ops.clip_ops as ops

from constants import constants


def rate_scheduler(epoch, _):
    if epoch <= constants.LEARNING_RATE_0:
        start_epoch = 1E-3
        end_epoch = 1E-2

        lr = start_epoch + epoch * (end_epoch - start_epoch) / constants.LEARNING_RATE_0
        return lr
    if epoch <= constants.LEARNING_RATE_1:
        lr = 1E-2
        return lr
    if epoch <= constants.LEARNING_RATE_2:
        lr = 1E-3
        return lr
    if epoch <= constants.LEARNING_RATE_3:
        lr = 1E-4
        return lr


def rate_scheduler_v2(epoch, _):
    if epoch <= 40:
        start_epoch = 1E-4
        end_epoch = 1E-3

        lr = start_epoch + epoch * (end_epoch - start_epoch) / 40
        return lr
    if epoch <= 115:
        lr = 1E-3
        return lr
    if epoch <= 145:
        lr = 5E-4
        return lr

    return 1E-4


def rate_scheduler_decaying_v3(epoch, _):
    start_rate = 1E-4
    end_rate = 1E-5
    return start_rate + epoch * (end_rate - start_rate) / constants.EPOCHS


class StoreModelHistory(Callback):
    def __init__(self, filepath, write_epochs, *metric_names):
        super().__init__()
        self.filepath = filepath
        self.write_epochs = write_epochs

        self.metrics = {}
        for metric in metric_names:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for metric in self.metrics.keys():
                self.metrics[metric].append(logs[metric])

        if epoch in self.write_epochs:
            filename, ext = self.filepath.split('.')
            dump_filename = f'{filename}_epoch_{epoch}.{ext}'
            dump_stream = open(dump_filename, 'x')
            dump_stream.write(json.dumps(self.metrics))


class GroundTruthMetrics(tf.keras.metrics.Metric):
    def __init__(self, name):
        super().__init__(name=name)
        self.res = self.add_variable(
            shape=(),
            initializer='zeros',
            name='res'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        accuracy_score = tf.py_function(self._accuracy, [y_true, y_pred], Tout=tf.float32)
        self.res.assign(accuracy_score)

    def result(self):
        return self.res

    def _accuracy(self, y_true_t, y_pred_t):
        y_true = y_true_t.numpy()
        y_pred = y_pred_t.numpy()

        total_score = 0
        for batch_i in range(constants.BATCH_SIZE):
            max_score = 0
            real_score = 0
            for s_i in range(constants.s):
                for s_j in range(constants.s):
                    confidence_start_index = constants.c + 4
                    class_exists_in_pred = float(np.max(y_pred[batch_i, s_i, s_j, confidence_start_index::5])) > constants.CONFIDENCE_THRESHOLD
                    class_exists_in_true = float(np.max(y_true[batch_i, s_i, s_j, confidence_start_index::5])) > 0.0

                    max_score = max_score + constants.TT_SCORE if class_exists_in_true else constants.FF_SCORE

                    if class_exists_in_pred and class_exists_in_true:
                        pred_class = int(np.argmax(y_pred[batch_i, s_i, s_j, :constants.c]))
                        if float(y_true[batch_i, s_i, s_j, pred_class]) > 0.0:
                            real_score = real_score + constants.TT_SCORE * constants.TT_SCORE_CTOBB_RATIO

                        # TODO: adapt for B bounding boxes; currently we assume that B = 2
                        first_true_bb = y_true[batch_i, s_i, s_j, constants.c:(constants.c + 5)]
                        second_true_bb = y_true[batch_i, s_i, s_j, (constants.c + 5):]
                        first_pred_bb = y_pred[batch_i, s_i, s_j, constants.c:(constants.c + 5)]
                        second_pred_bb = y_pred[batch_i, s_i, s_j, (constants.c + 5):]

                        if np.sum(np.abs(first_true_bb - second_true_bb)) < 1E-4:  # equality check
                            responsible_pred_bb = first_pred_bb if float(first_pred_bb[-1]) > float(second_pred_bb[-1]) else second_pred_bb  # pick higher confidence
                            if self._check_bbox_iou(first_true_bb, responsible_pred_bb, s_j, s_i):
                                real_score = real_score + constants.TT_SCORE * (1 - constants.TT_SCORE_CTOBB_RATIO)
                        else:
                            if (self._check_bbox_iou(first_true_bb, first_pred_bb, s_j, s_i) and self._check_bbox_iou(second_true_bb, second_pred_bb, s_j, s_i)) or (self._check_bbox_iou(first_true_bb, second_pred_bb, s_j, s_i) and self._check_bbox_iou(second_true_bb, first_pred_bb, s_j, s_i)):
                                real_score = real_score + constants.TT_SCORE * (1 - constants.TT_SCORE_CTOBB_RATIO)

                    if not class_exists_in_pred and not class_exists_in_true:
                        real_score = real_score + constants.FF_SCORE

            total_score = total_score + real_score / max_score

        return total_score / constants.BATCH_SIZE

    def _check_bbox_iou(self, true_bbox, pred_bbox, x_i, y_i):
        if float(pred_bbox[-1]) < constants.CONFIDENCE_THRESHOLD:
            return False

        true_tl, true_br = self._top_left_bottom_right(true_bbox, x_i, y_i)
        pred_tl, pred_br = self._top_left_bottom_right(pred_bbox, x_i, y_i)
        tl = np.maximum(true_tl, pred_tl)
        br = np.minimum(true_br, pred_br)
        intersection_sides = ops.clip_by_value(br - tl, clip_value_min=0.0, clip_value_max=np.inf)
        intersection = np.prod(intersection_sides)

        true_area = np.prod(true_br - true_tl)
        pred_area = np.prod(pred_br - pred_tl)

        union = pred_area + true_area - intersection
        iou = float(intersection / (union + constants.EPSILON))

        return iou > constants.IOU_THRESHOLD

    def _top_left_bottom_right(self, bbox, x_i, y_i):
        xc = (x_i + bbox[0]) * constants.image_resolution / constants.s
        yc = (y_i + bbox[1]) * constants.image_resolution / constants.s
        width = bbox[2] * constants.image_resolution
        height = bbox[3] * constants.image_resolution

        x_min, x_max, y_min, y_max = xc - width * 0.5, xc + width * 0.5, yc - height * 0.5, yc + height * 0.5
        return np.array([x_min, y_min]), np.array([x_max, y_max])
