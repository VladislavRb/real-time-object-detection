import json
import matplotlib.pyplot as plt
import numpy as np

from constants import constants
from labels import labels
from utils import clip


def plot_metrics(metrics_filepath, *plotted_metrics):
    file_content = open(metrics_filepath, 'r').read()
    metrics = json.loads(file_content)
    metric_lines = []

    for m in plotted_metrics:
        metric_lines.append(plt.plot(metrics[m])[0])

    plt.legend(metric_lines, plotted_metrics)
    plt.show()


def plot_validation_statistics(scale_truth):
    file_content = open('E:\\my-yolo\\validation_list_v2\\stats.json', 'r').read()
    validation_records = json.loads(file_content)[0:86]

    val_epochs = list(map(lambda record: record['epoch'], validation_records))

    val_loss = list(map(lambda record: record['val_result'][0], validation_records))
    val_accuracy = list(map(lambda record: record['val_result'][1], validation_records))
    val_ground_truth_accuracy = list(map(lambda record: record['val_result'][2], validation_records))

    val_loss_plot = plt.plot(val_epochs, val_loss)[0]

    plt.legend([val_loss_plot], ['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('metric value')

    plt.show()

    fit_scale = np.dot(val_accuracy, val_ground_truth_accuracy) / np.dot(val_ground_truth_accuracy, val_ground_truth_accuracy)
    scales = {
        'percents': 0.01,
        'fit': fit_scale
    }

    val_accuracy_plot = plt.plot(val_epochs, val_accuracy)[0]
    val_ground_truth_accuracy_plot = plt.plot(val_epochs, np.array(val_ground_truth_accuracy) * scales[scale_truth])[0]
    plt.legend([val_accuracy_plot, val_ground_truth_accuracy_plot],
               ['val_accuracy', 'val_ground_truth_accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('metric value')

    plt.show()


def plot_learning_rate_scheduler(lr_scheduler):
    x = range(constants.EPOCHS)
    y = [lr_scheduler(epoch_i, None) for epoch_i in x]

    lr_plot = plt.plot(x, y)[0]
    plt.legend([lr_plot], ['learning rate schedule'])
    plt.xlabel('epoch')
    plt.ylabel('lr value')

    plt.show()


def plot_batch(batch_x, batch_y, confidence_threshold):
    for i in range(constants.BATCH_SIZE):
        plot_prediction(batch_x[i], batch_y[i], confidence_threshold)


def plot_prediction(x, y, confidence_threshold):
    plt.imshow(x)
    rectangles = []
    rect_labels = []

    for y_i in range(constants.s):
        for x_i in range(constants.s):
            for b_i in range(constants.b):
                bbox = y[y_i, x_i, (constants.c + b_i * 5):(constants.c + (b_i + 1) * 5)]
                if bbox[4] > confidence_threshold:
                    rectangle = _plot_bbox(bbox, x_i, y_i)
                    rectangles.append(rectangle)
                    rectangle_label = labels[np.argmax(y[y_i, x_i, :constants.c])]
                    rect_labels.append(f'{rectangle_label} - {clip(bbox[4])}')

    plt.legend(rectangles, rect_labels)
    plt.show()


def _plot_bbox(bbox, x_i, y_i):
    xc = (x_i + bbox[0]) * constants.image_resolution / constants.s
    yc = (y_i + bbox[1]) * constants.image_resolution / constants.s
    width = bbox[2] * constants.image_resolution
    height = bbox[3] * constants.image_resolution

    x_min, x_max, y_min, y_max = xc - width * 0.5, xc + width * 0.5, yc - height * 0.5, yc + height * 0.5
    x_min, x_max, y_min, y_max = (clip(x_min, 0, constants.image_resolution),
                                  clip(x_max, 0, constants.image_resolution),
                                  clip(y_min, 0, constants.image_resolution),
                                  clip(y_max, 0, constants.image_resolution))
    return plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min])[0]
