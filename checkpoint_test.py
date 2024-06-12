import json

import matplotlib.pyplot as plt
import numpy as np

from dataset import load_voc_single, load_voc
from graphics import plot_prediction


def checkpoint_plot_multiple(model, checkpoint_path, from_dataset_type, amount):
    model.load_weights(checkpoint_path)
    samples = load_voc(from_dataset_type, 'voc/2007', 'E:\\my-yolo\\voc', repeat=False, batchify=True, take=amount)
    for batch_x, _ in samples:
        batch_y_pred = model.predict(batch_x)
        for i in range(amount):
            plot_prediction(batch_x[i], batch_y_pred[i], confidence_threshold=0.8)


def checkpoint_plot_metrics(checkpoint_metrics_path, scale_truth):
    file_content = open(checkpoint_metrics_path, 'r').read()
    metrics = json.loads(file_content)

    loss = np.array(metrics['loss'])
    accuracy = np.array(metrics['accuracy'])
    ground_truth_accuracy = np.array(metrics['ground truth accuracy'])

    loss_line = plt.plot(loss)[0]
    plt.legend([loss_line], ['train loss'])
    plt.xlabel('epoch')
    plt.ylabel('metric value')

    plt.show()

    fit_scale = np.dot(accuracy, ground_truth_accuracy) / np.dot(ground_truth_accuracy, ground_truth_accuracy)
    scales = {
        'percents': 0.01,
        'fit': fit_scale
    }

    accuracy_line = plt.plot(accuracy)[0]
    ground_truth_accuracy_line = plt.plot(ground_truth_accuracy * scales[scale_truth])[0]

    plt.legend([accuracy_line, ground_truth_accuracy_line], ['train accuracy', 'train ground truth accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('metric value')

    plt.show()
