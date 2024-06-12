from model import *
from device_config import configure_gpu, configure_tensorflow
from train import *
from checkpoint_test import *
from graphics import *


def show_demo():
    model = create_light_model_v2()
    model.compile(optimizer=Adam(epsilon=1E-8),
                  loss=general_loss,
                  metrics=['accuracy', GroundTruthMetrics(constants.GROUND_TRUTH_METRICS)])
    checkpoint_plot_multiple(model, 'E:\\my-yolo\\checkpoints_v2\\checkpoint-176.h5', 'train', 10)
    checkpoint_plot_metrics('E:\\my-yolo\\metrics_v2\\data_epoch_145.json', 'percents')
    plot_validation_statistics('percents')


def train_demo(version):
    train(version, create_light_model_v2, Adam(epsilon=1E-8), general_loss, rate_scheduler_v2)


# possible factors : batch_size   optimizer   model      learning_rate_schedule     epochs
#                         16        Adam      full               rate1               145
#                         32       RMSProp    light              rate2               175
#                         64         SGD     light_v2           constant             200
#                         128

configure_gpu()
configure_tensorflow(is_eager_execution=True)

train_demo(3)
