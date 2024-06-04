import tensorflow as tf


def configure_tensorflow(is_eager_execution):
    tf.keras.backend.clear_session()
    if is_eager_execution:
        tf.compat.v1.enable_eager_execution()
    else:
        tf.compat.v1.disable_eager_execution()


def configure_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
