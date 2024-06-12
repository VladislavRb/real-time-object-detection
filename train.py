import os.path
from tensorflow.keras.optimizers import Adam
from callbacks import *
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

from loss import general_loss
from dataset import load_voc


def train(version, model_factory, optimizer, loss, lr_scheduler):
    train_set = load_voc('train', 'voc/2007', 'E:\\my-yolo\\voc', batchify=True)
    validation_set = load_voc('validation', 'voc/2007', 'E:\\my-yolo\\voc', batchify=True,
                              repeat=True,
                              take=constants.VOC_VALIDATION_CARDINALITY - constants.VOC_VALIDATION_CARDINALITY % constants.BATCH_SIZE)

    model = model_factory()
    model.summary()
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=['accuracy', GroundTruthMetrics(constants.GROUND_TRUTH_METRICS)])

    checkpoints_folder = f'E:\\my-yolo\\checkpoints_v{version}\\'
    metrics_folder = f'E:\\my-yolo\\metrics_v{version}\\'

    val_loss_model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoints_folder + 'checkpoint-min-val-loss-{epoch:02d}.h5',
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    val_ground_truth_accuracy_model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoints_folder + 'checkpoint-max-val-ground-truth-accuracy-{epoch:02d}.h5',
        save_weights_only=True,
        monitor=f'val_{constants.GROUND_TRUTH_METRICS}',
        mode='max',
        save_best_only=True)
    store_history_callback = StoreModelHistory(metrics_folder + 'data.json',
                                               range(0, constants.EPOCHS, 5),
                                               'accuracy',
                                               'val_accuracy',
                                               'loss',
                                               'val_loss',
                                               constants.GROUND_TRUTH_METRICS,
                                               f'val_{constants.GROUND_TRUTH_METRICS}')

    model.fit(train_set,
              batch_size=constants.BATCH_SIZE,
              epochs=constants.EPOCHS,
              steps_per_epoch=(constants.VOC_TRAIN_CARDINALITY // constants.BATCH_SIZE),
              validation_data=validation_set,
              validation_steps=(constants.VOC_VALIDATION_CARDINALITY // constants.BATCH_SIZE),
              validation_batch_size=constants.BATCH_SIZE,
              callbacks=[
                  LearningRateScheduler(lr_scheduler, verbose=1),
                  val_loss_model_checkpoint_callback,
                  val_ground_truth_accuracy_model_checkpoint_callback,
                  store_history_callback])


def evaluate_validation(version, model_provider):
    model = model_provider()
    model.compile(optimizer=Adam(epsilon=1E-8),
                  loss=general_loss,
                  metrics=['accuracy', GroundTruthMetrics(constants.GROUND_TRUTH_METRICS)])

    checkpoints_folder = f'E:\\my-yolo\\checkpoints_v{version}\\'
    validation_list = []

    for epoch_i in range(constants.EPOCHS):
        ei_prefix = '0' if epoch_i < 10 else ''
        checkpoint_file_path = f'{checkpoints_folder}checkpoint-{ei_prefix}{epoch_i}.h5'
        if os.path.exists(checkpoint_file_path):
            validation_set = load_voc('validation', 'voc/2007', 'E:\\my-yolo\\voc', batchify=True,
                                      repeat=False,
                                      take=constants.VOC_VALIDATION_CARDINALITY - constants.VOC_VALIDATION_CARDINALITY % constants.BATCH_SIZE)
            model.load_weights(checkpoint_file_path)
            val_result = model.evaluate(validation_set)
            validation_list.append({
                'epoch': epoch_i,
                'val_result': val_result
            })

    dump_filename = f'E:\\my-yolo\\validation_list_v{version}\\stats.json'
    dump_stream = open(dump_filename, 'x')
    dump_stream.write(json.dumps(validation_list))
