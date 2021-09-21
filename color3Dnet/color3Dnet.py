import argparse
import os
import random
import sys
from pprint import pprint
from time import time
from uuid import uuid4

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np

import LiDARCarDataset
from utils import get_time_str, check_gpu, plot_confusion_matrix, write_data_distribution, labels_iter


def main(paths, granularity='manufacturer_main', split=0.2, augmentation_split=False, strict_split=False, shuffle=True,
         random_seed=None, shuffle_pointcloud=False, normalize_distr=False, color=True, augmented=True,
         num_points=10000, batch_size=20, epochs=30, dropout=0.3, device='/device:CPU:0', tensorflow_data_dir=None,
         artifacts_dir=None):
    uuid = str(uuid4())
    experiment_name = 'color3Dnet-{}'.format(uuid)
    os.mkdir(experiment_name)
    print('Experiment name:', experiment_name)

    if not tensorflow_data_dir:
        tensorflow_data_dir = os.path.join(experiment_name, 'tensorflow_data')
    if not os.path.exists(tensorflow_data_dir):
        os.mkdir(tensorflow_data_dir)

    if not artifacts_dir:
        artifacts_dir = experiment_name
    if not os.path.exists(artifacts_dir):
        os.mkdir(artifacts_dir)

    if not random_seed:
        random_seed = random.randint(0, sys.maxsize)
    print('Using random seed:', random_seed)

    arguments = locals()

    # fix compatibility with RTX cards:
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    tf.random.set_seed(random_seed)

    builder_kwargs = {'path': paths,
                      'granularity': granularity,
                      'split': split,
                      'augmentation_split': augmentation_split,
                      'strict_split': strict_split,
                      'shuffle': shuffle,
                      'random_seed': random_seed,
                      'shuffle_pointcloud': shuffle_pointcloud,
                      'normalize_distr': normalize_distr,
                      'color': color,
                      'augmented': augmented
                      }

    classes = LiDARCarDataset.GRANULARITY[granularity]
    num_classes = len(classes)

    (ds_train, ds_test), ds_info = tfds.load('LiDARCarDataset', split=['train', 'test'],
                                             data_dir=tensorflow_data_dir,
                                             shuffle_files=True, as_supervised=True, with_info=True,
                                             builder_kwargs=builder_kwargs)

    # train pipeline:
    ds_train = ds_train.cache()
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # test pipeline:
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    ### NEURAL NETWORK ###
    class OrthogonalRegularizer(keras.regularizers.Regularizer):
        def __init__(self, num_features, l2reg=0.001):
            self.num_features = num_features
            self.l2reg = l2reg
            self.eye = tf.eye(num_features)

        def __call__(self, x):
            x = tf.reshape(x, (-1, self.num_features, self.num_features))
            xxt = tf.tensordot(x, x, axes=(2, 2))
            xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
            return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

        def get_config(self):
            return {'num_features': self.num_features,
                    'l2reg': self.l2reg}

    def conv_bn(x, filters):
        x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)

    def dense_bn(x, filters):
        x = layers.Dense(filters)(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)

    def tnet(inputs, num_features):
        # Initalise bias as the indentity matrix
        bias = keras.initializers.Constant(np.eye(num_features).flatten())
        reg = OrthogonalRegularizer(num_features)

        x = conv_bn(inputs, 32)
        x = conv_bn(x, 64)
        x = conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        x = dense_bn(x, 256)
        x = dense_bn(x, 128)
        x = layers.Dense(
            num_features * num_features,
            kernel_initializer="zeros",
            bias_initializer=bias,
            activity_regularizer=reg,
        )(x)
        feat_T = layers.Reshape((num_features, num_features))(x)
        # Apply affine transformation to input features
        return layers.Dot(axes=(2, 1))([inputs, feat_T])

    def split_data(x):
        # add 1 dim for cropping layer
        x = layers.Reshape((num_points, 6, 1))(x)
        # separate points and colors
        points = layers.Cropping2D(cropping=((0, 0), (0, 3)))(x)
        colors = layers.Cropping2D(cropping=((0, 0), (3, 0)))(x)
        # get rid of the extra dim again
        points = layers.Reshape((num_points, 3))(colors)
        colors = layers.Reshape((num_points, 3))(colors)
        return points, colors

    def make_model(dropout=0.3):
        inputs = keras.Input(shape=(num_points, 6))

        # split points
        x, xc = split_data(inputs)

        # processing of normal points
        x = tnet(x, 3)
        x = conv_bn(x, 32)
        x = conv_bn(x, 32)
        x = tnet(x, 32)
        x = conv_bn(x, 32)
        x = conv_bn(x, 64)
        x = conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)

        # processing colors
        xc = tnet(xc, 3)
        xc = conv_bn(xc, 32)
        xc = conv_bn(xc, 32)
        xc = tnet(xc, 32)
        xc = conv_bn(xc, 32)
        xc = conv_bn(xc, 64)
        xc = conv_bn(xc, 512)
        xc = layers.GlobalMaxPooling1D()(xc)

        x = layers.Concatenate()([x, xc])
        x = dense_bn(x, 512)
        x = layers.Dropout(dropout)(x)
        x = dense_bn(x, 256)
        x = layers.Dropout(dropout)(x)
        x = dense_bn(x, 128)
        x = layers.Dropout(dropout)(x)

        outputs = layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="color3Dnet")

        return model

    model = make_model()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['sparse_categorical_accuracy'])

    checkpoint_filepath = os.path.join(artifacts_dir, 'best-weights-{}'.format(experiment_name))
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_sparse_categorical_accuracy',
        mode='max',
        save_best_only=True)

    checkpoint_path_all = os.path.join(artifacts_dir, 'checkpoints-{}'.format(experiment_name), 'cp-{epoch:04d}.ckpt')
    checkpoint_dir_all = os.path.dirname(checkpoint_path_all)

    # Create a callback that saves the model's weights
    model_checkpoint_callback_all = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_all,
                                                                       save_weights_only=True,
                                                                       verbose=1,
                                                                       save_freq='epoch')

    start = time()
    with tf.device(device):
        history = model.fit(ds_train, epochs=epochs, batch_size=batch_size, validation_data=ds_test, shuffle=True,
                            callbacks=[model_checkpoint_callback, model_checkpoint_callback_all])
    stop = time()
    time_seconds = stop - start

    ### log artifacts ###
    print('LOG ARTIFACTS')
    print('Save ds_info')
    with open(os.path.join(artifacts_dir, 'ds_info-{}.txt'.format(experiment_name)), 'w') as file:
        file.write(str(ds_info))

    print('Create and save accuracy graph')
    # accuracy graph:
    plt.clf()  # clear figure
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('sparse_categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    filename = os.path.join(artifacts_dir, 'accuracy-{}.png'.format(experiment_name))
    plt.savefig(filename)

    print('Create and save loss graph')
    # loss graph:
    plt.clf()  # clear figure
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    filename = os.path.join(artifacts_dir, 'loss-{}.png'.format(experiment_name))
    plt.savefig(filename)

    print('Save raw history')
    # raw history
    with open(os.path.join(artifacts_dir, 'history-{}.txt'.format(experiment_name)), 'w') as file:
        file.write(str(history.history))

    print('Restoring weights from best checkpoint')
    # get best checkpoint
    model.load_weights(checkpoint_filepath)
    print('Evaluate model with best weights')
    with tf.device(device):
        train_loss, train_acc = model.evaluate(ds_train)
        val_loss, val_acc = model.evaluate(ds_test)

    # confusion matrix train
    print('Predict training data for confusion matrix creation')
    with tf.device(device):
        train_pred = model.predict(ds_train)
    train_pred = np.argmax(train_pred, axis=1)
    cm = np.asmatrix(tf.math.confusion_matrix(list(map(int, labels_iter(ds_train))), train_pred))

    print('Create and save confusion matrix for training data')
    plt.clf()
    plot_confusion_matrix(cm, classes, normalize=False, title='CM - train')
    plt.savefig(os.path.join(artifacts_dir, 'cm-train-{}.png'.format(experiment_name)))

    print('Create and save normalized confusion matrix for training data')
    plt.clf()
    plot_confusion_matrix(cm, classes, normalize=True, title='CM - train - norm')
    plt.savefig(os.path.join(artifacts_dir, 'cm-train-norm-{}.png'.format(experiment_name)))

    # confusion matrix test
    print('Predict test data for confusion matrix creation')
    with tf.device(device):
        test_pred = model.predict(ds_test)
    test_pred = np.argmax(test_pred, axis=1)
    cm = np.asmatrix(tf.math.confusion_matrix(list(map(int, labels_iter(ds_test))), test_pred))

    print('Create and save confusion matrix for test data')
    plt.clf()
    plot_confusion_matrix(cm, classes, normalize=False, title='CM - test')
    plt.savefig(os.path.join(artifacts_dir, 'cm-test-{}.png'.format(experiment_name)))

    print('Create and save normalized confusion matrix for test data')
    plt.clf()
    plot_confusion_matrix(cm, classes, normalize=True, title='CM - test - norm')
    plt.savefig(os.path.join(artifacts_dir, 'cm-test-norm-{}.png'.format(experiment_name)))

    print('Save data distribution in training set')
    # data distribution train
    write_data_distribution(ds_train, classes,
                            os.path.join(artifacts_dir, 'data-distr-train-{}.txt'.format(experiment_name)))

    print('Save data distribution in test set')
    # data distribution test
    write_data_distribution(ds_test, classes,
                            os.path.join(artifacts_dir, 'data-distr-test-{}.txt'.format(experiment_name)))

    print('Save general information (arguments provided, runtime, ...)')
    # other info
    with open(os.path.join(artifacts_dir, 'info-{}.txt'.format(experiment_name)), 'w') as file:
        file.write('Experiment name: {}\n'.format(experiment_name))
        file.write('\n')
        file.write('best_train_loss: {}\n'.format(train_loss))
        file.write('best_train_acc : {}\n'.format(train_acc))
        file.write('best_val_loss  : {}\n'.format(val_loss))
        file.write('best_val_acc   : {}\n'.format(val_acc))
        file.write('\n')
        file.write('Random seed    : {}\n'.format(random_seed))
        file.write('Training time  : {} ({}s)\n'.format(get_time_str(time_seconds), time_seconds))
        file.write('Arguments      :\n')
        pprint(arguments, stream=file)

    print('All done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the color3Dnet.')
    parser.add_argument('paths', metavar='PATHS', nargs='+',
                        help='one or multiple data source path(s)')
    parser.add_argument('--granularity', metavar='GRANULARITY', default='manufacturer_main',
                        help='Granularity levels available: ' + str(LiDARCarDataset.GRANULARITY.keys()))
    parser.add_argument('--split', metavar='SPLIT', type=float, default=0.2, help='data split from range (0, 1)')
    parser.add_argument('--augmentation-split', metavar='AUGMENTATION_SPLIT', type=bool, default=False, help='If set to '
               'True, training is performed on augmented data only and testing is performed on non-augmented data only')
    parser.add_argument('--strict-split', metavar='STRICT_SPLIT', type=bool, default=False, help='Applies strict split; '
                                                                                 'refer to thesis for more information')
    parser.add_argument('--shuffle', metavar='SHUFFLE', type=bool, default=True, help='If set to True, the data set is '
                                                                                      'shuffled')
    parser.add_argument('--random-seed', metavar='RANDOM_SEED', type=int, default=None,
                        help='Sets random seed for the run, if not provided a random number is generated')
    parser.add_argument('--shuffle-pointcloud', metavar='SHUFFLE_POINTCLOUD', type=bool, default=False, help='If set to '
                                                                    'True, the points in each point cloud are shuffled')
    parser.add_argument('--normalize-distr', metavar='NORMALIZE_DISTR', type=bool, default=False, help='If set to True, '
                        'the distribution of the data is normalized, such that the number of examples used for each '
                                                                                    'class is the same for all classes')
    parser.add_argument('--color', metavar='COLOR', type=bool, default=True, help='If set to True, color values are '
                                                                                  'used')
    parser.add_argument('--augmented', metavar='AUGMENTED', type=bool, default=True, help='If set to False, no augmented '
                                                                                          'data is used')
    parser.add_argument('--num-points', metavar='NUM_POINTS', type=int, default=10000, help='Specifies the number of '
                                                                                            'points in each pointcloud')
    parser.add_argument('--batch-size', metavar='BATCH_SIZE', type=int, default=20)
    parser.add_argument('--epochs', metavar='EPOCHS', type=int, default=30)
    parser.add_argument('--dropout', metavar='DROPOUT', type=float, default=0.3, help='Specifies the dropout rate from '
                                                                                      'the range (0, 1)')
    parser.add_argument('--device', metavar='DEVICE', default='/device:CPU:0', help='Specifies the device to be used '
                                  'for training; some examples are "/device:CPU:0", "/device:GPU:0" or "/device:GPU:1"')
    parser.add_argument('--tensorflow-data-dir', metavar='TENSORFLOW_DATA_DIR', default=None, help='Specifies the '
                        'directory to store the tensorflow data generated when parsing the data set; this directory can '
                        'be reused to save some time when reusing the same data (with same split, etc.) for a different '
                                                                                                   'run')
    parser.add_argument('--artifacts-dir', metavar='ARTIFACTS_DIR', default=None, help='Specifies the directory where '
                                                                                       'the artifacts are saved')

    args = vars(parser.parse_args())

    main(paths=args['paths'], granularity=args['granularity'], split=args['split'],
         augmentation_split=args['augmentation_split'], strict_split=args['strict_split'], shuffle=args['shuffle'],
         random_seed=args['random_seed'], shuffle_pointcloud=args['shuffle_pointcloud'],
         normalize_distr=args['normalize_distr'], color=args['color'], augmented=args['augmented'],
         num_points=args['num_points'], batch_size=args['batch_size'], epochs=args['epochs'], dropout=args['dropout'],
         device=args['device'], tensorflow_data_dir=args['tensorflow_data_dir'], artifacts_dir=args['artifacts_dir'])
