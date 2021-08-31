import tensorflow as tf
from tensorflow.python.client import device_lib
import itertools
import numpy as np
import matplotlib.pyplot as plt


def get_time_str(s_total):
    if s_total < 0:
        raise Exception('Seconds must be positive, but is {}'.format(s_total))
    h = s_total // 3600
    rest = s_total % 3600
    m = rest // 60
    s = rest % 60
    return '{}h {}m {}s'.format(int(h), int(m), int(s))


def check_gpu():
    print(device_lib.list_local_devices())

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


# function from https://deeplizard.com/learn/video/0LhiS6yu2qQ
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm / cm.astype(np.float).sum(axis=1)

    plt.figure(figsize=(len(classes), len(classes)))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def examples_iter(dataset):
    for batch in dataset:
        for x in batch[0]:
            yield x


def labels_iter(dataset):
    for batch in dataset:
        for y in batch[1]:
            yield y


def get_data_distribution(dataset, classes):
    distr = {label: 0 for label in classes}

    for label in labels_iter(dataset):
        distr[classes[label]] += 1

    return distr


def write_data_distribution(dataset, classes, filename):
    distr = get_data_distribution(dataset, classes)
    num_items = sum([v for k, v in distr.items()])

    with open(filename, 'w') as file:
        for k, v in distr.items():
            file.write('{}: {} ({}%)\n'.format(k, v, (v / num_items) * 100))
