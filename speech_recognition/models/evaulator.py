import csv
from speech_recognition import models
import librosa
from sklearn.metrics import confusion_matrix
from kapre.utils import Normalization2D
from kapre.time_frequency import Melspectrogram
from speech_recognition.utils import TrainingConfiguration
from config import TRAINING_CONFIG_PATH, CURRENT_DIR, logger
# from utils.confusion_matrix import plot_confusion_matrix
from keras.models import load_model

import numpy as np


def plot_conf(cm,
              target_names,
              title='Confusion matrix',
              cmap=None,
              normalize=True,
              save=True
              ):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    save:
                If True save plot as image

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    if save:
       plt.savefig("confusion_matrix.png")

    else:
        plt.show()


class Evaluator(object):

    def __init__(self, csv_path, num_classes, model_path):

        self.csv_path = csv_path
        self.num_classes = num_classes
        self.model_path = model_path
        self.train_conf = TrainingConfiguration(TRAINING_CONFIG_PATH)

        self._test_dataset_path = self.train_conf.test_dataset_path
        self.base_model = getattr(models, self.train_conf.base_model)(self.num_classes)
        self.y_predict = []

        try:
            self.X, self.y_one_hot = self.create_test_dataset()
            self.y = [np.argmax(t) for t in self.y_one_hot]
        except Exception as _data:
            logger.exception("Failed creating test dataset...")
            logger.exception(f"Reason: {_data}")

        try:
            self.model = load_model(self.model_path, custom_objects=self.custom_objects())
        except Exception as _load:
            logger.exception(f"Failed loading model...")
            logger.exception(f"Reason: {_load}")

    @staticmethod
    def custom_objects():
        return {
            "Melspectrogram": Melspectrogram,
            "Normalization2D": Normalization2D
        }

    @property
    def prediction_model(self):
        return self.model

    def create_test_dataset(self, sampling_rate=16000):
        x_test = []
        y_test = []

        with open(self.csv_path, "r") as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row:
                    wav_full_path = f"{CURRENT_DIR}/{row[0]}"
                    label = int(row[1])
                    signal, sr = librosa.load(wav_full_path, sr=sampling_rate)
                    signal = librosa.resample(signal, sr, sampling_rate)

                    _tmp = [0 for _ in range(35)]
                    _tmp[label] = 1

                    x_test.append(signal)
                    y_test.append(_tmp)

        return np.asarray(x_test), np.asarray(y_test)

    def predict_raw_wave(self, wav_path):
        signal, sr = librosa.load(wav_path, sr=16000)
        signal = librosa.resample(signal, sr, 16000)
        print(signal.shape)
        return self.model.predict(signal)

    def evaluate(self):
        results = self.model.evaluate(self.X, self.y)
        return results

    def predict(self):

        print(self.X.shape)
        predictions = self.model.predict(self.X)

        for prediction in predictions:
            label = np.argmax(prediction)
            self.y_predict.append(label)

    def plot(self):
        # plot_confusion_matrix(self.y, self.y_predict, self.train_conf.label_map_config)
        # plot_confusion_matrix(self.model, self.X, self.y, labels=[0, 1])
        cm = confusion_matrix(self.y, self.y_predict, range(self.num_classes))
        print(self.train_conf.classes_of_interests)
        plot_conf(
            cm,
            self.train_conf.classes_of_interests,
            normalize=False,
            save=False
        )


e = Evaluator(
    csv_path=f"{CURRENT_DIR}/dataset/test/test.csv",
    num_classes=35,
    model_path=f"{CURRENT_DIR}/classifier35.h5"
)

e.predict()
e.plot()