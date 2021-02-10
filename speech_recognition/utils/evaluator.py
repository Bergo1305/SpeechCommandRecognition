import os
import csv
import librosa
import numpy as np

from speech_recognition import architectures
from sklearn.metrics import confusion_matrix
from kapre.utils import Normalization2D
from kapre.time_frequency import Melspectrogram
from config import CURRENT_DIR, logger, trainingConfiguration
from tensorflow.keras.models import load_model
from speech_recognition.utils.confusion_matrix import plot_conf


class Evaluator(object):

    def __init__(self, wav_files, num_classes, model_path, test_type="csv"):

        self.wav_files = wav_files
        self.num_classes = num_classes
        self.model_path = model_path
        self.test_type = test_type

        self._test_dataset_path = trainingConfiguration.test_dataset_path
        self.base_model = getattr(architectures, trainingConfiguration.base_model)(self.num_classes)
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

        if self.test_type == "csv":
            with open(self.wav_files, "r") as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    if row:
                        wav_full_path = f"{CURRENT_DIR}/{row[0]}"
                        label = int(row[1])
                        signal, sr = librosa.load(wav_full_path, sr=sampling_rate)
                        signal = librosa.resample(signal, sr, sampling_rate)

                        _tmp = [0 for _ in range(self.num_classes)]
                        _tmp[label] = 1

                        x_test.append(signal)
                        y_test.append(_tmp)

        if self.test_type == "raw":

            for root, _, files in os.walk(self.wav_files):

                for file in files:
                    wav_path = os.path.join(root, file)
                    label = wav_path.split('/')[-2]

                    if label not in trainingConfiguration.classes_of_interests:
                        continue

                    converted_label = trainingConfiguration.label_map_config[label]

                    signal, sr = librosa.load(wav_path, sr=sampling_rate)
                    signal = librosa.resample(signal, sr, sampling_rate)

                    _tmp = [0 for _ in range(self.num_classes)]
                    _tmp[converted_label] = 1

                    x_test.append(signal)
                    y_test.append(_tmp)

        return np.asarray(x_test), np.asarray(y_test)

    def predict_raw_wave(self, wav_path):
        signal, sr = librosa.load(wav_path, sr=trainingConfiguration.sampling_rate)
        signal = librosa.resample(signal, sr, trainingConfiguration.sampling_rate)
        return self.model.predict(signal)

    def evaluate(self):
        results = self.model.evaluate(self.X, self.y)
        return results

    def predict(self):

        predictions = self.base_model.predict(self.X)

        for prediction in predictions:
            label = np.argmax(prediction)
            self.y_predict.append(label)

    def plot(self):

        if self.y_predict is None:
            self.predict()

        cm = confusion_matrix(self.y, self.y_predict, range(self.num_classes))

        plot_conf(
            cm,
            trainingConfiguration.classes_of_interests,
            normalize=False,
            save=False
        )