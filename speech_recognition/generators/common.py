import csv
import numpy as np
from tensorflow import keras
import librosa
from tensorflow import keras
from config import CURRENT_DIR
from speech_recognition.utils.augmentation import Augmentation


class CSVDataGenerator(keras.utils.Sequence):

    def __init__(self,
                 dataset_csv_path,
                 batch_size=32,
                 sampling_rate=16000,
                 n_classes=35,
                 shuffle=False
                 ):

        self.dim = sampling_rate
        self.batch_size = batch_size
        self.dataset_path = dataset_csv_path
        self.n_classes = n_classes
        self.indexes = []
        self.shuffle = shuffle

        self.augmentation = Augmentation()
        self.waves, self.labels = self.read_csv()
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.waves) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """

        indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]
        batch_files = [self.waves[k] for k in indexes]

        X, y = self.__data_generation(batch_files)
        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.waves))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, waves_tmp):
        """
        Generates data. It will consists of batch size samples (X, y)
        """

        X = np.empty((self.batch_size, self.dim))
        y = np.empty(self.batch_size, dtype=int)

        for _idx, sample in enumerate(waves_tmp):
            signal, sr = librosa.load(sample, sr=self.dim)

            X[_idx] = signal
            y[_idx] = self.labels[sample]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def read_csv(self):
        waves = []
        labels = {}

        with open(self.dataset_path, "r") as file:
            csv_reader = csv.reader(file)

            for row in csv_reader:
                if row:
                    label = row[-1]
                    wav_path = f"{CURRENT_DIR}/{row[0]}" if row[0][0] != '/' else f"{CURRENT_DIR}{row[0]}"

                    signal, sr = librosa.load(wav_path, sr=self.dim)
                    signal = librosa.resample(signal, sr, self.dim)

                    if signal.shape[0] != self.dim:
                        continue

                    waves.append(wav_path)
                    labels[wav_path] = label

        return waves, labels

