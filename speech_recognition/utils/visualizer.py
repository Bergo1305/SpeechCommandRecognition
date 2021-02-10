import os
import numpy as np
import librosa
import librosa.display

from matplotlib import pyplot as plt


class WaveVisualizer(object):

    def __init__(self, wave_file_path):
        self.wave_file_path = wave_file_path
        self.wave_name = wave_file_path.split('/')[-1].split('.')[0]
        try:
            self.signal, self.sampling_rate = librosa.load(wave_file_path, sr=16000)
        except Exception as _exc:
            raise Exception(f"Can not load wave file: {wave_file_path}")

    @staticmethod
    def parse_path(path):
        if not os.path.isdir(path):
            os.mkdir(path)

        if path[-1] == "/":
            return path[:-1]

        return path

    def plot(self, x_label, y_label, name, show=True, save_dir=None, colorbar=False):
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(self.wave_name)

        if colorbar:
            plt.colorbar()

        if save_dir:
            plt.savefig(f"{self.parse_path(save_dir)}/{self.wave_name}-{name}.png")

        if show:
            plt.show()

        plt.close()

    def waveform(self, show=True, save_dir=None):
        """
        Calculate time domain audio file (time vs amplitude graph)
        Params
        :param show:
        Optional bool for showing plot figure
        :param save_dir:
        Path for saving figure as png image
        """
        librosa.display.waveplot(self.signal, sr=self.sampling_rate)
        self.plot("Time", "Amplitude", "waveform", show, save_dir)

    def spectrum(self, show=True, save_dir=None):
        """
        Calculate spectrum from Fast Furrier Transform(FFT).
        Because graph is symmetric around half, we will use only half graph
        Params:
        :param show:
        Optional bool for showing plot figure
        :param save_dir:
        Path for saving figure as png image
        """

        fft = np.fft.fft(self.signal)

        # Results of fft are complex numbers, we will use magnitude as feature representation.
        magnitude = np.abs(fft)
        frequency = np.linspace(0, self.sampling_rate, len(magnitude))

        left_frequency = frequency[: len(frequency) // 2]
        left_magnitude = magnitude[: len(frequency) // 2]

        plt.plot(left_frequency, left_magnitude)
        self.plot("Frequency", "Magnitude", "spectrum", show, save_dir)

    def spectrogram(self, n_fft=2048, hop_length=512, show=True, save_dir=None):
        """
        Calculate spectrogram from Short Time Furrier Transform(STFT).
        Params
        :param n_fft
        Number of samples used for STFT.
        :param hop_length
        Amount of shifting furrier transform points to the right direction.
        """

        stft = librosa.core.stft(self.signal, hop_length=hop_length, n_fft=n_fft)
        spectrogram = np.abs(stft)

        librosa.display.specshow(spectrogram, sr=self.sampling_rate, hop_length=hop_length)

        self.plot("Time", "Frequency", "spectrogram", show, save_dir, colorbar=True)

    def mfcc(self, n_fft=2048, hop_length=512, n_mfcc=13, show=True, save_dir=None):

        MFCCs = librosa.feature.mfcc(self.signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
        librosa.display.specshow(MFCCs)

        self.plot("Time", "MFCC", "mfcc", show, save_dir, colorbar=True)

    def mel_spectrogram(self, n_fft=2048, hop_length=512, show=True, save_dir=None):
        """
        Calculate spectrogram from Short Time Furrier Transform(STFT).
        Params
        :param n_fft
        Number of samples used for STFT.
        :param hop_length
        Amount of shifting furrier transform points to the right direction.
        """
        mel_spectrogram = librosa.feature.melspectrogram(
            self.signal,
            self.sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length
        )

        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        librosa.display.specshow(mel_spectrogram)

        self.plot("Time", "Mel", "mel-spectrogram", show, save_dir, colorbar=True)


class ModelVisualizer(object):

    def __init__(self, model_history):
        self.model_history = model_history

    def plot(self, accuracy_img_path, loss_img_path, show=False):
        acc = self.model_history.history['accuracy']
        val_acc = self.model_history.history['val_accuracy']

        loss = self.model_history.history['loss']
        val_loss = self.model_history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.savefig(accuracy_img_path)
        plt.show()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.savefig(loss_img_path)

        if show:
            plt.show()
