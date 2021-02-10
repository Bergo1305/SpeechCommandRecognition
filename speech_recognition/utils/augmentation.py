from config import trainingConfiguration
import librosa
import random
from glob import glob


class Augmentation(object):

    def __init__(self):
        pass

    @staticmethod
    def _generate_background_noise():
        background_wav_files = []
        background_noises = []

        for _noise_wav in trainingConfiguration.background_classes:
            background_wav_files.extend(
                glob(f"{trainingConfiguration.train_dataset_path}/{_noise_wav}/*.wav")

            )

        for _wave in background_wav_files:
            samples, sr = librosa.load(_wave)
            samples = librosa.resample(samples, sr, 16000)
            background_noises.append(samples)

        return background_noises

    def get_random_noise(self, idx, sr=16000):
        background_noises = self._generate_background_noise()
        selected_noise = background_noises[idx]
        start_idx = random.randint(0, len(selected_noise) - 1 - sr)

        return selected_noise[start_idx:(start_idx + sr)]

    def add_noise(self, image_sample, num_augments=1, noise_deviation=0.1):

        for augment in range(0, num_augments):

            noise = self.get_random_noise(augment)

            yield image_sample + (noise_deviation * noise)









