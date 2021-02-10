from tensorflow.keras.models import Model
from tensorflow.keras import layers
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D


def convolution_speech_model(num_category, sampling_rate=16000, input_length=16000):

    inputs = layers.Input((input_length,))

    x = layers.Reshape((1, -1))(inputs)

    x = Melspectrogram(
        n_dft=1024,
        n_hop=128,
        input_shape=(1, input_length),
        padding='same',
        sr=sampling_rate,
        n_mels=80,
        fmin=40.0,
        fmax=sampling_rate / 2,
        power_melgram=1.0,
        return_decibel_melgram=True,
        trainable_fb=False,
        trainable_kernel=False,
        name='mel_stft')(x)

    x = Normalization2D(int_axis=0)(x)

    x = layers.Permute((2, 1, 3))(x)

    c1 = layers.Conv2D(20, (5, 1), activation='relu', padding='same')(x)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 1))(c1)
    p1 = layers.Dropout(0.03)(p1)

    c2 = layers.Conv2D(40, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(0.01)(p2)

    c3 = layers.Conv2D(80, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    p3 = layers.Flatten()(p3)
    p3 = layers.Dense(64, activation='relu')(p3)
    p3 = layers.Dense(32, activation='relu')(p3)

    output = layers.Dense(num_category, activation='softmax')(p3)

    model = Model(inputs=[inputs], outputs=[output], name='ConvSpeechModel')

    return model
