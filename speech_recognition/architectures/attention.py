from tensorflow.keras.models import Model
from tensorflow.keras import layers, backend as K
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D


def attention_speech_model(num_category, sampling_rate=16000, input_length=16000):

    inputs = layers.Input((input_length, ), name='input')
    x = layers.Reshape((1, -1))(inputs)

    m = Melspectrogram(
        input_shape=(1, input_length),
        n_dft=1024,
        n_hop=128,
        padding='same',
        sr=sampling_rate,
        n_mels=80,
        fmin=40.0,
        fmax=sampling_rate / 2,
        power_melgram=1.0,
        return_decibel_melgram=True,
        trainable_fb=False,
        trainable_kernel=False,
        name='mel_tft'
    )
    m.trainable = False
    x = m(x)

    x = Normalization2D(int_axis=0, name='norm')(x)
    x = layers.Permute((2, 1, 3))(x)

    x = layers.Conv2D(10, (5, 1), activation='relu', padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(1, (5, 1), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Lambda(lambda t: K.squeeze(t, -1), name='squeeze_last_dim')(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    x_first = layers.Lambda(lambda t: t[:, t.shape[1]//2])(x)
    query = layers.Dense(128)(x_first)

    attention_scores = layers.Dot([1, 2])([query, x])
    attention_scores = layers.Softmax(name='attention_softmax')(attention_scores)
    attention_vector = layers.Dot(axes=[1, 1])([attention_scores, x])

    x = layers.Dense(64)(attention_vector)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(32)(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(num_category, activation='softmax', name="output")(x)
    model = Model(inputs=inputs, outputs=out)
    return model
