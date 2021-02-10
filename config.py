import os
import logging
from speech_recognition.utils.configuration import TrainingConfiguration


def create_logger(name, level=logging.DEBUG):

    logg = logging.getLogger(name)
    logg.setLevel(level)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(
        logging.Formatter(
            '[%(name)s:%(filename)s:%(lineno)d] - [%(process)d] - %(asctime)s - %(levelname)s - %(message)s'
        )
    )

    logg.addHandler(stdout_handler)

    return logg


logger = create_logger("VOICE-RECOGNITION")

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

TRAINING_CONFIG_PATH = f"{CURRENT_DIR}/config/training.json"

trainingConfiguration = TrainingConfiguration(TRAINING_CONFIG_PATH)


class FileTypes(object):
    CSV = 1
    JSON = 2
    TXT = 3
