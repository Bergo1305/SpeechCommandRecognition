import json
from typing import List


class TrainingConfiguration(object):

    def __init__(self, config_path):
        with open(config_path, "r") as file:
            self.train_conf = json.load(file)

    @property
    def train_dataset_path(self) -> str:
        return self.train_conf.get("train_full_path")

    @property
    def validation_dataset_path(self) -> str:
        return self.train_conf.get("val_full_path")

    @property
    def test_dataset_path(self) -> str:
        return self.train_conf.get("test_full_path")

    @property
    def training_filename(self) -> str:
        return self.train_conf.get("training_file_name")

    @property
    def validation_filename(self) -> str:
        return self.train_conf.get("val_file_name")

    @property
    def test_filename(self) -> str:
        return self.train_conf.get("test_file_name")

    @property
    def base_model(self) -> str:
        return self.train_conf.get("base_model")

    @property
    def sampling_rate(self) -> int:
        return self.train_conf.get("sampling_rate")

    @property
    def num_epochs(self) -> int:
        return self.train_conf.get("epochs")

    @property
    def optimizer(self) -> str:
        return self.train_conf.get("optimizer")

    @property
    def learning_rate(self) -> float:
        return self.train_conf.get("learning_rate")

    @property
    def decay_rate(self) -> float:
        return self.train_conf.get("decay_rate")

    @property
    def decay_epochs(self) -> int:
        return self.train_conf.get("decay_epoch")

    @property
    def patience(self) -> int:
        return self.train_conf.get("patience")

    @property
    def classes_of_interests(self) -> List[str]:
        return self.train_conf.get("classes_of_interests")

    @property
    def background_classes(self) -> List[str]:
        return self.train_conf.get("background_classes")

    @property
    def num_classes(self) -> int:
        return len(self.train_conf.get("classes_of_interests"))

    @property
    def label_map_config(self) -> str:
        with open(self.train_conf.get("label_map_config"), "r") as file:
            mapper = json.load(file)

        return mapper

    @property
    def batch_size(self) -> int:
        return self.train_conf.get("batch_size")

    @property
    def early_stopping(self) -> bool:
        return self.train_conf.get("early_stopping")

    @property
    def model_checkpoint(self) -> bool:
        return self.train_conf.get("model_checkpoint")

    @property
    def loss(self) -> str:
        return self.train_conf.get("loss")

