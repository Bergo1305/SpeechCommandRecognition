import os
import json
import csv
import glob
import math
import shutil
import tarfile
import random
import requests

from tqdm import tqdm
from typing import List
from config import logger
from config import FileTypes


def download_file_tar(url, tar_file):
    r = requests.get(url, stream=True)

    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0

    logger.info(f"Downloading {url} to {tar_file}")

    with open(tar_file, 'wb') as f:
        for data in tqdm(
                r.iter_content(block_size),
                total=math.ceil(total_size // block_size),
                unit='KB',
                unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)

    if total_size != 0 and wrote != total_size:
        logger.error(f"Error, something went wrong...")


def _extract_tar(file_name, folder):
    logger.info(f"Extracting {file_name} to {folder}")
    tar = tarfile.open(file_name, "r:gz")
    tar.extractall(path=folder)
    tar.close()


def download_google_speech_v2(save_dir):
    if os.path.isdir(save_dir):
        logger.warning("Provided directory is not empty.")
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    train_files = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
    test_files = 'http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz'

    download_file_tar(train_files, f"{save_dir}/train.tar.gz")
    download_file_tar(test_files, f"{save_dir}/test.tar.gz")

    if not os.path.isdir(f"{save_dir}/test/"):
        _extract_tar(f"{save_dir}/test.tar.gz", f"{save_dir}/test/")

    if not os.path.isdir(f"{save_dir}/train/"):
        _extract_tar(f"{save_dir}/train.tar.gz", f"{save_dir}/train/")


def create_label_map(train_dir, included_labels: List[str] = None, save_config=None):
    """" Create labeling mapper between label and and ID. (ID will be generated randomly.

          Parameters
          ----------
          :param train_dir: str
          Dictionary of dataset.

          :param included_labels: List[Str]
          List of classes used for mapping (optional)

          :param save_config: boolean
          Save json mapper (optional)
      """
    labels = [dir_path.split('/')[-1] for dir_path in [x[0] for x in os.walk(train_dir)][1:]]

    if included_labels:
        if set(labels) & set(included_labels) != set(included_labels):
            raise Exception("Provided classes are not in train dataset.")
        labels = included_labels

    labels = list(filter(lambda label: label not in TrainingConfiguration.background_classes, labels))

    labels_map = {label: idx for idx, label in enumerate(labels)}

    if save_config:
        with open(f"{save_config}", "w") as file:
            json.dump(labels_map, file)

    return labels_map


def _save(images_dict, file_type, filename):
    """" Save file based on dict mapper[img -> label] with provided filetype and name.

        Parameters
        ----------
        :param images_dict: Dict
        Dictionary with image path as key and label as value.

        :param file_type: FileTypes
        Save format of file.

        :param filename: str
        Name of file to be saved
    """

    _list = list(images_dict.items())
    random.shuffle(_list)
    images_dict = dict(_list)

    for image_path, label in images_dict.items():

        if file_type == FileTypes.CSV:

            with open(f"{filename}.csv", "a+") as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow([image_path, label])

        if file_type == FileTypes.TXT:

            with open(f"{filename}.txt", "a+") as file:
                file.write(f"{image_path}, {label}")

        if file_type == FileTypes.JSON:

            with open(f"{filename}.json", "a+") as file:
                json.dump(images_dict, file)


def create_dataset_files(
        dataset_path,
        split_ratio,
        file_type: FileTypes = FileTypes.CSV,
        included_labels=None
):
    """Load dataset and create training, validation and test files.

        Parameters
        ----------
        :param dataset_path : string
            This file needs to have train/test folders. If not create it manually.

        :param split_ratio: float
            Split ratio between train and validation set.

        :param file_type   : FileTypes
            Output format of file. It can be txt, json or csv. This file will contain path of image
            (real path from dataset folder) plus containing label of image.

        :param included_labels : List[str]
            Create files based on specific labels for training

        Returns
        -------
        It will create three files train.{format}, validation.{format} and test.{format}.
        First two files will be saved in dataset_dir/train and last one in dataset_dir/test.
        """

    train_images = []
    test_images = []

    if dataset_path[-1] == "/":
        dataset_path = dataset_path[:-1]

    if included_labels:
        label_map = create_label_map(dataset_path, included_labels, "labelmap_included.json")
        for label in included_labels:
            train_images.extend(
                glob.glob(f"{dataset_path}/train/{label}/*wav")
            )
            test_images.extend(
                glob.glob(f"{dataset_path}/test/{label}/*wav")
            )
    else:
        label_map = create_label_map(f"{dataset_path}/train", None, None)
        train_images = glob.glob(f"{dataset_path}/train/*/*.wav")
        test_images = glob.glob(f"{dataset_path}/test/*/*wav")

    train_mapper = {}
    test_dict = {}

    for test_image in test_images:
        label = test_image.split('/')[-2]
        if label_map.get(label, None) is not None:
            real_path = '/'.join(t for t in test_image.split('/')[-4:])
            test_dict[real_path] = label_map.get(label)

    for train_image in train_images:
        label = train_image.split('/')[-2]
        mapped_label = label_map.get(label, None)

        if mapped_label is not None:
            real_path = '/'.join(t for t in train_image.split('/')[-4:])
            if train_mapper.get(mapped_label, None) is None:
                train_mapper[mapped_label] = [real_path]
            else:
                train_mapper[mapped_label].append(real_path)

    train_dict = {}
    validation_dict = {}

    for label, images in train_mapper.items():
        random.shuffle(images)
        val_size = int(split_ratio * len(images))
        train_size = len(images) - val_size

        _train = images[0:train_size]
        _val = images[train_size:]

        for _img in _train:
            train_dict[_img] = label

        for _img in _val:
            validation_dict[_img] = label

    if included_labels:
        _save(test_dict, file_type, "test_included")
        _save(train_dict, file_type, "train_included")
        _save(validation_dict, file_type, "validation_included")

    else:

        _save(test_dict, file_type, "test")
        _save(train_dict, file_type, "train")
        _save(validation_dict, file_type, "validation")
