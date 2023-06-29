from datasets import load_dataset
import os.path as path

from data.reuters import BaseDataset
from utils.path_utils import PathUtils


class News20Dataset(BaseDataset):
    def __init__(self):
        super(News20Dataset, self).__init__()
        self.file_path = path.join(PathUtils.DATA_HOME_PATH, "20news.json")
        self.train_datasets = load_dataset("json", data_files=self.file_path)

        self.label_list = [_["label"] for _ in self.train_datasets["train"]]
        train_classes, valid_classes, test_classes = get_20newsgroup_classes()
        self.stage2classes_dict = {
            "train": train_classes,
            "valid": valid_classes,
            "test": test_classes
        }


def get_20newsgroup_classes():
    train_classes = list(range(5, 13))
    val_classes = list(range(5))
    test_classes = list(range(13, 20))

    return train_classes, val_classes, test_classes