from datasets import load_dataset
import os.path as path

from data.reuters import BaseDataset
from utils.path_utils import PathUtils


class Hwu64Dataset(BaseDataset):
    def __init__(self):
        super(Hwu64Dataset, self).__init__()
        self.file_path = path.join(PathUtils.DATA_HOME_PATH, "hwu64.json")
        self.train_datasets = load_dataset("json", data_files=self.file_path)

        self.label_list = [_["label"] for _ in self.train_datasets["train"]]
        train_classes, valid_classes, test_classes = _get_hwu64_classes()
        self.stage2classes_dict = {
            "train": train_classes,
            "valid": valid_classes,
            "test": test_classes
        }


def _get_hwu64_classes():
    # "contrastnet: a contrastive learning framework for few-shot text classification" do not have a fixed split
    # we fix it here for fair comparison with baselines.

    class_ids = [36, 0, 50, 59, 23, 52, 35, 28, 53, 32, 1, 30, 46, 27, 9, 10, 13, 60, 29, 54, 17, 62, 44, 57, 12, 14, 61, 16, 2, 25, 19, 48, 6, 55, 24, 5, 42, 33, 41, 58, 4, 38, 3, 39, 37, 20, 26, 51, 47, 21, 56, 31, 49, 34, 7, 63, 11, 18, 43, 22, 8, 45, 15, 40]
    train_classes = class_ids[:23]
    val_classes = class_ids[23:39]
    test_classes = class_ids[39:64]

    return train_classes, val_classes, test_classes