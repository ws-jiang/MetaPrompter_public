from datasets import load_dataset
import os.path as path

from data.reuters import BaseDataset
from utils.path_utils import PathUtils


class Liu54Dataset(BaseDataset):
    def __init__(self):
        super(Liu54Dataset, self).__init__()
        self.file_path = path.join(PathUtils.DATA_HOME_PATH, "liu54.json")
        self.train_datasets = load_dataset("json", data_files=self.file_path)

        self.label_list = [_["label"] for _ in self.train_datasets["train"]]
        train_classes, valid_classes, test_classes = _get_liu54_classes()
        self.stage2classes_dict = {
            "train": train_classes,
            "valid": valid_classes,
            "test": test_classes
        }


def _get_liu54_classes():
    # "contrastnet: a contrastive learning framework for few-shot text classification" do not have a fixed split
    # we fix it here for fair comparison with baselines.
    class_ids = [53, 43, 41, 19, 7, 0, 6, 36, 14, 34, 52, 23, 24, 3, 37, 32, 30, 48, 21, 38, 17, 28, 29, 46, 18, 22, 44, 47, 40, 5, 51, 31, 27, 50, 12, 2, 8, 39, 4, 33, 26, 13, 11, 1, 25, 16, 49, 15, 45, 42, 20, 35, 9, 10]
    train_classes = class_ids[:18]
    val_classes = class_ids[18:18*2]
    test_classes = class_ids[18*2:18*3]

    return train_classes, val_classes, test_classes