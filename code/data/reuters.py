from datasets import load_dataset
import os.path as path
from utils.path_utils import PathUtils


class BaseDataset():
    def __init__(self):
        self.max_len = 200


class ReutersDataset(BaseDataset):
    def __init__(self):
        super(ReutersDataset, self).__init__()
        self.file_path = path.join(PathUtils.DATA_HOME_PATH, "reuters.json")
        self.train_datasets = load_dataset("json", data_files=self.file_path)

        self.label_list = [_["label"] for _ in self.train_datasets["train"]]

        train_classes = list(range(15))
        val_classes = list(range(15, 20))
        test_classes = list(range(20, 31))

        self.stage2classes_dict = {
            "train": train_classes,
            "valid": val_classes,
            "test": test_classes
        }
