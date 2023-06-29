
from enum import Enum

from data.hwu64 import Hwu64Dataset
from data.liu54 import Liu54Dataset
from data.reuters import ReutersDataset
from data.news20 import News20Dataset
from data.amazon import AmazonDataset
from data.huffpost import HuffPostDataset


class DatasetEnum(Enum):
    News20 = News20Dataset
    Amazon = AmazonDataset
    HuffPost = HuffPostDataset
    Reuters = ReutersDataset
    Hwu64 = Hwu64Dataset
    Liu54 = Liu54Dataset

    @classmethod
    def get_value_by_name(cls, ds_name):
        method_dict = dict([(_.name, _.value) for _ in DatasetEnum])
        return method_dict[ds_name]

    @classmethod
    def is_in(cls, ds_name):
        method_dict = dict([(_.name, _.value) for _ in DatasetEnum])
        return ds_name in method_dict
