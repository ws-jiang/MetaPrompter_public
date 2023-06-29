
from data._dataset_enum import DatasetEnum


def get_dataset(ds_name):
    if DatasetEnum.is_in(ds_name):
        return DatasetEnum.get_value_by_name(ds_name)()
    else:
        raise ValueError("unknown dataset: {}".format(ds_name))
