from utils.path_utils import PathUtils

import yaml
import torch
import os


class ConfigUtils(object):

    def __init__(self):
        pass

    @staticmethod
    def get_device(device_id=0):
        device = torch.device("cpu")
        if torch.cuda.is_available():
            print("GPU is available, using GPU:{}".format(device_id))
            device = torch.device('cuda:{}'.format(device_id))
        else:
            print("GPU is unavailable, using CPU")
        return device

    @staticmethod
    def dump_config_file(config_file_name, config_dict):
        config_file_full_path = os.path.join(PathUtils.CONFIG_HOME_PATH, config_file_name)
        with open(config_file_full_path, 'w') as yaml_file:
            yaml.dump(config_dict, yaml_file, default_flow_style=False)

    @staticmethod
    def get_config_dict(config_file_name):
        config_file_full_path = os.path.join(PathUtils.CONFIG_HOME_PATH, config_file_name)
        return yaml.load(open(config_file_full_path, "r"), Loader=yaml.Loader)

    @staticmethod
    def get_basic_config():
        configs = ["few_shot_settings.yaml", "process_control_config.yaml"]
        config_dict = {}
        for config_file in configs:
            config_dict.update(ConfigUtils.get_config_dict(config_file))
        return config_dict
