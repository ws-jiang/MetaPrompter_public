import path_init
import argparse
import copy
import torch

from data._dataset_enum import DatasetEnum
from method._method_enum import MethodEnum
from utils.config_utils import ConfigUtils
from utils.progress_utils import ExptEnum
from utils.time_utils import TimeUtils
from utils.torch_utils import TorchUtils


def run_expt(parser):
    ######################
    # load the configs
    ######################
    config_dict_template = ConfigUtils.get_basic_config()

    ######################
    # Loop over all seeds
    ######################
    for seed in parser.seeds:
        TorchUtils.set_random_seed(seed)
        parser.cur_seed = seed
        config_dict = copy.deepcopy(config_dict_template)
        job_id = "{}_{}_{}_{}_{}_{}".format(
            parser.method, parser.ds, parser.expt, parser.model_name, parser.job_type, seed)
        parser.job_id = job_id

        config_dict["args_parser"] = parser
        method_class = MethodEnum.get_value_by_name(method_name=parser.method)
        method_obj = method_class(config_dict)
        method_obj.logger.info(config_dict)

        if parser.eval:
            method_obj.eval(stage="test", step=0)
        else:
            method_obj.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ######################
    # config
    ######################
    ds = DatasetEnum.Reuters.name
    expt = ExptEnum.S_5_WAY_5_SHOT.value
    method = MethodEnum.MetaPrompter.name
    GPU_ID = 3

    ######################
    # run the code
    ######################
    parser.add_argument('--gpu_id', default=GPU_ID, type=int)
    parser.add_argument('--index_key', default=TimeUtils.get_now_str(fmt=TimeUtils.YYYYMMDDHHMMSS_COMPACT), type=str)
    parser.add_argument('--job_type', default="DEBUG", type=str)
    parser.add_argument('--ds', default=ds, type=str)
    parser.add_argument('--meta_batch_size', default=1, type=int)
    parser.add_argument('--eval_ft_step', default=5, type=int)
    parser.add_argument('--train_ft_step', default=5, type=int)
    parser.add_argument('--max_length', default=220, type=int)
    parser.add_argument('--prompt_length', default=8, type=int)
    parser.add_argument('--K', default=8, type=int)
    parser.add_argument('--meta_lr', default=1e-5, type=float)
    parser.add_argument('--base_lr', default=5e-3, type=float)
    parser.add_argument('--lambdax', default=0.5, type=float)
    parser.add_argument('--model_name', default="bert", type=str)
    parser.add_argument('--expt', default=expt, type=str)
    parser.add_argument('--method', default=method, type=str)
    parser.add_argument('--seeds', default=[1], nargs='+', type=int)
    parser.add_argument('--eval', action='store_true')

    args = parser.parse_args()
    print(args)

    device_id = args.gpu_id
    torch.cuda.set_device(device=device_id)
    run_expt(args)
