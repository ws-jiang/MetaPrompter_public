import torch
from torch import nn
from torch.nn import functional as F

from method.abstract_alg import AbstractAlg
from utils.evaluation_utils import count_acc
from utils.tensor_utils import split_support_query_for_x_in_cls


class SoftVerbalizer(AbstractAlg):
    """
    prompt + soft verbalizers
    """
    def __init__(self, config):
        self.args_parser = config["args_parser"]
        self.embed_dim = 768
        self.num_way = 5

        super(SoftVerbalizer, self).__init__(config=config)

    def meta_update(self, batch_list, stage, frac, summarizer):
        pass

    def meta_eval(self, batch, stage, summarizer):
        pass
        # refer the original paper
