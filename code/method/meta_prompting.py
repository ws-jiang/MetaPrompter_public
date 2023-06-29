import torch
from torch.nn import functional as F

from backbone.prompt_model import ContinuousPrompt
from method.abstract_alg import AbstractAlg
from utils.evaluation_utils import count_acc


class MetaPrompting(AbstractAlg):
    def __init__(self, config):
        super(MetaPrompting, self).__init__(config=config)

        pass
        # refer the original paper

    def get_method(self):
        return f"{self.method}_{self.base_lr}"

    def get_identifier(self):
        return "{}_{}_{}_{}_{}".format(self.job_id, self.meta_lr, self.base_lr, self.train_ft_step, self.eval_ft_step)

    def meta_update(self, batch_list, stage, frac, summarizer):
        pass
        # refer the original paper

class MetaPromptingNoTuningLM(MetaPrompting):
    def __init__(self, config):
        super(MetaPromptingNoTuningLM, self).__init__(config=config)

        pass
        # refer the original paper
