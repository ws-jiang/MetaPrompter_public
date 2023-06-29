from enum import Enum

from method.verbalizer.rep_verbalizer import RepVerbalizer
from method.verbalizer.proto_verbalizer import ProtoVerbalizer
from method.verbalizer.soft_verbalizer import SoftVerbalizer
from method.verbalizer.hard_verbalizer import HardVerbalizer

from method.meta_prompter import MetaPrompter
from method.meta_prompting import MetaPromptingNoTuningLM
from method.meta_prompting import MetaPrompting


class MethodEnum(Enum):

    RepVerbalizer = RepVerbalizer
    ProtoVerbalizer = ProtoVerbalizer
    SoftVerbalizer = SoftVerbalizer
    HardVerbalizer = HardVerbalizer

    MetaPrompter = MetaPrompter
    MetaPromptingNoTuningLM = MetaPromptingNoTuningLM
    MetaPrompting = MetaPrompting

    @classmethod
    def get_value_by_name(cls, method_name):
        method_dict = dict([(_.name, _.value) for _ in MethodEnum])
        return method_dict[method_name]
