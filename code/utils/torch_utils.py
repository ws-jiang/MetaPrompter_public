import torch
import numpy as np

def np_to_string(np_array, precision=2):
    """
    convert numpy array to string, for printing
    :param np_array:
    :return:
    """
    return np.array2string(np_array, precision=precision, separator=", ")


class TorchUtils(object):
    def __init__(self):
        pass

    @staticmethod
    def compute_num_params(m):
        return sum([np.prod(p.size()) for p in m.parameters()])

    @staticmethod
    def set_device(device_id):
        torch.cuda.set_device(device=device_id)

    @staticmethod
    def set_random_seed(seed):
        seed_id = seed
        torch.manual_seed(seed_id)
        np.random.seed(seed_id)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def reset_grad(model, grad_dict):
        for name, param in model.named_parameters():
            if name not in grad_dict:
                raise ValueError("cannot find {} in reset_grad".format(name))
            else:
                param.grad.copy_(grad_dict[name])

    @staticmethod
    def copy_model_parameters(module_src, module_dest):
        params_src = module_src.named_parameters()
        params_dest = module_dest.named_parameters()

        dict_dest = dict(params_dest)

        for name, param in params_src:
            if name in dict_dest:
                dict_dest[name].data = param.data.detach().clone()

    @staticmethod
    def update_model_parameters(module_src, module_dest, lr=0.001):
        params_src = module_src.named_parameters()
        params_dest = module_dest.named_parameters()

        dict_dest = dict(params_dest)

        for name, param in params_src:
            if name in dict_dest:
                dict_dest[name].data.copy_((1-lr) * dict_dest[name].data + lr * param.data)

    @staticmethod
    def get_model_diff(model_src, model_dest):
        params_src = model_src.named_parameters()
        params_dest = model_dest.named_parameters()

        dict_dest = dict(params_dest)
        model_diff_dict = {}
        for name, param in params_src:
            if name in dict_dest:
                model_diff_dict[name] = (param.data - dict_dest[name].data).detach().clone()
        return model_diff_dict

    @staticmethod
    def compute_model_distance(model_src, model_dest):
        return sum([torch.sum(torch.pow(x-y,2)) for x,y in zip(model_src.parameters(), model_dest.parameters())])

    @staticmethod
    def set_parameter_requires_grad(model, enable=True):
        for param in model.parameters():
            param.requires_grad = enable
