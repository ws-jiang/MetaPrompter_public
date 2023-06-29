import torch
import numpy as np

def dump_list_2_json(data_list):
    import json
    return json.dumps(data_list)

def relabel_id(labels):
    """
    input: [1, 1, 2, 2, 0]
    output: [0, 0, 1, 1, 2]
    """
    label_map ={}
    results = []
    relabel_id = 0
    for i in labels:
        if i not in label_map:
            label_map[i] = relabel_id
            relabel_id += 1
        results.append(label_map[i])
    return results

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))


def split_support_query_reg(x_, y_, n_support):
    return x_[: n_support, :], y_[: n_support], \
           x_[n_support:, :], y_[n_support:]


def rand_simplex(k):
    result_array = np.random.dirichlet((1,)*k)
    return result_array, torch.FloatTensor(result_array)

def split_support_query_xy(x, y, n_support):
    return x[: n_support, :], y[: n_support, :], \
           x[n_support:, :], y[n_support:, :],

def split_support_query_x_only(x, n_support):
    return x[: n_support, :], x[n_support:, :]

def get_all_weights_as_numpy(model):
    para_arrays = []
    total_length = 0
    info_dict = {}
    for name, param in model.named_parameters():
        paras = param.flatten().data.cpu().numpy()
        para_arrays.append(paras)
        info_dict[name] = (total_length, total_length + len(paras), param.shape)
        total_length += len(paras)
    return np.hstack(para_arrays), info_dict

def split_support_query(x, n_support, n_query, n_way):
    """
    x: n_sample * shape
    :param x:
    :param n_support:
    :return:
    """
    x_reshaped = x.contiguous().view(n_way, n_support + n_query, *x.shape[1:])
    x_support = x_reshaped[:, :n_support].contiguous().view(n_way * n_support, *x.shape[1:])
    x_query = x_reshaped[:, n_support:].contiguous().view(n_way * n_query, *x.shape[1:])
    return x_support, x_query

def split_support_query_for_x_in_cls(z, n_support):
    """
    z: n_way * n_shot * shape
    :param z:
    :param n_support:
    :return:
    """
    z_support = z[:, :n_support]
    z_query = z[:, n_support:]
    return z_support, z_query

def accumulate_sum(input_list):
    """
    inputs = [1, 2, 3, 4]
    return = [1, 3, 6, 10]
    :param input_list:
    :return:
    """
    total = 0
    result = []
    for x in input_list:
        total += x
        result.append(total)
    return result