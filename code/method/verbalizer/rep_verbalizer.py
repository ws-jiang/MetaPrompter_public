import torch

from method.abstract_alg import AbstractAlg
from torch.nn import functional as F

from utils.evaluation_utils import count_acc
from utils.tensor_utils import split_support_query_for_x_in_cls
from utils.kernel_utils import KernelZoo as K


class RepVerbalizer(AbstractAlg):
    def __init__(self, config):
        self.args_parser = config["args_parser"]
        self.lambdax = self.args_parser.lambdax
        super(RepVerbalizer, self).__init__(config=config)

    def get_method(self):
        return "{}_{}".format(self.method, self.lambdax)

    def get_identifier(self):
        return "{}_{}_{}_{}".format(self.job_id, self.meta_lr, self.base_lr, self.lambdax)

    def meta_update(self, batch_list, stage, frac, summarizer):
        pass

    def meta_eval(self, batch, stage, summarizer):
        n_way, n_support, n_query, y_support, y_query = self.get_basic_expt_info(stage)
        ft_step = self.eval_ft_step

        all_parameters = list(self.meta_model.parameters())
        self.base_optimizer = torch.optim.SGD(all_parameters, lr=self.base_lr)

        # backup PLM
        for param in all_parameters:
            param.meta_data = param.data.clone()

        self.add_addition_tokens(batch)
        support_batch, query_batch = self.split_batch(batch)

        y_support_label, y_query_label = support_batch['labels'], query_batch['labels']
        label_words, label_words_attention = self.get_words(y_support_label)
        del support_batch['labels'], query_batch['labels'], batch['labels']

        for i in range(ft_step):
            support_batch["output_hidden_states"] = True
            support_output = self.meta_model(**support_batch)

            # (1) pred from soft verbalizer
            z = support_output["hidden_states"][-1]  # last hidden layer
            z = z[support_batch["input_ids"] == self.tokenizer.mask_token_id, :]
            # support_mean = z.mean(0)
            z = z - z.mean(0)
            z_support = z.contiguous().view(n_way, n_support, -1)
            protos = z_support.contiguous().view(n_way, n_support, -1).mean(dim=1)
            z_support = z_support.contiguous().view(n_way * n_support, -1)
            y_support_pred_soft = K.compute_cosine(z_support, protos)

            # (2) pred from hard verbalizer
            logits = support_output.logits
            mask_logits = logits[support_batch["input_ids"] == self.tokenizer.mask_token_id, :]
            y_support_pred_hard = mask_logits[:, label_words] @ label_words_attention

            # (3) final pred
            y_support_pred = self.lambdax * F.softmax(y_support_pred_soft, dim=1) + (1 - self.lambdax) * F.softmax(
                y_support_pred_hard, dim=1)

            support_loss = F.nll_loss(y_support_pred.log(), y_support)
            support_acc = count_acc(y_support_pred, y_support)

            self.base_optimizer.zero_grad()
            support_loss.backward()
            self.base_optimizer.step()

        batch["output_hidden_states"] = True
        output = self.meta_model(**batch)

        # (1) pred from soft verbalizer
        z = output["hidden_states"][-1]
        z = z[batch["input_ids"] == self.tokenizer.mask_token_id,:]
        z = z - z.mean(0)
        z = z.contiguous().view(n_way, n_support + n_query, -1)
        x_support, x_query = split_support_query_for_x_in_cls(z, n_support=n_support)
        protos = x_support.contiguous().view(n_way, n_support, -1).mean(dim=1)
        x_query = x_query.contiguous().view(n_way * n_query, -1)
        y_query_pred_soft = K.compute_cosine(x_query, protos)

        # (2) pref from hard verbalizer
        logits = output.logits.contiguous().view(n_way, n_support + n_query, *output.logits.shape[1:])
        _, query_logits = split_support_query_for_x_in_cls(logits, n_support=n_support)
        query_logits = query_logits.contiguous().view(n_way*n_query, *query_logits.shape[2:])
        mask_logits = query_logits[query_batch["input_ids"] == self.tokenizer.mask_token_id, :]
        y_query_pred_hard = mask_logits[:, label_words] @ label_words_attention

        # (3) final pred
        y_query_pred = self.lambdax*F.softmax(y_query_pred_soft, dim=1) + (1-self.lambdax) * F.softmax(y_query_pred_hard, dim=1)

        query_loss = F.nll_loss(y_query_pred.log(), y_query)
        query_acc = count_acc(y_query_pred, y_query)
        meta_loss = query_loss

        # reset the LM parameters
        for param in all_parameters:
            param.data = param.meta_data.clone()

        self.base_optimizer.zero_grad()

        summarizer.add("support_acc", support_acc)
        summarizer.add("query_acc", query_acc)
        summarizer.add("support_loss", support_loss.item())
        summarizer.add("query_loss", query_loss.item())
        summarizer.add("meta_loss", meta_loss.item())