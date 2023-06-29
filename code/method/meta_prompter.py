import math

import torch
from torch.nn import functional as F

from backbone.prompt_model import ContinuousPromptPool
from method.abstract_alg import AbstractAlg
from utils.evaluation_utils import count_acc
from utils.kernel_utils import KernelZoo as K
from utils.tensor_utils import split_support_query_for_x_in_cls, split_support_query


class MetaPrompter(AbstractAlg):
    def __init__(self, config):
        self.args_parser = config["args_parser"]
        self.lambdax = self.args_parser.lambdax
        self.prompt_length = self.args_parser.prompt_length
        self.K = self.args_parser.K
        super(MetaPrompter, self).__init__(config=config)

        self.embed_dim = 768
        self.prompt_model = ContinuousPromptPool({"K": self.K, "prompt_len": self.prompt_length, "embed_size": self.embed_dim},
                                             init_emb=self.meta_model.bert.embeddings.word_embeddings(self.get_label_words_id()[0:self.K*self.prompt_length]))
        self.prompt_model.cuda()

        self.frozen_model = self.get_model()
        self.frozen_model.cuda()

        for param in self.frozen_model.parameters(): # for query function
            param.requires_grad = False

        for param in self.meta_model.parameters():
            param.requires_grad = False

        self.meta_optimizer = torch.optim.AdamW(self.prompt_model.parameters(), lr=self.meta_lr)

    def __tokenize_func(self, batched_text):
        return self.tokenizer(batched_text['raw'], truncation=True, padding=True, max_length=self.args_parser.max_length)

    def get_identifier(self):
        return "{}_{}_{}_{}_{}_{}".format(self.job_id, self.meta_lr, self.base_lr, self.lambdax, self.prompt_length, self.K)

    def get_method(self):
        return "{}_{}_{}_{}_{}".format(self.method, self.lambdax, self.K, self.prompt_length, self.base_lr)

    def meta_update(self, batch_list, stage, frac, summarizer):
        temperature = math.sqrt(self.embed_dim)
        n_way, n_support, n_query, y_support, y_query = self.get_basic_expt_info(stage)
        ft_step = self.train_ft_step if stage == "train" else self.eval_ft_step

        all_parameters = list(self.prompt_model.parameters())
        self.base_optimizer = torch.optim.SGD(self.prompt_model.parameters(), lr=self.base_lr)

        for param in all_parameters:
            param.meta_data = param.data.clone()
            param.grad_list = []

        batch = batch_list[0]
        self.add_addition_tokens(batch)
        support_batch, query_batch = self.split_batch(batch)
        y_support_label, y_query_label = support_batch['labels'], query_batch['labels']
        label_words, label_words_attention = self.get_words(y_support_label)
        del support_batch['labels'], query_batch['labels'], batch['labels']

        batch["output_hidden_states"] = True
        with torch.no_grad():
            output = self.frozen_model(**batch)
            z = output["hidden_states"][-1]
            sample_embeddings = z[batch["input_ids"] == self.tokenizer.mask_token_id, :]
            supp_emb, query_emb = split_support_query(sample_embeddings, n_support, n_query, n_way)

        for i in range(ft_step):
            raw_embeds = self.meta_model.bert.embeddings(input_ids=support_batch["input_ids"])
            weight = F.softmax(supp_emb.matmul(self.prompt_model.keys.T) / temperature)
            prompt = weight.matmul(self.prompt_model.prompt_embeddings).contiguous().view(supp_emb.size(0), self.prompt_length, self.embed_dim)

            input_embedding, attention_mask = self.wrap_input(support_batch, raw_embeds, prompt)
            revised_batch = {'inputs_embeds': input_embedding, 'attention_mask': attention_mask,
                             "output_hidden_states": True}

            support_output = self.meta_model(**revised_batch)

            z = support_output["hidden_states"][-1]  # last layer
            z = z[:, self.prompt_length:, :][support_batch["input_ids"] == self.tokenizer.mask_token_id,:]
            z = z - z.mean(0)
            z_support = z.contiguous().view(n_way, n_support, -1)
            protos = z_support.contiguous().view(n_way, n_support, -1).mean(dim=1)
            z_support = z_support.contiguous().view(n_way * n_support, -1)
            y_support_pred_soft = K.compute_cosine(z_support, protos)

            #  pred from hard verbalizer
            logits = support_output.logits
            mask_logits = logits[:, self.prompt_length:, :][
                          support_batch["input_ids"] == self.tokenizer.mask_token_id, :]
            y_support_pred_hard = mask_logits[:, label_words] @ label_words_attention

            y_support_pred = self.lambdax * F.softmax(y_support_pred_soft, dim=1) \
                       + (1-self.lambdax) * F.softmax(y_support_pred_hard, dim=1)

            support_loss = F.nll_loss(y_support_pred.log(), y_support)
            support_acc = count_acc(y_support_pred, y_support)

            self.meta_model.zero_grad()
            self.base_optimizer.zero_grad()
            support_loss.backward()
            self.base_optimizer.step()

        raw_embeds = self.meta_model.bert.embeddings(input_ids=batch["input_ids"])
        weight = F.softmax(sample_embeddings.matmul(self.prompt_model.keys.T) / temperature)
        prompt = weight.matmul(self.prompt_model.prompt_embeddings).contiguous().view(sample_embeddings.size(0),
                                                                                      self.prompt_length,
                                                                                      self.embed_dim)

        input_embedding, attention_mask = self.wrap_input(batch, raw_embeds, prompt)
        revised_batch = {'inputs_embeds': input_embedding, 'attention_mask': attention_mask,
                         "output_hidden_states": True}

        output = self.meta_model(**revised_batch)

        # (1) pred from soft verbalizer
        z = output["hidden_states"][-1]
        z = z[:, self.prompt_length:, :][batch["input_ids"] == self.tokenizer.mask_token_id,:]
        z = z - z.mean(0)
        z = z.contiguous().view(n_way, n_support + n_query, -1)
        z_support, z_query = split_support_query_for_x_in_cls(z, n_support=n_support)
        protos = z_support.contiguous().view(n_way, n_support, -1).mean(dim=1)
        z_query = z_query.contiguous().view(n_way * n_query, -1)
        y_query_pred_soft = K.compute_cosine(z_query, protos)

        # (2) pred from hard verbalizer
        logits = output.logits.contiguous().view(n_way, n_support + n_query, *output.logits.shape[1:])
        _, query_logits = split_support_query_for_x_in_cls(logits, n_support=n_support)
        query_logits = query_logits.contiguous().view(n_way*n_query, *query_logits.shape[2:])
        mask_logits = query_logits[:, self.prompt_length:, :][query_batch["input_ids"] == self.tokenizer.mask_token_id, :]
        y_query_pred_hard = mask_logits[:, label_words] @ label_words_attention

        # (3) final pred
        y_query_pred = self.lambdax*F.softmax(10*y_query_pred_soft, dim=1) \
                       + (1-self.lambdax) * F.softmax(y_query_pred_hard, dim=1)

        query_loss = F.nll_loss(y_query_pred.log(), y_query)
        query_acc = count_acc(y_query_pred, y_query)
        meta_loss = query_loss

        if stage == "train":
            self.meta_model.zero_grad()
            self.base_optimizer.zero_grad()

            meta_loss.backward()

            for param in all_parameters:
                param.data = param.meta_data.clone()
            self.meta_optimizer.step()

            self.meta_model.zero_grad()
            self.meta_optimizer.zero_grad()
        else:
            for param in all_parameters:
                param.data = param.meta_data.clone()

        summarizer.add("support_acc", support_acc)
        summarizer.add("query_acc", query_acc)
        summarizer.add("support_loss", support_loss.item())
        summarizer.add("query_loss", query_loss.item())
        summarizer.add("meta_loss", meta_loss.item())

