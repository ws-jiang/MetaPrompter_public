from abc import ABC, abstractmethod

import numpy as np
import copy
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForMaskedLM

from utils import service_utils
from utils.config_utils import ConfigUtils
from utils.evaluation_utils import Summarization
from utils.log_utils import LogUtils
from utils.path_utils import PathUtils
from utils.progress_utils import ProgressConfig
from data.dataset_utils import get_dataset
from data.fsl_samplers import NWayKShotSampler
from utils.tensor_utils import split_support_query
from utils.time_utils import TimeUtils


class AbstractAlg(ABC):
    def __init__(self, config):
        self.config = config

        self.args_parser = self.config["args_parser"]
        self.model_name = self.args_parser.model_name

        self.meta_lr = self.args_parser.meta_lr
        self.base_lr = self.args_parser.base_lr

        self.train_ft_step = self.args_parser.train_ft_step
        self.eval_ft_step = self.args_parser.eval_ft_step

        self.step = 0

        # config mysql for reporting data
        if self.config["use_mysql"]:
            self.mysql_connect = service_utils.get_mysql_conn()

        self.method = self.args_parser.method

        self.seed = self.args_parser.cur_seed
        self.stages = ["train", "valid", "test"]
        self.experiment_config = config[self.args_parser.expt]
        self.ways = {x: self.experiment_config["way"][x] for x in self.stages}
        self.n_support_dict = self.experiment_config["shot"]["support"]
        self.n_query_dict = self.experiment_config["shot"]["query"]

        self.index_key = self.args_parser.index_key
        self.job_id = self.args_parser.job_id

        self.task_name = self.method
        self.logger = LogUtils.get_or_init_logger(dir_name=self.task_name, file_name=self.get_identifier())

        self.meta_batch_size = self.args_parser.meta_batch_size

        self.progress = ProgressConfig(self.config)

        self.ds = self.args_parser.ds
        self.m_dataset = get_dataset(self.ds)
        self.checkpoint = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        tokenized_datasets = self.m_dataset.train_datasets.map(self.__tokenize_func, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["raw"])
        if self.ds in ["Reuters", "News20", "Amazon", "HuffPost"]:
            self.added_token_ids = self.tokenizer("Topic is [MASK].", return_tensors='pt')["input_ids"].cuda()
        else:  # "about" is from https://aclanthology.org/2022.acl-long.483.pdf
            self.added_token_ids = self.tokenizer("Topic is about [MASK].", return_tensors='pt')["input_ids"].cuda()
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        tokenized_datasets.set_format("torch")

        self.m_sampler = {stage: NWayKShotSampler(self.m_dataset.label_list,
                                                  target_labels=self.m_dataset.stage2classes_dict[stage],
                                                  total_steps=self.config["num_iteration"][stage]*self.get_meta_batch_size(stage),
                                                  n_way=self.ways[stage],
                                                  k_shot=self.n_support_dict[stage] + self.n_query_dict[stage],)
                          for stage in self.stages}
        self.m_dataloader = {
            stage: DataLoader(tokenized_datasets['train'], batch_sampler=self.m_sampler[stage],
                              collate_fn=data_collator,
                              num_workers=self.progress.num_workers,
                              pin_memory=True)
            for stage in self.stages
        }

        self.meta_model = self.get_model()
        self.meta_model.cuda()
        self.optimizer = torch.optim.AdamW(self.meta_model.parameters(), lr=self.meta_lr)

        self.added_token_id_len = self.added_token_ids.size(1)

        self.verbalize_dict = ConfigUtils.get_config_dict("hard_verbalizers_tokenized.yaml")[self.ds]

        self.prompt_model = None

    def wrap_input(self, batch, raw_embeds, prompt):
        attention_mask = batch["attention_mask"]

        input_embedding = torch.cat(
            (raw_embeds[:, : - self.added_token_id_len, :],
             prompt,
             raw_embeds[:, -self.added_token_id_len:, :]), dim=1)
        prompt_attention_mask = torch.ones(raw_embeds.size(0), self.prompt_length).cuda()
        attention_mask = torch.cat((attention_mask[:, :-self.added_token_id_len], prompt_attention_mask,
                                    attention_mask[:, -self.added_token_id_len:]), dim=1)

        return input_embedding, attention_mask

    def get_label_words(self, label_tensor):
        labels = list(set(label_tensor.cpu().tolist()))
        labels = sorted(labels)
        seq = " ".join([" ".join(self.verbalize_dict[l]) for l in labels])
        return seq.split()

    def get_label_words_id(self):
        labels = self.verbalize_dict.keys()
        seq = " ".join([" ".join(self.verbalize_dict[l]) for l in labels])
        tokens = self.tokenizer.tokenize(seq)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = list(set(ids))
        np.random.shuffle(ids)
        ids_list = []
        for i in range(20):
            id_ = copy.deepcopy(ids)
            np.random.shuffle(id_)
            ids_list.append(id_)
        word_ids = torch.LongTensor(np.concatenate(ids_list)).cuda()
        return word_ids

    def get_words(self, label_tensor):
        labels = list(set(label_tensor.cpu().tolist()))
        labels = sorted(labels)
        seq = " ".join([" ".join(self.verbalize_dict[l]) for l in labels])
        tokens = self.tokenizer.tokenize(seq)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        word_ids = torch.LongTensor(list(set(ids))).cuda()

        label_words_attention = []
        for label_id in labels:
            tokens = self.tokenizer.tokenize(" ".join(self.verbalize_dict[label_id]))
            ids = list(set(self.tokenizer.convert_tokens_to_ids(tokens)))
            label_words_attention.append(torch.isin(word_ids, torch.LongTensor(ids).cuda()) * 1/len(ids))

        return word_ids, torch.stack(label_words_attention, dim=1)

    def __tokenize_func(self, batched_text):
        return self.tokenizer(batched_text['raw'], truncation=True, padding=True, max_length=self.args_parser.max_length)

    def get_meta_batch_size(self, stage):
        return self.meta_batch_size if stage == "train" else 1

    def get_model(self):
        return AutoModelForMaskedLM.from_pretrained(self.checkpoint)

    def get_method(self):
        return self.method

    def get_identifier(self):
        return "{}_{}_{}".format(self.job_id, self.meta_lr, self.base_lr)

    def meta_eval(self, batch, stage, summarizer):
        return self.meta_update([batch], stage, 1, summarizer)

    @abstractmethod
    def meta_update(self, batch_list, stage, frac, summarizer):
        pass

    def get_best_model(self):
        return "{}_{}".format(self.index_key, "best")

    def prepare_eval(self):
        self.meta_model.train(False)

    def prepare_train(self):
        self.meta_model.train(True)

    def post_update(self):
        pass

    def get_basic_expt_info(self, stage):
        n_way = self.ways[stage]
        n_support = self.n_support_dict[stage]
        n_query = self.n_query_dict[stage]
        y_support = torch.from_numpy(np.repeat(range(n_way), n_support)).cuda()
        y_query = torch.from_numpy(np.repeat(range(n_way), n_query)).cuda()
        return n_way, n_support, n_query, y_support, y_query

    def split_batch(self, batch):
        stage = "train"
        n_way = self.ways[stage]
        n_support = self.n_support_dict[stage]
        n_query = self.n_query_dict[stage]
        support_batch = {}
        query_batch = {}
        for k in batch.keys():
            support, query = split_support_query(batch[k], n_support, n_query, n_way)
            support_batch[k] = support
            query_batch[k] = query
        return support_batch, query_batch

    def eval(self, stage, step):
        self.prepare_eval()
        summarizer = Summarization()

        total_step = len(self.m_dataloader[stage]) if stage == "valid" or (stage == "test" and step == 0) else 100

        for i_iter, batch in enumerate(self.m_dataloader[stage], start=1):
            batch = {k: v.cuda() for k, v in batch.items()}
            self.meta_eval(batch, stage, summarizer)
            if i_iter % self.progress.debug_log_freq == 0:
                msg = "{} {}:\t{}/{} {}\t{}".format(self.get_identifier(), stage.upper(),
                                                       i_iter, total_step, step,
                                                       LogUtils.get_stat_from_dict(summarizer.get_avg_dict()))
                self.logger.debug(msg)

            if i_iter >= total_step:
                break

        msg = "{} {}\t{}\t{}".format(self.get_identifier(), stage.upper(), step,
                                     LogUtils.get_stat_from_dict(summarizer.get_avg_dict()))
        self.logger.info(msg)
        self.report_2_mysql(step, stage, summarizer)

        return summarizer

    def add_addition_tokens(self, batch):
        batch["input_ids"] = torch.cat(
            (batch["input_ids"], self.added_token_ids.expand(batch["input_ids"].size(0), -1)), dim=1)
        batch["attention_mask"] = torch.cat(
            (batch["attention_mask"], torch.ones(batch["input_ids"].size(0), self.added_token_ids.size(-1)).long().cuda()), dim=1)
        batch["token_type_ids"] = torch.cat(
            (batch["token_type_ids"], torch.zeros(batch["input_ids"].size(0), self.added_token_ids.size(-1)).long().cuda()), dim=1)

    def train(self):
        stage = "train"
        num_train_iteration = self.config["num_iteration"][stage]

        batch_list = []
        summarizer = Summarization()

        no_improvement_cnt = 0
        max_acc = 0.0
        for batch in self.m_dataloader[stage]:
            self.prepare_train()
            batch = {k: v.cuda() for k, v in batch.items()}
            batch_list.append(batch)
            if len(batch_list) < self.meta_batch_size:
                continue
            else:
                self.meta_update(batch_list, stage, self.step / num_train_iteration, summarizer)
                self.post_update()
                self.step += 1
                batch_list = []

            if self.step % self.progress.debug_log_freq == 0:
                msg = "{} {}:\t{}/{}\t{}".format(self.get_identifier(), stage.upper(),
                                                       self.step, num_train_iteration,
                                                       LogUtils.get_stat_from_dict(summarizer.get_avg_dict()))
                self.logger.debug(msg)

            if self.step % self.progress.info_log_freq == 0:
                msg = "{} {}:\t{}/{}\t{}".format(self.get_identifier(), stage.upper(), self.step,
                                                 num_train_iteration,
                                                 LogUtils.get_stat_from_dict(summarizer.get_avg_dict()))
                self.logger.info(msg)
                self.report_2_mysql(self.step, stage, summarizer)

                summarizer = Summarization()

            if self.step % self.progress.meta_valid_freq == 0:
                eval_summarizer = self.eval(stage="valid", step=self.step)
                if eval_summarizer.get_item("query_acc") > max_acc + 0.0005:
                    max_acc = eval_summarizer.get_item("query_acc")
                    self.save_best_model(self.step, max_acc)
                    no_improvement_cnt = 0
                else:
                    no_improvement_cnt += 1
                    self.logger.info("NO IMPROVEMENT. {}/{} {:.2f}/{:.2f}".format(no_improvement_cnt, self.progress.max_not_impr_cnt,
                                                                          eval_summarizer.get_item("query_acc")*100, max_acc*100))

            if no_improvement_cnt >= self.progress.max_not_impr_cnt:
                self.logger.info("EARLY STOPPED")
                break

        self.load_best_model()
        self.eval(stage="test", step=0)

    def save_best_model(self, step, acc):
        checkpoint = {
            'model_state_dict': self.meta_model.state_dict(),
            "step": step,
            "best_acc": acc
        }

        if self.prompt_model is not None:
            checkpoint['prompt_state_dict'] = self.prompt_model.state_dict()
        PathUtils.save_ckp(checkpoint, dir_name=self.task_name, identifier=self.get_identifier(),
                           index_key=self.index_key)

    def load_best_model(self):
        checkpoint = PathUtils.load_ckp(dir_name=self.task_name, identifier=self.get_identifier(),
                                        index_key=self.index_key)
        self.logger.info("-----------------LOADED BEST MODEL---------------")
        self.logger.info("-----------------step: {}, best-acc: {:.2f}---------------".format(checkpoint['step'],
                                                                                             checkpoint['best_acc']*100))
        self.meta_model.load_state_dict(checkpoint["model_state_dict"])
        if self.prompt_model is not None:
            self.prompt_model.load_state_dict(checkpoint["prompt_state_dict"])

    def report_2_mysql(self, train_iter, stage, summarization):
        pass
