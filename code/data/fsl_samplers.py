
import numpy as np
import torch


class NWayKShotSampler:

    def __init__(self, labels, target_labels=None, total_steps=20, n_way=5, k_shot=1):
        """
        n way k shot sampler for few-shot learning problem
        :param labels: list, index means sample index, value means label
        :param total_steps: train step number
        :param n_way: number of class per batch
        :param k_shot: number of samples per class in one batch
        """

        self.total_steps = total_steps
        self.n_way = n_way
        self.k_shot = k_shot
        self.label_2_instance_ids = []

        for i in range(max(labels) + 1):
            ids = np.argwhere(np.array(labels) == i).reshape(-1)
            ids = torch.from_numpy(ids)
            self.label_2_instance_ids.append(ids)

        self.labels_num = len(self.label_2_instance_ids)
        self.labels = labels
        self.class_ids_shuffle = torch.randperm(self.labels_num)

        if target_labels is None:
            self.target_labels = list(range(self.labels_num))
        else:
            self.target_labels = target_labels

    def __len__(self):
        return self.total_steps

    def __iter__(self):

        for i_batch in range(self.total_steps):
            batch = []

            # randomly choose n_way classes
            class_ids = np.random.choice(self.target_labels, self.n_way, replace=False)
            class_ids = sorted(class_ids)
            for class_id in class_ids:
                instances_ids = self.label_2_instance_ids[class_id]
                # randomly pick k_shot samples per each class
                instances_ids_selected = torch.randperm(len(instances_ids))[0:self.k_shot]
                batch.append(instances_ids[instances_ids_selected])
            batch = torch.stack(batch).reshape(-1)
            yield batch.tolist()

