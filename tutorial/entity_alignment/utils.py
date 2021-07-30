import itertools

import numpy as np

from openea.modules.load.kg import KG
from openea.modules.load.kgs import KGs


def find_alignment(sim_mat, k):
    nearest_k_neighbors = search_nearest_k(sim_mat, k)
    return nearest_k_neighbors


def search_nearest_k(sim_mat, k):
    assert k > 0
    neighbors = list()
    num = sim_mat.shape[0]
    for i in range(num):
        rank = np.argpartition(-sim_mat[i, :], k)
        pairs = [j for j in itertools.product([i], rank[0:k])]
        neighbors.append(pairs)
    assert len(neighbors) == num
    return neighbors


def read_items(file_path):
    print("read items:", file_path)
    items = list()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        items.append(line.strip('\n').strip())
    return items


class MyKGs(KGs):

    def __init__(self, kg1: KG, kg2: KG, train_links, test_links,
                 train_unlinked_entities1, valid_unlinked_entities1, test_unlinked_entities1,
                 train_unlinked_entities2, valid_unlinked_entities2, test_unlinked_entities2,
                 valid_links=None, mode='mapping', ordered=True):
        super().__init__(kg1, kg2, train_links, test_links, valid_links=valid_links, mode=mode, ordered=ordered)

        linked_label = 0.
        unlinked_label = 1.

        self.train_unlinked_entities1 = [(self.kg1.entities_id_dict.get(ent), unlinked_label) for ent in train_unlinked_entities1]
        self.valid_unlinked_entities1 = [(self.kg1.entities_id_dict.get(ent), unlinked_label) for ent in valid_unlinked_entities1]
        self.test_unlinked_entities1 = [(self.kg1.entities_id_dict.get(ent), unlinked_label) for ent in test_unlinked_entities1]

        self.train_linked_entities1 = [(ent, linked_label) for ent in self.train_entities1]
        self.valid_linked_entities1 = [(ent, linked_label) for ent in self.valid_entities1]
        self.test_linked_entities1 = [(ent, linked_label) for ent in self.test_entities1]

        self.train_null_data1 = self.train_unlinked_entities1 + self.train_linked_entities1
        self.valid_null_data1 = self.valid_unlinked_entities1 + self.valid_linked_entities1
        self.test_null_data1 = self.test_unlinked_entities1 + self.test_linked_entities1
    
        print("training/valid/test/total null ea data in KG1:",
              len(self.train_null_data1), len(self.valid_null_data1), len(self.test_null_data1),
              len(self.train_null_data1) + len(self.valid_null_data1) + len(self.test_null_data1),
              self.kg1.entities_num)

        self.train_unlinked_entities2 = [(self.kg2.entities_id_dict.get(ent), unlinked_label) for ent in train_unlinked_entities2]
        self.valid_unlinked_entities2 = [(self.kg2.entities_id_dict.get(ent), unlinked_label) for ent in valid_unlinked_entities2]
        self.test_unlinked_entities2 = [(self.kg2.entities_id_dict.get(ent), unlinked_label) for ent in test_unlinked_entities2]

        self.train_linked_entities2 = [(ent, linked_label) for ent in self.train_entities2]
        self.valid_linked_entities2 = [(ent, linked_label) for ent in self.valid_entities2]
        self.test_linked_entities2 = [(ent, linked_label) for ent in self.test_entities2]

        self.train_null_data2 = self.train_unlinked_entities2 + self.train_linked_entities2
        self.valid_null_data2 = self.valid_unlinked_entities2 + self.valid_linked_entities2
        self.test_null_data2 = self.test_unlinked_entities2 + self.test_linked_entities2

        print("training/valid/test/total null ea data in KG2:",
              len(self.train_null_data2), len(self.valid_null_data2), len(self.test_null_data2),
              len(self.train_null_data2) + len(self.valid_null_data2) + len(self.test_null_data2),
              self.kg2.entities_num)


def viz_sim_list(sim_list, interval=0.1):
    num = int(1 / interval)
    nums = []
    for i in range(num):
        nums.append(round(i * interval, 1))
    nums.append(1.0)
    sim_nums = [0] * num
    for sim in sim_list:
        idx = int(sim * 10)
        sim_nums[idx] += 1
    res = []
    n_inst = len(sim_list)
    for sim in sim_nums:
        res.append(round(sim / n_inst, 3))
    print(res)
