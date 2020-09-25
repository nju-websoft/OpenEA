import time
import os
import pickle
import random
import math

import numpy as np
import pandas as pd

import scipy
import tensorflow as tf
import scipy.sparse as sp

from scipy.sparse.linalg.eigen.arpack import eigsh
from openea.modules.utils.util import generate_out_folder

from openea.modules.bootstrapping.alignment_finder import find_alignment
from openea.modules.finding.alignment import greedy_alignment
from openea.modules.utils.util import task_divide, merge_dic
from openea.modules.finding.similarity import sim
from openea.modules.finding.evaluation import early_stop
import openea.modules.load.read as rd
from openea.models.basic_model import BasicModel


# ***************************adj & sparse**************************
def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN gnn and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))
    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])
    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))
    return sparse_to_tuple(t_k)


def func(triples):
    head = {}
    cnt = {}
    for tri in triples:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = {tri[0]}
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
    r2f = {}
    for r in cnt:
        r2f[r] = len(head[r]) / cnt[r]
    return r2f


def ifunc(triples):
    tail = {}
    cnt = {}
    for tri in triples:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            tail[tri[1]] = {tri[2]}
        else:
            cnt[tri[1]] += 1
            tail[tri[1]].add(tri[2])
    r2if = {}
    for r in cnt:
        r2if[r] = len(tail[r]) / cnt[r]
    return r2if


def get_weighted_adj(e, triples):
    r2f = func(triples)
    r2if = ifunc(triples)
    M = {}
    for tri in triples:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = max(r2if[tri[1]], 0.3)
        else:
            M[(tri[0], tri[2])] += max(r2if[tri[1]], 0.3)
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = max(r2f[tri[1]], 0.3)
        else:
            M[(tri[2], tri[0])] += max(r2f[tri[1]], 0.3)
    row = []
    col = []
    data = []
    for key in M:
        row.append(key[1])
        col.append(key[0])
        data.append(M[key])
    data = np.array(data, dtype='float32')
    return sp.coo_matrix((data, (row, col)), shape=(e, e))


def generate_rel_ht(triples):
    rel_ht_dict = dict()
    for h, r, t in triples:
        hts = rel_ht_dict.get(r, list())
        hts.append((h, t))
        rel_ht_dict[r] = hts
    return rel_ht_dict


def diag_adj(adj):
    d = np.array(adj.sum(1)).flatten()
    d_inv = 1. / d
    d_inv[np.isinf(d_inv)] = 0
    d_inv = sp.diags(d_inv)
    return sparse_to_tuple(d_inv.dot(adj))


def no_weighted_adj(total_ent_num, triple_list, is_two_adj=False):
    start = time.time()
    edge = dict()
    for item in triple_list:
        if item[0] not in edge.keys():
            edge[item[0]] = set()
        if item[2] not in edge.keys():
            edge[item[2]] = set()
        edge[item[0]].add(item[2])
        edge[item[2]].add(item[0])
    row = list()
    col = list()
    for i in range(total_ent_num):
        if i not in edge.keys():
            continue
        key = i
        value = edge[key]
        add_key_len = len(value)
        add_key = (key * np.ones(add_key_len)).tolist()
        row.extend(add_key)
        col.extend(list(value))
    data_len = len(row)
    data = np.ones(data_len)
    one_adj = sp.coo_matrix((data, (row, col)), shape=(total_ent_num, total_ent_num))
    one_adj = preprocess_adj(one_adj)
    print('generating one-adj costs time: {:.4f}s'.format(time.time() - start))
    if not is_two_adj:
        return one_adj, None
    expend_edge = dict()
    row = list()
    col = list()
    temp_len = 0
    for key, values in edge.items():
        if key not in expend_edge.keys():
            expend_edge[key] = set()
        for value in values:
            add_value = edge[value]
            for item in add_value:
                if item not in values and item != key:
                    expend_edge[key].add(item)
                    no_len = len(expend_edge[key])
                    if temp_len != no_len:
                        row.append(key)
                        col.append(item)
                    temp_len = no_len
    data = np.ones(len(row))
    two_adj = sp.coo_matrix((data, (row, col)), shape=(total_ent_num, total_ent_num))
    two_adj = preprocess_adj(two_adj)
    print('generating one- and two-adj costs time: {:.4f}s'.format(time.time() - start))
    return one_adj, two_adj


def relation_adj_list(kg1, kg2, adj_number, all_rel_num, all_ent_num, linked_ents, rel_id_mapping):
    rel_dict = rel_id_mapping
    adj_list = list()
    triple_list = kg1.triple_list + kg2.triple_list
    edge = dict()
    edge_length = np.zeros(all_rel_num)

    for item in triple_list:
        if rel_dict[item[1]] is not None and rel_dict[item[1]] != "":
            edge_id = rel_dict[item[1]]
        else:
            edge_id = item[1]
        if edge_id not in edge.keys():
            edge[edge_id] = list()
        edge[edge_id].append([item[0], item[2]])
        edge_length[edge_id] += 1
    sort_edge_length = np.argsort(-edge_length)

    # **********************************************************************
    adj_len = list()
    for i in range(adj_number):
        pos = np.array(edge[sort_edge_length[i]])
        row, col = np.transpose(pos)
        data = np.ones(shape=int(edge_length[sort_edge_length[i]]))

        adj_len.append(int(edge_length[sort_edge_length[i]]))

        adj = sp.coo_matrix((data, (row, col)), shape=(all_ent_num, all_ent_num))
        adj = sparse_to_tuple(adj)
        adj_list.append(adj)
    return adj_list


def remove_unlinked_triples(triples, linked_ents):
    print("before removing unlinked triples:", len(triples))
    new_triples = set()
    for h, r, t in triples:
        if h in linked_ents and t in linked_ents:
            new_triples.add((h, r, t))
    print("after removing unlinked triples:", len(new_triples))
    return list(new_triples)


def generate_2hop_triples(kg, linked_ents=None):
    triples = kg.triples
    if linked_ents is not None:
        triples = remove_unlinked_triples(triples, linked_ents)
    triple_df = np.array([[tr[0], tr[1], tr[2]] for tr in triples])
    triple_df = pd.DataFrame(triple_df, columns=['h', 'r', 't'])
    # print(triple_df)
    two_hop_triple_df = pd.merge(triple_df, triple_df, left_on='t', right_on='h')
    # print(two_hop_triple_df)
    two_step_quadruples = set()
    relation_patterns = dict()
    for index, row in two_hop_triple_df.iterrows():
        head = row["h_x"]
        tail = row["t_y"]
        r_x = row["r_x"]
        r_y = row['r_y']
        if tail not in kg.out_related_ents_dict.get(head, set()) and \
                head not in kg.in_related_ents_dict.get(tail, set()):
            relation_patterns[(r_x, r_y)] = relation_patterns.get((r_x, r_y), 0) + 1
            two_step_quadruples.add((head, r_x, r_y, tail))
    print("total 2-hop neighbors:", len(two_step_quadruples))
    print("total 2-hop relation patterns:", len(relation_patterns))
    relation_patterns = sorted(relation_patterns.items(), key=lambda x: x[1], reverse=True)
    p = 0.05
    num = int(p * len(relation_patterns))
    selected_patterns = set()
    # for i in range(20, num):
    for i in range(5, len(relation_patterns)):
        pattern = relation_patterns[i][0]
        selected_patterns.add(pattern)
    print("selected relation patterns:", len(selected_patterns))
    two_step_triples = set()
    for head, rx, ry, tail in two_step_quadruples:
        if (rx, ry) in selected_patterns:
            two_step_triples.add((head, 0, head))
            two_step_triples.add((head, rx + ry, tail))
    print("selected 2-hop neighbors:", len(two_step_triples))
    return two_step_triples


def transloss_add2hop(kg1, kg2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, total_e_num):
    linked_ents = set(sup_ent1 + sup_ent2 + ref_ent1 + ref_ent2)
    enhanced_triples1 = generate_2hop_triples(kg1, linked_ents=linked_ents)
    enhanced_triples2 = generate_2hop_triples(kg2, linked_ents=linked_ents)
    triples = enhanced_triples1 | enhanced_triples2
    edge = dict()
    for item in triples:
        if item[0] not in edge.keys():
            edge[item[0]] = set()
        if item[2] not in edge.keys():
            edge[item[2]] = set()
        edge[item[0]].add(item[2])
        edge[item[2]].add(item[0])
    row = list()
    col = list()
    for i in range(total_e_num):
        if i not in edge.keys():
            continue
        key = i
        value = edge[key]
        add_key_len = len(value)
        add_key = (key * np.ones(add_key_len)).tolist()
        row.extend(add_key)
        col.extend(list(value))
    data_len = len(row)
    data = np.ones(data_len)
    one_adj = sp.coo_matrix((data, (row, col)), shape=(total_e_num, total_e_num))
    one_adj = sparse_to_tuple(one_adj)
    return one_adj


def get_neighbor_dict(out_dict, in_dict):
    dic = dict()
    for key, value in out_dict.items():
        dic[key] = value
    for key, value in in_dict.items():
        values = dic.get(key, set())
        values |= value
        dic[key] = values
    return dic


def get_neighbor_counterparts(neighbors, alignment_dic):
    neighbor_counterparts = set()
    for n in neighbors:
        if n in alignment_dic:
            neighbor_counterparts.add(alignment_dic.get(n))
    return neighbor_counterparts


def check_new_alignment(aligned_pairs, context="check align"):
    if aligned_pairs is None or len(aligned_pairs) == 0:
        print("{}, empty aligned pairs".format(context))
        return
    num = 0
    for x, y in aligned_pairs:
        if x == y:
            num += 1
    print("{}, right align: {}/{}={:.3f}".format(context, num, len(aligned_pairs), num / len(aligned_pairs)))


def update_labeled_alignment_x(pre_labeled_alignment, curr_labeled_alignment, sim_mat):
    check_new_alignment(pre_labeled_alignment, context="before editing (<-)")
    labeled_alignment_dict = dict(pre_labeled_alignment)
    n1, n2 = 0, 0
    for i, j in curr_labeled_alignment:
        if labeled_alignment_dict.get(i, -1) == i and j != i:
            n2 += 1
        if i in labeled_alignment_dict.keys():
            pre_j = labeled_alignment_dict.get(i)
            if pre_j == j:
                continue
            pre_sim = sim_mat[i, pre_j]
            new_sim = sim_mat[i, j]
            if new_sim >= pre_sim:
                if pre_j == i and j != i:
                    n1 += 1
                labeled_alignment_dict[i] = j
        else:
            labeled_alignment_dict[i] = j
    print("update wrongly: ", n1, "greedy update wrongly: ", n2)
    pre_labeled_alignment = set(zip(labeled_alignment_dict.keys(), labeled_alignment_dict.values()))
    check_new_alignment(pre_labeled_alignment, context="after editing (<-)")
    return pre_labeled_alignment


def update_labeled_alignment_y(labeled_alignment, sim_mat):
    labeled_alignment_dict = dict()
    updated_alignment = set()
    for i, j in labeled_alignment:
        i_set = labeled_alignment_dict.get(j, set())
        i_set.add(i)
        labeled_alignment_dict[j] = i_set
    for j, i_set in labeled_alignment_dict.items():
        if len(i_set) == 1:
            for i in i_set:
                updated_alignment.add((i, j))
        else:
            max_i = -1
            max_sim = -10
            for i in i_set:
                if sim_mat[i, j] > max_sim:
                    max_sim = sim_mat[i, j]
                    max_i = i
            updated_alignment.add((max_i, j))
    check_new_alignment(updated_alignment, context="after editing (->)")
    return updated_alignment


def enhance_triples(kg1, kg2, ents1, ents2):
    assert len(ents1) == len(ents2)
    print("before enhanced:", len(kg1.triples), len(kg2.triples))
    enhanced_triples1, enhanced_triples2 = set(), set()
    links1 = dict(zip(ents1, ents2))
    links2 = dict(zip(ents2, ents1))
    for h1, r1, t1 in kg1.triples:
        h2 = links1.get(h1, None)
        t2 = links1.get(t1, None)
        if h2 is not None and t2 is not None and t2 not in kg2.out_related_ents_dict.get(h2, set()):
            enhanced_triples2.add((h2, r1, t2))
    for h2, r2, t2 in kg2.triples:
        h1 = links2.get(h2, None)
        t1 = links2.get(t2, None)
        if h1 is not None and t1 is not None and t1 not in kg1.out_related_ents_dict.get(h1, set()):
            enhanced_triples1.add((h1, r2, t1))
    print("after enhanced:", len(enhanced_triples1), len(enhanced_triples2))
    return enhanced_triples1, enhanced_triples2


def dropout(inputs, drop_rate, noise_shape, is_sparse):
    if not is_sparse:
        return tf.nn.dropout(inputs, drop_rate)
    return sparse_dropout(inputs, drop_rate, noise_shape)


def sparse_dropout(x, drop_rate, noise_shape):
    """
    Dropout for sparse tensors.
    """
    keep_prob = 1 - drop_rate
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def generate_neighbours(entity_embeds1, entity_list1, entity_embeds2, entity_list2, neighbors_num, threads_num=4):
    ent_frags = task_divide(np.array(entity_list1), threads_num)
    ent_frag_indexes = task_divide(np.array(range(len(entity_list1))), threads_num)
    dic = dict()
    for i in range(len(ent_frags)):
        res = find_neighbours(ent_frags[i], entity_embeds1[ent_frag_indexes[i], :], np.array(entity_list2),
                              entity_embeds2, neighbors_num)
        dic = merge_dic(dic, res)
    return dic


def find_neighbours(frags, sub_embed1, entity_list2, embed2, k):
    dic = dict()
    sim_mat = np.matmul(sub_embed1, embed2.T)
    for i in range(sim_mat.shape[0]):
        sort_index = np.argpartition(-sim_mat[i, :], k)
        neighbors_index = sort_index[0:k]
        neighbors = entity_list2[neighbors_index].tolist()
        dic[frags[i]] = neighbors
    return dic


class AKG:
    def __init__(self, triples, ori_triples=None):
        self.triples = set(triples)
        self.triple_list = list(self.triples)
        self.triples_num = len(self.triples)

        self.heads = set([triple[0] for triple in self.triple_list])
        self.props = set([triple[1] for triple in self.triple_list])
        self.tails = set([triple[2] for triple in self.triple_list])
        self.ents = self.heads | self.tails

        print("triples num", self.triples_num)

        print("head ent num", len(self.heads))
        print("total ent num", len(self.ents))

        self.prop_list = list(self.props)
        self.ent_list = list(self.ents)
        self.prop_list.sort()
        self.ent_list.sort()

        if ori_triples is None:
            self.ori_triples = None
        else:
            self.ori_triples = set(ori_triples)

        self._generate_related_ents()
        self._generate_triple_dict()
        self._generate_ht()
        self.__generate_weight()

    def _generate_related_ents(self):
        self.out_related_ents_dict = dict()
        self.in_related_ents_dict = dict()
        for h, r, t in self.triple_list:
            out_related_ents = self.out_related_ents_dict.get(h, set())
            out_related_ents.add(t)
            self.out_related_ents_dict[h] = out_related_ents

            in_related_ents = self.in_related_ents_dict.get(t, set())
            in_related_ents.add(h)
            self.in_related_ents_dict[t] = in_related_ents

    def _generate_triple_dict(self):
        self.rt_dict, self.hr_dict = dict(), dict()
        for h, r, t in self.triple_list:
            rt_set = self.rt_dict.get(h, set())
            rt_set.add((r, t))
            self.rt_dict[h] = rt_set
            hr_set = self.hr_dict.get(t, set())
            hr_set.add((h, r))
            self.hr_dict[t] = hr_set

    def _generate_ht(self):
        self.ht = set()
        for h, r, t in self.triples:
            self.ht.add((h, t))

    def __generate_weight(self):
        triple_num = dict()
        n = 0
        for h, r, t in self.triples:
            if t in self.heads:
                n = n + 1
                triple_num[h] = triple_num.get(h, 0) + 1
                triple_num[t] = triple_num.get(t, 0) + 1
        self.weighted_triples = list()
        self.additional_triples = list()
        ave = math.ceil(n / len(self.heads))
        print("ave outs:", ave)

        for h, r, t in self.triples:
            w = 1
            if t in self.heads and triple_num[h] <= ave:
                w = 2.0
                self.additional_triples.append((h, r, t))
            self.weighted_triples.append((h, r, t, w))
        print("additional triples:", len(self.additional_triples))


class GraphConvolution:
    def __init__(self, input_dim, output_dim, adj,
                 num_features_nonzero,
                 dropout_rate=0.0,
                 name='GCN',
                 is_sparse_inputs=False,
                 activation=tf.tanh,
                 use_bias=True):
        self.activation = activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adjs = [tf.SparseTensor(indices=am[0], values=am[1], dense_shape=am[2]) for am in adj]
        self.num_features_nonzero = num_features_nonzero
        self.dropout_rate = dropout_rate
        self.is_sparse_inputs = is_sparse_inputs
        self.use_bias = use_bias
        self.kernels = list()
        self.bias = list()
        self.name = name
        self.data_type = tf.float32
        self._get_variable()

    def _get_variable(self):
        self.batch_normalization = tf.keras.layers.BatchNormalization()
        for i in range(len(self.adjs)):
            self.kernels.append(tf.get_variable(self.name + '_kernel_' + str(i),
                                                shape=(self.input_dim, self.output_dim),
                                                initializer=tf.glorot_uniform_initializer(),
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                                dtype=self.data_type))
        if self.use_bias:
            self.bias = tf.get_variable(self.name + '_bias', shape=[self.output_dim, ],
                                        initializer=tf.zeros_initializer(),
                                        dtype=self.data_type)

    def call(self, inputs):
        inputs = self.batch_normalization(inputs)
        if self.dropout_rate > 0.0:
            inputs = dropout(inputs, self.dropout_rate, self.num_features_nonzero, self.is_sparse_inputs)
        hidden_vectors = list()
        for i in range(len(self.adjs)):
            pre_sup = tf.matmul(inputs, self.kernels[i], a_is_sparse=self.is_sparse_inputs)
            hidden_vector = tf.sparse_tensor_dense_matmul(tf.cast(self.adjs[i], tf.float32), pre_sup)
            hidden_vectors.append(hidden_vector)
        outputs = tf.add_n(hidden_vectors)
        # bias
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        # activation
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def update_adj(self, adj):
        print("gcn update adj...")
        self.adjs = [tf.SparseTensor(indices=am[0], values=am[1], dense_shape=am[2]) for am in adj]


class HighwayLayer:
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, name="highway"):
        self.input_shape = (input_dim, output_dim)
        self.name = name
        self.data_type = tf.float32
        self.dropout_rate = dropout_rate
        self._get_variable()

    def _get_variable(self):
        self.weight = tf.get_variable(self.name + 'kernel', shape=self.input_shape,
                                      initializer=tf.glorot_uniform_initializer(),
                                      regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                      dtype=self.data_type)
        self.activation = tf.tanh
        self.batch_normal = tf.keras.layers.BatchNormalization()

    def call(self, input1, input2):
        input1 = self.batch_normal(input1)
        input2 = self.batch_normal(input2)
        gate = tf.matmul(input1, self.weight)
        gate = self.activation(gate)
        if self.dropout_rate > 0:
            gate = tf.nn.dropout(gate, self.dropout_rate)
        gate = tf.keras.activations.relu(gate)
        output = tf.add(tf.multiply(input2, 1 - gate), tf.multiply(input1, gate))
        return self.activation(output)


class AliNetGraphAttentionLayer:
    def __init__(self, input_dim, output_dim, adj, nodes_num,
                 dropout_rate, is_sparse_input=False, use_bias=True,
                 activation=None, name="alinet"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adjs = [tf.SparseTensor(indices=adj[0][0], values=adj[0][1], dense_shape=adj[0][2])]
        self.dropout_rate = dropout_rate
        self.is_sparse_input = is_sparse_input
        self.nodes_num = nodes_num
        self.use_bias = use_bias
        self.activation = activation
        self.name = name
        self.data_type = tf.float32
        self._get_variable()

    def _get_variable(self):
        self.kernel = tf.get_variable(self.name + '_kernel', shape=(self.input_dim, self.output_dim),
                                      initializer=tf.glorot_uniform_initializer(),
                                      regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                      dtype=self.data_type)
        self.kernel1 = tf.get_variable(self.name + '_kernel_1', shape=(self.input_dim, self.input_dim),
                                       initializer=tf.glorot_uniform_initializer(),
                                       regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                       dtype=self.data_type)
        self.kernel2 = tf.get_variable(self.name + '_kernel_2', shape=(self.input_dim, self.input_dim),
                                       initializer=tf.glorot_uniform_initializer(),
                                       regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                       dtype=self.data_type)
        self.batch_normlization = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        inputs = self.batch_normlization(inputs)
        mapped_inputs = tf.matmul(inputs, self.kernel)
        attention_inputs1 = tf.matmul(inputs, self.kernel1)
        attention_inputs2 = tf.matmul(inputs, self.kernel2)
        con_sa_1 = tf.reduce_sum(tf.multiply(attention_inputs1, inputs), 1, keepdims=True)
        con_sa_2 = tf.reduce_sum(tf.multiply(attention_inputs2, inputs), 1, keepdims=True)
        con_sa_1 = tf.keras.activations.tanh(con_sa_1)
        con_sa_2 = tf.keras.activations.tanh(con_sa_2)
        if self.dropout_rate > 0.0:
            con_sa_1 = tf.nn.dropout(con_sa_1, self.dropout_rate)
            con_sa_2 = tf.nn.dropout(con_sa_2, self.dropout_rate)
        con_sa_1 = tf.cast(self.adjs[0], dtype=tf.float32) * con_sa_1
        con_sa_2 = tf.cast(self.adjs[0], dtype=tf.float32) * tf.transpose(con_sa_2, [1, 0])
        weights = tf.sparse_add(con_sa_1, con_sa_2)
        weights = tf.SparseTensor(indices=weights.indices,
                                  values=tf.nn.leaky_relu(weights.values),
                                  dense_shape=weights.dense_shape)
        attention_adj = tf.sparse_softmax(weights)
        attention_adj = tf.sparse_reshape(attention_adj, shape=[self.nodes_num, self.nodes_num])
        value = tf.sparse_tensor_dense_matmul(attention_adj, mapped_inputs)
        return self.activation(value)


class AliNet(BasicModel):

    def set_kgs(self, kgs):
        self.kgs = kgs
        self.kg1 = AKG(self.kgs.kg1.relation_triples_set)
        self.kg2 = AKG(self.kgs.kg2.relation_triples_set)

    def set_args(self, args):
        self.args = args
        self.out_folder = generate_out_folder(self.args.output, self.args.training_data, self.args.dataset_division,
                                              self.__class__.__name__)

    def init(self):
        self.ref_ent1 = self.kgs.test_entities1 + self.kgs.valid_entities1
        self.ref_ent2 = self.kgs.test_entities2 + self.kgs.valid_entities2
        self.sup_ent1 = self.kgs.train_entities1
        self.sup_ent2 = self.kgs.train_entities2
        self.linked_ents = set(self.kgs.train_entities1 +
                               self.kgs.train_entities2 +
                               self.kgs.valid_entities1 +
                               self.kgs.test_entities1 +
                               self.kgs.test_entities2 +
                               self.kgs.valid_entities2)
        enhanced_triples1, enhanced_triples2 = enhance_triples(self.kg1,
                                                               self.kg2,
                                                               self.sup_ent1,
                                                               self.sup_ent2)
        ori_triples = self.kg1.triple_list + self.kg2.triple_list
        triples = remove_unlinked_triples(ori_triples + list(enhanced_triples1) +
                                          list(enhanced_triples2), self.linked_ents)
        rel_ht_dict = generate_rel_ht(triples)
        saved_data_path = self.args.training_data + self.args.dataset_division + 'alinet_saved_data.pkl'
        if os.path.exists(saved_data_path):
            print('load saved adj data from', saved_data_path)
            adj = pickle.load(open(saved_data_path, 'rb'))
        else:
            one_adj, _ = no_weighted_adj(self.kgs.entities_num, triples, is_two_adj=False)
            adj = [one_adj]
            if self.is_two:
                two_hop_triples1 = generate_2hop_triples(self.kg1, linked_ents=self.linked_ents)
                two_hop_triples2 = generate_2hop_triples(self.kg2, linked_ents=self.linked_ents)
                triples = two_hop_triples1 | two_hop_triples2
                two_adj, _ = no_weighted_adj(self.kgs.entities_num, triples, is_two_adj=False)
                adj.append(two_adj)
            print('save adj data to', saved_data_path)
            pickle.dump(adj, open(saved_data_path, 'wb'))

        self.adj = adj
        self.rel_ht_dict = rel_ht_dict
        self.rel_win_size = self.args.batch_size // len(rel_ht_dict)
        if self.rel_win_size <= 1:
            self.rel_win_size = self.args.min_rel_win
        self.sim_th = self.args.sim_th

        sup_ent1 = np.array(self.sup_ent1).reshape((len(self.sup_ent1), 1))
        sup_ent2 = np.array(self.sup_ent2).reshape((len(self.sup_ent1), 1))
        weight = np.ones((len(self.kgs.train_entities1), 1), dtype=np.float)
        self.sup_links = np.hstack((sup_ent1, sup_ent2, weight))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self._get_variable()
        if self.args.rel_param > 0.0:
            self._generate_rel_graph()
        else:
            self._generate_graph()

        tf.global_variables_initializer().run(session=self.session)

    def __init__(self):
        super().__init__()
        self.adj = None
        self.one_hop_layers = None
        self.two_hop_layers = None
        self.layers_outputs = None

        self.new_edges1, self.new_edges2 = set(), set()
        self.new_links = set()
        self.pos_link_batch = None
        self.neg_link_batch = None
        self.sup_links_set = set()
        self.rel_ht_dict = None
        self.rel_win_size = None
        self.sim_th = None
        self.start_augment = None
        self.is_two = True
        self.new_sup_links_set = set()
        self.input_embeds, self.output_embeds_list = None, None
        self.sup_links = None
        self.model = None
        self.optimizer = None
        self.ref_ent1 = None
        self.ref_ent2 = None
        self.sup_ent1 = None
        self.sup_ent2 = None
        self.linked_ents = None
        self.session = None

    def _get_variable(self):
        self.init_embedding = tf.get_variable('init_embedding',
                                              shape=(self.kgs.entities_num, self.args.layer_dims[0]),
                                              initializer=tf.glorot_uniform_initializer(),
                                              dtype=tf.float32)

    def _define_model(self):
        print('Getting AliNet model...')
        layer_num = len(self.args.layer_dims) - 1
        output_embeds = self.init_embedding
        one_layers = list()
        two_layers = list()
        layers_outputs = list()
        for i in range(layer_num):
            gcn_layer = GraphConvolution(input_dim=self.args.layer_dims[i],
                                         output_dim=self.args.layer_dims[i + 1],
                                         adj=[self.adj[0]],
                                         num_features_nonzero=self.args.num_features_nonzero,
                                         dropout_rate=0.0,
                                         name='gcn_' + str(i))
            one_layers.append(gcn_layer)
            one_output_embeds = gcn_layer.call(output_embeds)

            if i < layer_num - 1:
                gat_layer = AliNetGraphAttentionLayer(input_dim=self.args.layer_dims[i],
                                                      output_dim=self.args.layer_dims[i + 1],
                                                      adj=[self.adj[1]],
                                                      nodes_num=self.kgs.entities_num,
                                                      dropout_rate=self.args.dropout,
                                                      is_sparse_input=False,
                                                      use_bias=True,
                                                      activation=tf.tanh,
                                                      name='alinet_' + str(i)
                                                      )
                two_layers.append(gat_layer)
                two_output_embeds = gat_layer.call(output_embeds)

                highway_layer = HighwayLayer(self.args.layer_dims[i + 1],
                                             self.args.layer_dims[i + 1],
                                             dropout_rate=self.args.dropout)
                output_embeds = highway_layer.call(two_output_embeds, one_output_embeds)
            else:
                output_embeds = one_output_embeds

            layers_outputs.append(output_embeds)

        self.one_hop_layers = one_layers
        self.two_hop_layers = two_layers
        self.output_embeds_list = layers_outputs

    def compute_loss(self, pos_links, neg_links, only_pos=False):
        index1 = pos_links[:, 0]
        index2 = pos_links[:, 1]
        neg_index1 = neg_links[:, 0]
        neg_index2 = neg_links[:, 1]

        embeds_list = list()
        for output_embeds in self.output_embeds_list + [self.init_embedding]:
            output_embeds = tf.nn.l2_normalize(output_embeds, 1)
            embeds_list.append(output_embeds)
        output_embeds = tf.concat(embeds_list, axis=1)
        output_embeds = tf.nn.l2_normalize(output_embeds, 1)

        embeds1 = tf.nn.embedding_lookup(output_embeds, tf.cast(index1, tf.int32))
        embeds2 = tf.nn.embedding_lookup(output_embeds, tf.cast(index2, tf.int32))
        pos_loss = tf.reduce_sum(tf.reduce_sum(tf.square(embeds1 - embeds2), 1))

        embeds1 = tf.nn.embedding_lookup(output_embeds, tf.cast(neg_index1, tf.int32))
        embeds2 = tf.nn.embedding_lookup(output_embeds, tf.cast(neg_index2, tf.int32))
        neg_distance = tf.reduce_sum(tf.square(embeds1 - embeds2), 1)
        neg_loss = tf.reduce_sum(tf.keras.activations.relu(self.args.neg_margin - neg_distance))

        return pos_loss + self.args.neg_margin_balance * neg_loss

    def compute_rel_loss(self, hs, ts):
        embeds_list = list()
        for output_embeds in self.output_embeds_list + [self.init_embedding]:
            output_embeds = tf.nn.l2_normalize(output_embeds, 1)
            embeds_list.append(output_embeds)
        output_embeds = tf.concat(embeds_list, axis=1)
        output_embeds = tf.nn.l2_normalize(output_embeds, 1)
        h_embeds = tf.nn.embedding_lookup(output_embeds, tf.cast(hs, tf.int32))
        t_embeds = tf.nn.embedding_lookup(output_embeds, tf.cast(ts, tf.int32))
        r_temp_embeds = tf.reshape(h_embeds - t_embeds, [-1, self.rel_win_size, output_embeds.shape[-1]])
        r_temp_embeds = tf.reduce_mean(r_temp_embeds, axis=1, keepdims=True)
        r_embeds = tf.tile(r_temp_embeds, [1, self.rel_win_size, 1])
        r_embeds = tf.reshape(r_embeds, [-1, output_embeds.shape[-1]])
        r_embeds = tf.nn.l2_normalize(r_embeds, 1)
        return tf.reduce_sum(tf.reduce_sum(tf.square(h_embeds - t_embeds - r_embeds), 1)) * self.args.rel_param

    def _generate_graph(self):
        self.pos_links = tf.placeholder(tf.int32, shape=[None, 3], name="pos")
        self.neg_links = tf.placeholder(tf.int32, shape=[None, 2], name='neg')
        self._define_model()
        self.loss = self.compute_loss(self.pos_links, self.neg_links)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(self.loss)

    def _generate_rel_graph(self):
        self.rel_pos_links = tf.placeholder(tf.int32, shape=[None, 3], name="pos")
        self.rel_neg_links = tf.placeholder(tf.int32, shape=[None, 2], name='neg')
        self.hs = tf.placeholder(tf.int32, shape=[None], name="hs")
        self.ts = tf.placeholder(tf.int32, shape=[None], name='ts')
        self._define_model()
        self.loss = self.compute_loss(self.rel_pos_links, self.rel_neg_links) + \
                    self.compute_rel_loss(self.hs, self.ts)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(self.loss)

    def augment(self):
        embeds1 = tf.nn.embedding_lookup(self.output_embeds_list[-1], self.ref_ent1)
        embeds2 = tf.nn.embedding_lookup(self.output_embeds_list[-1], self.ref_ent2)
        embeds1 = tf.nn.l2_normalize(embeds1, 1)
        embeds2 = tf.nn.l2_normalize(embeds2, 1)
        embeds1 = np.array(embeds1.eval(session=self.session))
        embeds2 = np.array(embeds2.eval(session=self.session))
        print("calculate sim mat...")
        sim_mat = sim(embeds1, embeds2, csls_k=self.args.csls)
        sim_mat = scipy.special.expit(sim_mat)
        th = self.sim_th
        print("sim th:", th)
        pair_index = find_alignment(sim_mat, th, 1)
        return pair_index, sim_mat

    def augment_neighborhood(self):
        pair_index, sim_mat = self.augment()
        if pair_index is None or len(pair_index) == 0:
            return
        self.new_links = update_labeled_alignment_x(self.new_links, pair_index, sim_mat)
        self.new_links = update_labeled_alignment_y(self.new_links, sim_mat)
        new_sup_ent1 = [self.ref_ent1[i] for i, _, in self.new_links]
        new_sup_ent2 = [self.ref_ent2[i] for _, i, in self.new_links]
        self.new_sup_links_set = set([(new_sup_ent1[i], new_sup_ent2[i]) for i in range(len(new_sup_ent1))])
        if new_sup_ent1 is None or len(new_sup_ent1) == 0:
            return
        enhanced_triples1, enhanced_triples2 = enhance_triples(self.kg1, self.kg2, self.sup_ent1 + new_sup_ent1,
                                                               self.sup_ent2 + new_sup_ent2)
        self.new_edges1 = enhanced_triples1
        self.new_edges2 = enhanced_triples2
        triples = self.kg1.triple_list + self.kg2.triple_list + list(self.new_edges1) + list(self.new_edges2)
        triples = remove_unlinked_triples(triples, self.linked_ents)
        one_adj, _ = no_weighted_adj(self.kgs.entities_num, triples, is_two_adj=False)
        adj = [one_adj]
        for layer in self.one_hop_layers:
            layer.update_adj(adj)

    def _eval_valid_embeddings(self):
        if len(self.kgs.valid_links) > 0:
            ent1 = self.kgs.valid_entities1
            ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        else:
            ent1 = self.kgs.test_entities1
            ent2 = self.kgs.test_entities2
        embeds_list1, embeds_list2 = list(), list()
        input_embeds = self.init_embedding
        output_embeds_list = self.output_embeds_list
        for output_embeds in [input_embeds] + output_embeds_list:
            output_embeds = tf.nn.l2_normalize(output_embeds, 1)
            embeds1 = tf.nn.embedding_lookup(output_embeds, ent1)
            embeds2 = tf.nn.embedding_lookup(output_embeds, ent2)
            embeds1 = tf.nn.l2_normalize(embeds1, 1)
            embeds2 = tf.nn.l2_normalize(embeds2, 1)
            embeds1 = np.array(embeds1.eval(session=self.session))
            embeds2 = np.array(embeds2.eval(session=self.session))
            embeds_list1.append(embeds1)
            embeds_list2.append(embeds2)
        embeds1 = np.concatenate(embeds_list1, axis=1)
        embeds2 = np.concatenate(embeds_list2, axis=1)
        mapping = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        return embeds1, embeds2, mapping

    def _eval_test_embeddings(self):
        ent1 = self.kgs.test_entities1
        ent2 = self.kgs.test_entities2
        embeds_list1, embeds_list2 = list(), list()
        input_embeds = self.init_embedding
        output_embeds_list = self.output_embeds_list
        for output_embeds in [input_embeds] + output_embeds_list:
            output_embeds = tf.nn.l2_normalize(output_embeds, 1)
            embeds1 = tf.nn.embedding_lookup(output_embeds, ent1)
            embeds2 = tf.nn.embedding_lookup(output_embeds, ent2)
            embeds1 = tf.nn.l2_normalize(embeds1, 1)
            embeds2 = tf.nn.l2_normalize(embeds2, 1)
            embeds1 = np.array(embeds1.eval(session=self.session))
            embeds2 = np.array(embeds2.eval(session=self.session))
            embeds_list1.append(embeds1)
            embeds_list2.append(embeds2)
        embeds1 = np.concatenate(embeds_list1, axis=1)
        embeds2 = np.concatenate(embeds_list2, axis=1)
        mapping = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        return embeds1, embeds2, mapping

    def save(self):
        embeds_list = list()
        input_embeds = self.init_embedding
        output_embeds_list = self.output_embeds_list
        for output_embeds in [input_embeds] + output_embeds_list:
            output_embeds = tf.nn.l2_normalize(output_embeds, 1)
            output_embeds = np.array(output_embeds.eval(session=self.session))
            embeds_list.append(output_embeds)
        ent_embeds = np.concatenate(embeds_list, axis=1)
        rd.save_embeddings(self.out_folder, self.kgs, ent_embeds, None, None, mapping_mat=None)

    # def save(self):
    #     ent_embeds = self.init_embedding
    #     rd.save_embeddings(self.out_folder, self.kgs, ent_embeds, None, None, mapping_mat=None)

    def generate_input_batch(self, batch_size, neighbors1=None, neighbors2=None):
        if batch_size > len(self.sup_ent1):
            batch_size = len(self.sup_ent1)
        index = np.random.choice(len(self.sup_ent1), batch_size)
        pos_links = self.sup_links[index,]
        neg_links = list()
        if neighbors1 is None:
            neg_ent1 = list()
            neg_ent2 = list()
            for i in range(self.args.neg_triple_num):
                neg_ent1.extend(random.sample(self.sup_ent1 + self.ref_ent1, batch_size))
                neg_ent2.extend(random.sample(self.sup_ent2 + self.ref_ent2, batch_size))
            neg_links.extend([(neg_ent1[i], neg_ent2[i]) for i in range(len(neg_ent1))])
        else:
            for i in range(batch_size):
                e1 = pos_links[i, 0]
                candidates = random.sample(neighbors1.get(e1), self.args.neg_triple_num)
                neg_links.extend([(e1, candidate) for candidate in candidates])
                e2 = pos_links[i, 1]
                candidates = random.sample(neighbors2.get(e2), self.args.neg_triple_num)
                neg_links.extend([(candidate, e2) for candidate in candidates])
        neg_links = set(neg_links) - self.sup_links_set
        neg_links = neg_links - self.new_sup_links_set
        neg_links = np.array(list(neg_links))
        return pos_links, neg_links

    def generate_rel_batch(self):
        hs, rs, ts = list(), list(), list()
        for r, hts in self.rel_ht_dict.items():
            hts_batch = [random.choice(hts) for _ in range(self.rel_win_size)]
            for h, t in hts_batch:
                hs.append(h)
                ts.append(t)
                rs.append(r)
        return hs, rs, ts

    def find_neighbors(self):
        if self.args.truncated_epsilon <= 0.0:
            return None, None
        start = time.time()
        output_embeds_list = self.output_embeds_list
        ents1 = self.sup_ent1 + self.ref_ent1
        ents2 = self.sup_ent2 + self.ref_ent2
        embeds1 = tf.nn.embedding_lookup(output_embeds_list[-1], ents1)
        embeds2 = tf.nn.embedding_lookup(output_embeds_list[-1], ents2)
        embeds1 = tf.nn.l2_normalize(embeds1, 1)
        embeds2 = tf.nn.l2_normalize(embeds2, 1)
        embeds1 = np.array(embeds1.eval(session=self.session))
        embeds2 = np.array(embeds2.eval(session=self.session))
        num = int((1 - self.args.truncated_epsilon) * len(ents1))
        print("neighbors num", num)
        neighbors1 = generate_neighbours(embeds1, ents1, embeds2, ents2, num,
                                         threads_num=self.args.test_threads_num)
        neighbors2 = generate_neighbours(embeds2, ents2, embeds1, ents1, num,
                                         threads_num=self.args.test_threads_num)
        print('finding neighbors for sampling costs time: {:.4f}s'.format(time.time() - start))
        return neighbors1, neighbors2

    def run(self):
        flag1 = 0
        flag2 = 0
        steps = len(self.sup_ent2) // self.args.batch_size
        neighbors1, neighbors2 = None, None
        if steps == 0:
            steps = 1
        for epoch in range(1, self.args.max_epoch + 1):
            start = time.time()
            epoch_loss = 0.0
            for step in range(steps):
                self.pos_link_batch, self.neg_link_batch = self.generate_input_batch(self.args.batch_size,
                                                                                     neighbors1=neighbors1,
                                                                                     neighbors2=neighbors2)
                fetches = {"loss": self.loss, "optimizer": self.optimizer}
                if self.args.rel_param > 0:
                    hs, _, ts = self.generate_rel_batch()
                    feed_dict = {self.rel_pos_links: self.pos_link_batch,
                                 self.rel_neg_links: self.neg_link_batch,
                                 self.hs: hs,
                                 self.ts: ts}
                    results = self.session.run(fetches=fetches, feed_dict=feed_dict)
                else:
                    feed_dict = {self.pos_links: self.pos_link_batch,
                                 self.neg_links: self.neg_link_batch}
                    results = self.session.run(fetches=fetches, feed_dict=feed_dict)
                batch_loss = results["loss"]
                epoch_loss += batch_loss
            print('epoch {}, loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))
            if epoch % self.args.eval_freq == 0 and epoch >= self.args.start_valid:
                flag = self.valid(self.args.stop_metric)
                flag1, flag2, is_stop = early_stop(flag1, flag2, flag)
                if is_stop:
                    print("\n == training stop == \n")
                    break
                neighbors1, neighbors2 = self.find_neighbors()
                if epoch >= self.args.start_augment * self.args.eval_freq:
                    if self.args.sim_th > 0.0:
                        self.augment_neighborhood()
