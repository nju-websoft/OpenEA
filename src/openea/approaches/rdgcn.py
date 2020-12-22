import math
import time
import numpy as np
import tensorflow as tf
import scipy
import pandas as pd
import string

import openea.modules.load.read as rd
from openea.modules.finding.evaluation import valid, test, early_stop
from openea.approaches.gcn_align import ones, glorot, zeros
from openea.models.basic_model import BasicModel
from sklearn import preprocessing
from openea.approaches.literal_encoder import LiteralEncoder


def rfunc(triple_list, ent_num, rel_num):
    head = dict()
    tail = dict()
    rel_count = dict()
    r_mat_ind = list()
    r_mat_val = list()
    head_r = np.zeros((ent_num, rel_num))
    tail_r = np.zeros((ent_num, rel_num))
    for triple in triple_list:
        head_r[triple[0]][triple[1]] = 1
        tail_r[triple[2]][triple[1]] = 1
        r_mat_ind.append([triple[0], triple[2]])
        r_mat_val.append(triple[1])
        if triple[1] not in rel_count:
            rel_count[triple[1]] = 1
            head[triple[1]] = set()
            tail[triple[1]] = set()
            head[triple[1]].add(triple[0])
            tail[triple[1]].add(triple[2])
        else:
            rel_count[triple[1]] += 1
            head[triple[1]].add(triple[0])
            tail[triple[1]].add(triple[2])
    r_mat = tf.SparseTensor(indices=r_mat_ind, values=r_mat_val, dense_shape=[ent_num, ent_num])

    return head, tail, head_r, tail_r, r_mat


def get_mat(triple_list, ent_num):
    degree = [1] * ent_num
    pos = dict()
    for triple in triple_list:
        if triple[0] != triple[1]:
            degree[triple[0]] += 1
            degree[triple[1]] += 1
        if triple[0] == triple[2]:
            continue
        if (triple[0], triple[2]) not in pos:
            pos[(triple[0], triple[2])] = 1
            pos[(triple[2], triple[0])] = 1

    for i in range(ent_num):
        pos[(i, i)] = 1
    return pos, degree


def get_sparse_tensor(triple_list, ent_num):
    pos, degree = get_mat(triple_list, ent_num)
    ind = []
    val = []
    for fir, sec in pos:
        ind.append((sec, fir))
        val.append(pos[(fir, sec)] / math.sqrt(degree[fir]) / math.sqrt(degree[sec]))
    pos = tf.SparseTensor(indices=ind, values=val, dense_shape=[ent_num, ent_num])

    return pos


def get_neg(ILL, output_layer, k):
    neg = []
    t = len(ILL)
    ILL_vec = np.array([output_layer[e1] for e1 in ILL])
    KG_vec = np.array(output_layer)
    sim = scipy.spatial.distance.cdist(ILL_vec, KG_vec, metric='cityblock')
    for i in range(t):
        rank = sim[i, :].argsort()
        neg.append(rank[0:k])

    neg = np.array(neg)
    neg = neg.reshape((t * k,))
    return neg


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def clear_attribute_triples(attribute_triples):
    print('\nbefore clear:', len(attribute_triples))
    # step 1
    attribute_triples_new = set()
    attr_num = {}
    for (e, a, _) in attribute_triples:
        ent_num = 1
        if a in attr_num:
            ent_num += attr_num[a]
        attr_num[a] = ent_num
    attr_set = set(attr_num.keys())
    attr_set_new = set()
    for a in attr_set:
        if attr_num[a] >= 10:
            attr_set_new.add(a)
    for (e, a, v) in attribute_triples:
        if a in attr_set_new:
            attribute_triples_new.add((e, a, v))
    attribute_triples = attribute_triples_new
    print('after step 1:', len(attribute_triples))

    # step 2
    attribute_triples_new = []
    literals_number, literals_string = [], []
    for (e, a, v) in attribute_triples:
        if '"^^' in v:
            v = v[:v.index('"^^')]
        if v.endswith('"@en'):
            v = v[:v.index('"@en')]
        if is_number(v):
            literals_number.append(v)
        else:
            literals_string.append(v)
        v = v.replace('.', '').replace('(', '').replace(')', '').replace(',', '').replace('"', '')
        v = v.replace('_', ' ').replace('-', ' ').replace('/', ' ')
        if 'http' in v:
            continue
        attribute_triples_new.append((e, a, v))
    attribute_triples = attribute_triples_new
    print('after step 2:', len(attribute_triples))
    return attribute_triples, literals_number, literals_string


def read_word2vec(file_path, vector_dimension):
    print('\n', file_path)
    word2vec = dict()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n').split(' ')
            if len(line) != vector_dimension + 1:
                continue
            v = np.array(list(map(float, line[1:])), dtype=np.float32)
            word2vec[line[0]] = v
    file.close()
    return word2vec


class Layer:
    def __init__(self, args, kg, embedding):
        self.dim = args.dim
        self.dropout = args.dropout
        self.act_func = tf.keras.activations.get("relu")
        self.gamma = args.gamma
        self.ILL = np.array(kg.train_links)
        self.k = args.neg_triple_num
        self.alpha = args.alpha
        self.beta = args.beta
        self.triple_list = kg.kg1.relation_triples_list + kg.kg2.relation_triples_list
        self.rel_num = kg.relations_num
        self.ent_num = kg.entities_num
        self.head = None
        self.tail = None
        self.head_r = None
        self.tail_r = None
        self.r_mat = None
        self.M = None
        self.adj = None
        self.pretrianed_embedding = embedding

    def add_diag_layer(self, inlayer, init=ones):
        inlayer = tf.nn.dropout(inlayer, 1 - self.dropout)
        w0 = init([1, self.dim])
        tosum = tf.sparse_tensor_dense_matmul(self.M, tf.multiply(inlayer, w0))
        if self.act_func is None:
            return tosum
        else:
            return self.act_func(tosum)

    def add_full_layer(self, inlayer, init=glorot):
        inlayer = tf.nn.dropout(inlayer, 1 - self.dropout)
        w0 = init([self.dim, self.dim])
        tosum = tf.sparse_tensor_dense_matmul(self.M, tf.matmul(inlayer, w0))
        if self.act_func is None:
            return tosum
        else:
            return self.act_func(tosum)

    def add_sparse_att_layer(self, inlayer, dual_layer):
        dual_transform = tf.reshape(tf.layers.conv1d(
            tf.expand_dims(dual_layer, 0), 1, 1), (-1, 1))
        logits = tf.reshape(tf.nn.embedding_lookup(
            dual_transform, self.r_mat.values), [-1])
        lrelu = tf.SparseTensor(indices=self.r_mat.indices,
                                values=tf.nn.leaky_relu(logits),
                                dense_shape=self.r_mat.dense_shape)
        coefs = tf.sparse_softmax(lrelu)
        vals = tf.sparse_tensor_dense_matmul(coefs, inlayer)
        if self.act_func is None:
            return vals
        else:
            return self.act_func(vals)

    def add_dual_att_layer(self, inlayer, inlayer2, adj):
        in_fts = tf.layers.conv1d(tf.expand_dims(inlayer2, 0), self.dim, 1)
        f_1 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
        f_2 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
        logits = f_1 + tf.transpose(f_2)
        adj_tensor = tf.constant(adj, dtype=tf.float32)
        bias_mat = -1e9 * (1.0 - (adj > 0))
        logits = tf.multiply(adj_tensor, logits)
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        vals = tf.matmul(coefs, inlayer)
        if self.act_func is None:
            return vals
        else:
            return self.act_func(vals)

    def add_self_att_layer(self, inlayer, adj):
        in_fts = tf.layers.conv1d(tf.expand_dims(
            inlayer, 0), self.dim, 1, use_bias=False)
        f_1 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
        f_2 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
        logits = f_1 + tf.transpose(f_2)
        adj_tensor = tf.constant(adj, dtype=tf.float32)
        logits = tf.multiply(adj_tensor, logits)
        bias_mat = -1e9 * (1.0 - (adj > 0))
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        vals = tf.matmul(coefs, inlayer)
        if self.act_func is None:
            return vals
        else:
            return self.act_func(vals)

    def highway(self, layer1, layer2):
        kernel_gate = glorot([self.dim, self.dim])
        bias_gate = zeros([self.dim])
        transform_gate = tf.matmul(layer1, kernel_gate) + bias_gate
        transform_gate = tf.nn.sigmoid(transform_gate)
        carry_gate = 1.0 - transform_gate
        return transform_gate * layer2 + carry_gate * layer1

    def compute_r(self, inlayer):
        head_l = tf.transpose(tf.constant(self.head_r, dtype=tf.float32))
        tail_l = tf.transpose(tf.constant(self.tail_r, dtype=tf.float32))
        L = tf.matmul(head_l, inlayer) / \
            tf.expand_dims(tf.reduce_sum(head_l, axis=-1), -1)
        R = tf.matmul(tail_l, inlayer) / \
            tf.expand_dims(tf.reduce_sum(tail_l, axis=-1), -1)
        r_embeddings = tf.concat([L, R], axis=-1)
        return r_embeddings

    def get_dual_input(self, inlayer):
        dual_X = self.compute_r(inlayer)
        count_r = len(self.head)
        dual_A = np.zeros((count_r, count_r))
        for i in range(count_r):
            for j in range(count_r):
                a_h = len(self.head[i] & self.head[j]) / len(self.head[i] | self.head[j])
                a_t = len(self.tail[i] & self.tail[j]) / len(self.tail[i] | self.tail[j])
                dual_A[i][j] = a_h + a_t
        return dual_X, dual_A

    # ******************************get_input_layer is used to initialize embeddings**********
    def get_input_layer(self):
        ent_embeddings = glorot((self.ent_num, self.dim), "input")
        return ent_embeddings
        # input_embeddings = tf.random_uniform([self.ent_num, self.dim], minval=-1, maxval=1)
        # ent_embeddings = tf.Variable(input_embeddings)
        # return tf.nn.l2_normalize(ent_embeddings, 1)

    def get_pretrained_input(self, embedding):
        embedding = tf.cast(embedding, dtype=tf.float32)
        ent_embeddings = tf.Variable(embedding)
        return ent_embeddings
        # return tf.nn.l2_normalize(ent_embeddings, 1)

    def get_loss(self, outlayer):
        left = self.ILL[:, 0]
        right = self.ILL[:, 1]
        t = len(self.ILL)
        left_x = tf.nn.embedding_lookup(outlayer, left)
        right_x = tf.nn.embedding_lookup(outlayer, right)
        A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
        neg_left = tf.placeholder(tf.int32, [t * self.k], "neg_left")
        neg_right = tf.placeholder(tf.int32, [t * self.k], "neg_right")
        neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
        neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
        B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
        C = - tf.reshape(B, [t, self.k])
        D = A + self.gamma
        L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
        neg_left = tf.placeholder(tf.int32, [t * self.k], "neg2_left")
        neg_right = tf.placeholder(tf.int32, [t * self.k], "neg2_right")
        neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
        neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
        B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
        C = - tf.reshape(B, [t, self.k])
        L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
        return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * self.k * t)

    def build(self):
        tf.reset_default_graph()
        # primal_X_0 = self.get_input_layer()
        primal_X_0 = self.get_pretrained_input(self.pretrianed_embedding)
        self.M = get_sparse_tensor(self.triple_list, self.ent_num)
        self.head, self.tail, self.head_r, self.tail_r, self.r_mat = rfunc(self.triple_list, self.ent_num, self.rel_num)
        dual_X_1, dual_A_1 = self.get_dual_input(primal_X_0)
        dual_H_1 = self.add_self_att_layer(dual_X_1, dual_A_1)
        primal_H_1 = self.add_sparse_att_layer(primal_X_0, dual_H_1)
        primal_X_1 = primal_X_0 + self.alpha * primal_H_1

        dual_X_2, dual_A_2 = self.get_dual_input(primal_X_1)
        dual_H_2 = self.add_dual_att_layer(dual_H_1, dual_X_2, dual_A_2)
        primal_H_2 = self.add_sparse_att_layer(primal_X_1, dual_H_2)
        primal_X_2 = primal_X_0 + self.beta * primal_H_2

        gcn_layer_1 = self.add_diag_layer(primal_X_2)
        gcn_layer_1 = self.highway(primal_X_2, gcn_layer_1)
        gcn_layer_2 = self.add_diag_layer(gcn_layer_1, )
        output_layer = self.highway(gcn_layer_1, gcn_layer_2)
        loss = self.get_loss(output_layer)
        return output_layer, loss


class RDGCN(BasicModel):
    def __init__(self):
        super().__init__()
        self.loss = 0
        self.output = None
        self.optimizer = None
        self.model_init = None
        self.sess = None
        self.feeddict = None
        self.gcn_model = None
        self.local_name_vectors = None
        self.entity_local_name_dict = None
        self.entities = None
        self.word_embed = '../../datasets/wiki-news-300d-1M.vec'

    def init(self):
        self.entities = self.kgs.kg1.entities_set | self.kgs.kg2.entities_set
        _, _, self.local_name_vectors = self._get_desc_input()
        self.gcn_model = Layer(self.args, self.kgs, self.local_name_vectors)
        self.output, self.loss = self.gcn_model.build()
        self.optimizer = tf.train.AdamOptimizer(self.args.learning_rate).minimize(self.loss)
        self.model_init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.model_init)

    def _get_local_name_by_name_triple(self, name_attribute_list=None):
        if name_attribute_list is None:
            if 'D_Y' in self.args.training_data:
                name_attribute_list = {'skos:prefLabel', 'http://dbpedia.org/ontology/birthName'}
            elif 'D_W' in self.args.training_data:
                name_attribute_list = {'http://www.wikidata.org/entity/P373', 'http://www.wikidata.org/entity/P1476'}
            else:
                name_attribute_list = {}

        local_triples = self.kgs.kg1.local_attribute_triples_set | self.kgs.kg2.local_attribute_triples_set
        triples = list()
        for h, a, v in local_triples:
            v = v.strip('"')
            if v.endswith('"@eng'):
                v = v.rstrip('"@eng')
            triples.append((h, a, v))
        id_ent_dict = {}
        for e, e_id in self.kgs.kg1.entities_id_dict.items():
            id_ent_dict[e_id] = e
        for e, e_id in self.kgs.kg2.entities_id_dict.items():
            id_ent_dict[e_id] = e

        name_ids = set()
        for a, a_id in self.kgs.kg1.attributes_id_dict.items():
            if a in name_attribute_list:
                name_ids.add(a_id)
        for a, a_id in self.kgs.kg2.attributes_id_dict.items():
            if a in name_attribute_list:
                name_ids.add(a_id)

        for a, a_id in self.kgs.kg1.attributes_id_dict.items():
            if a_id in name_ids:
                print(a)
        for a, a_id in self.kgs.kg2.attributes_id_dict.items():
            if a_id in name_ids:
                print(a)
        local_name_dict = {}
        ents = self.kgs.kg1.entities_set | self.kgs.kg2.entities_set
        for (e, a, v) in triples:
            if a in name_ids:
                local_name_dict[e] = v
        for e in ents:
            if e not in local_name_dict:
                local_name_dict[e] = id_ent_dict[e].split('/')[-1].replace('_', ' ')
        name_triples = list()
        for e, n in local_name_dict.items():
            name_triples.append((e, -1, n))
        return name_triples

    def _get_desc_input(self):
        # desc graph settings
        start = time.time()
        model = self

        name_triples = self._get_local_name_by_name_triple()
        names = pd.DataFrame(name_triples)
        names.iloc[:, 2] = names.iloc[:, 2].str.replace(r'[{}]+'.format(string.punctuation), '').str.split(' ')
        # load word embedding
        with open(self.word_embed, 'r') as f:
            w = f.readlines()
            w = pd.Series(w[1:])

        we = w.str.split(' ')
        word = we.apply(lambda x: x[0])
        w_em = we.apply(lambda x: x[1:])
        print('concat word embeddings')
        word_em = np.stack(w_em.values, axis=0).astype(np.float)
        word_em = np.append(word_em, np.zeros([1, 300]), axis=0)
        print('convert words to ids')
        w_in_desc = []
        for l in names.iloc[:, 2].values:
            w_in_desc += l
        w_in_desc = pd.Series(list(set(w_in_desc)))
        un_logged_words = w_in_desc[~w_in_desc.isin(word)]
        un_logged_id = len(word)

        all_word = pd.concat(
            [pd.Series(word.index, word.values),
             pd.Series([un_logged_id, ] * len(un_logged_words), index=un_logged_words)])

        def lookup_and_padding(x):
            default_length = 4
            ids = list(all_word.loc[x].values) + [all_word.iloc[-1], ] * default_length
            return ids[:default_length]

        print('look up desc embeddings')
        names.iloc[:, 2] = names.iloc[:, 2].apply(lookup_and_padding)

        # entity-desc-embedding dataframe
        e_desc_input = pd.DataFrame(np.repeat([[un_logged_id, ] * 4], model.kgs.entities_num, axis=0),
                                    range(model.kgs.entities_num))

        e_desc_input.iloc[names.iloc[:, 0].values] = np.stack(names.iloc[:, 2].values)

        print('generating desc input costs time: {:.4f}s'.format(time.time() - start))
        name_embeds = word_em[e_desc_input.values]
        name_embeds = np.sum(name_embeds, axis=1)

        return word_em, e_desc_input, name_embeds

    def training(self):
        neg_num = self.args.neg_triple_num
        train_num = len(self.kgs.train_links)
        train_links = np.array(self.kgs.train_links)
        pos = np.ones((train_num, neg_num)) * (train_links[:, 0].reshape((train_num, 1)))
        neg_left = pos.reshape((train_num * neg_num,))
        pos = np.ones((train_num, neg_num)) * (train_links[:, 1].reshape((train_num, 1)))
        neg2_right = pos.reshape((train_num * neg_num,))
        # output = self.sess.run(self.output)
        # neg2_left = get_neg(train_links[:, 1], output, self.args.neg_triple_num)
        # neg_right = get_neg(train_links[:, 0], output, self.args.neg_triple_num)
        # self.feeddict = {"neg_left:0": neg_left,
        #                  "neg_right:0": neg_right,
        #                  "neg2_left:0": neg2_left,
        #                  "neg2_right:0": neg2_right}

        for i in range(1, self.args.max_epoch + 1):
            start = time.time()
            if i % 10 == 1:
                output = self.sess.run(self.output)
                neg2_left = get_neg(train_links[:, 1], output, self.args.neg_triple_num)
                neg_right = get_neg(train_links[:, 0], output, self.args.neg_triple_num)
                self.feeddict = {"neg_left:0": neg_left,
                                 "neg_right:0": neg_right,
                                 "neg2_left:0": neg2_left,
                                 "neg2_right:0": neg2_right}

            _, batch_loss = self.sess.run([self.optimizer, self.loss], feed_dict=self.feeddict)
            print('epoch {}, avg. relation triple loss: {:.4f}, cost time: {:.4f}s'.format(i, batch_loss,
                                                                                           time.time() - start))

            # ********************no early stop********************************************
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                flag = self.valid_(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break

    def test(self, save=True):
        embedding = self.sess.run(self.output)
        embeds1 = np.array([embedding[e] for e in self.kgs.test_entities1])
        embeds2 = np.array([embedding[e] for e in self.kgs.test_entities2])
        rest_12, _, _ = test(embeds1, embeds2, None, self.args.top_k, self.args.test_threads_num,
                             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        test(embeds1, embeds2, None, self.args.top_k, self.args.test_threads_num,
             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=self.args.csls, accurate=True)
        if save:
            ent_ids_rest_12 = [(self.kgs.test_entities1[i], self.kgs.test_entities2[j]) for i, j in rest_12]
            rd.save_results(self.out_folder, ent_ids_rest_12)

    def save(self):
        embedding = self.sess.run(self.output)
        rd.save_embeddings(self.out_folder, self.kgs, embedding, None, None, mapping_mat=None)

    def valid_(self, stop_metric):
        embedding = self.sess.run(self.output)
        embeds1 = np.array([embedding[e] for e in self.kgs.valid_entities1])
        embeds2 = np.array([embedding[e] for e in self.kgs.valid_entities2 + self.kgs.test_entities2])
        hits1_12, mrr_12 = valid(embeds1, embeds2, None, self.args.top_k, self.args.test_threads_num,
                                 metric=self.args.eval_metric)
        if stop_metric == 'hits1':
            return hits1_12
        return mrr_12

    def run(self):
        t = time.time()
        self.training()
        print("training finish")
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
