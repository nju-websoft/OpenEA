import itertools
import random
import time

import numpy as np
import tensorflow as tf
from sklearn import preprocessing

import openea.modules.finding.evaluation as evaluation
import openea.modules.load.read as read
from openea.modules.load.kg import KG
from openea.modules.load.kgs import KGs
from openea.modules.utils.util import generate_out_folder
from openea.modules.utils.util import load_session, merge_dic
from openea.modules.base.initializers import xavier_init
from openea.modules.base.optimizers import generate_optimizer


def get_kg_popular_attributes(kg: KG, threshold):
    count_dic = dict()
    for _, attr, _ in kg.attribute_triples_list:
        count_dic[attr] = count_dic.get(attr, 0) + 1
    print("total attributes:", len(count_dic))
    used_attributes_num = int(len(count_dic) * threshold)
    sorted_attributes = sorted(count_dic, key=count_dic.get, reverse=True)
    selected_attributes = set(sorted_attributes[0: used_attributes_num])
    print("selected attributes", len(selected_attributes))
    return selected_attributes


def get_kgs_popular_attributes(kgs: KGs, threshold):
    kg1_selected_attributes = get_kg_popular_attributes(kgs.kg1, threshold)
    kg2_selected_attributes = get_kg_popular_attributes(kgs.kg2, threshold)
    selected_attributes = kg1_selected_attributes | kg2_selected_attributes
    print("total selected attributes", len(selected_attributes))
    return kg1_selected_attributes, kg2_selected_attributes, selected_attributes


def generate_training_data(kgs: KGs, threshold=1.0):
    kg1_selected_attributes, kg2_selected_attributes, selected_attributes = get_kgs_popular_attributes(kgs, threshold)
    entity_attributes_dict = merge_dic(kgs.kg1.entity_attributes_dict, kgs.kg2.entity_attributes_dict)
    print("entity attribute dict", len(entity_attributes_dict))
    training_data_list = list()
    training_links_dict12 = dict(zip(kgs.train_entities1, kgs.train_entities2))
    training_links_dict21 = dict(zip(kgs.train_entities2, kgs.train_entities1))
    training_links_dict = merge_dic(training_links_dict12, training_links_dict21)
    for ent, attributes in entity_attributes_dict.items():
        if ent in training_links_dict.keys():
            attributes = attributes | entity_attributes_dict.get(training_links_dict.get(ent), set())
        attributes = attributes & selected_attributes
        for attr, context_attr in itertools.combinations(attributes, 2):
            if attr != context_attr:
                training_data_list.append((attr, context_attr))
    print("training data of attribute correlations", len(training_data_list))
    return training_data_list


def get_ent_embeds_from_attributes(kgs: KGs, attr_embeds, selected_attributes):
    print("get entity embeddings from attributes")
    start = time.time()
    ent_mat = None
    entity_attributes_dict = merge_dic(kgs.kg1.entity_attributes_dict, kgs.kg2.entity_attributes_dict)
    zero_vec = np.zeros([1, attr_embeds.shape[1]], dtype=np.float32)
    for i in range(kgs.entities_num):
        attr_vec = zero_vec
        attributes = entity_attributes_dict.get(i, set())
        attributes = attributes & selected_attributes
        if len(attributes) > 0:
            attr_vecs = attr_embeds[list(attributes), ]
            attr_vec = np.mean(attr_vecs, axis=0, keepdims=True)
        if ent_mat is None:
            ent_mat = attr_vec
        else:
            ent_mat = np.row_stack([ent_mat, attr_vec])
    print('cost time: {:.4f}s'.format(time.time() - start))
    return preprocessing.normalize(ent_mat)


class Attr2Vec:
    def set_kgs(self, kgs):
        self.kgs = kgs
        _, _, self.selected_attributes = get_kgs_popular_attributes(kgs, self.args.top_attr_threshold)
        self.num_sampled_negs = len(self.selected_attributes) // 5

    def set_args(self, args):
        self.args = args
        self.out_folder = generate_out_folder(self.args.output, self.args.training_data, self.args.dataset_division, self.__class__.__name__)

    def init(self):
        self._define_variables()
        self._define_embed_graph()
        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)

    def __init__(self):
        self.num_sampled_negs = -1
        self.kgs = None
        self.args = None
        self.out_folder = None
        self.flag1, self.flag2 = -1, -1
        self.early_stop = False
        self.session = None
        self.selected_attributes = None
        self.opt = 'Adagrad'

    def _define_variables(self):
        with tf.variable_scope('attribute' + 'embeddings'):
            self.embeds = xavier_init([self.kgs.attributes_num, self.args.dim], 'attr_embed', True)
        with tf.variable_scope('nce' + 'embeddings'):
            self.nce_weights = xavier_init([self.kgs.attributes_num, self.args.dim], 'nce_weights', True)
            self.nce_biases = tf.Variable(tf.zeros([self.kgs.attributes_num]))

    def _define_embed_graph(self):
        with tf.name_scope('attribute_placeholder'):
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.args.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.args.batch_size])
        with tf.name_scope('attribute_lookup'):
            self.train_inputs_embed = tf.nn.embedding_lookup(self.embeds, self.train_inputs)
        with tf.name_scope('attribute_nce_loss'):
            self.train_labels = tf.reshape(self.train_labels, [-1, 1])
            self.loss = tf.reduce_mean(tf.nn.nce_loss(self.nce_weights, self.nce_biases,
                                                      self.train_labels, self.train_inputs_embed,
                                                      self.num_sampled_negs, self.kgs.attributes_num))
            self.optimizer = generate_optimizer(self.loss, self.args.learning_rate, opt=self.opt)

    def eval_attribute_embeddings(self):
        return self.embeds.eval(session=self.session)

    def eval_kg1_ent_embeddings(self):
        mat = get_ent_embeds_from_attributes(self.kgs, self.eval_attribute_embeddings(), self.selected_attributes)
        embeds1 = mat[self.kgs.kg1.entities_list, ]
        return embeds1

    def eval_kg2_ent_embeddings(self):
        mat = get_ent_embeds_from_attributes(self.kgs, self.eval_attribute_embeddings(), self.selected_attributes)
        embeds2 = mat[self.kgs.kg2.entities_list, ]
        return embeds2

    def eval_sim_mat(self):
        mat = get_ent_embeds_from_attributes(self.kgs, self.eval_attribute_embeddings(), self.selected_attributes)
        embeds1 = mat[self.kgs.valid_entities1 + self.kgs.test_entities1, ]
        embeds2 = mat[self.kgs.valid_entities2 + self.kgs.test_entities2, ]
        return np.matmul(embeds1, embeds2.T)

    def launch_training_1epo(self, epoch, steps, training_data_list):
        start = time.time()
        epoch_loss = 0
        trained_pos_triples = 0
        for i in range(steps):
            training_batch = random.sample(training_data_list, self.args.batch_size)
            batch = np.ndarray(shape=(self.args.batch_size,), dtype=np.int32)
            labels = np.ndarray(shape=(self.args.batch_size, 1), dtype=np.int32)
            for index, x in enumerate(training_batch):
                batch[index] = x[0]
                labels[index, 0] = x[1]
            batch_loss, _ = self.session.run(fetches=[self.loss, self.optimizer],
                                             feed_dict={self.train_inputs: batch,
                                                        self.train_labels: labels})
            trained_pos_triples += len(training_batch)
            epoch_loss += batch_loss
        print('epoch {}, attribute loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def valid(self):
        mat = get_ent_embeds_from_attributes(self.kgs, self.eval_attribute_embeddings(), self.selected_attributes)
        embeds1 = mat[self.kgs.valid_entities1, ]
        embeds2 = mat[self.kgs.valid_entities2, ]
        hits1_12, mrr_12 = evaluation.valid(embeds1, embeds2, None, self.args.top_k,
                                            self.args.test_threads_num, metric=self.args.eval_metric)
        if self.args.stop_metric == 'hits1':
            return hits1_12
        return mrr_12

    def test(self, save=True):
        mat = get_ent_embeds_from_attributes(self.kgs, self.eval_attribute_embeddings(), self.selected_attributes)
        embeds1 = mat[self.kgs.test_entities1,]
        embeds2 = mat[self.kgs.test_entities2,]
        rest_12, _, _, rest_21, _, _ = evaluation.test(embeds1, embeds2, None, self.args.top_k,
                                                       self.args.test_threads_num, metric=self.args.eval_metric,
                                                       csls_k=self.args.csls)
        if save:
            ent_ids_rest_12 = [(self.kgs.test_entities1[i], self.kgs.test_entities2[j]) for i, j in rest_12]
            read.save_results(self.out_folder, ent_ids_rest_12)

    def run(self):
        training_data_list = generate_training_data(self.kgs, threshold=0.9)
        steps = len(training_data_list) // self.args.batch_size
        for i in range(1, self.args.attr_max_epoch + 1):
            self.launch_training_1epo(i, steps, training_data_list)
            # if i % 10 == 0:
            #     self.valid()

    def save(self):
        embeds = self.embeds.eval(session=self.session)
        read.save_embeddings(self.out_folder, self.kgs, None, None, embeds)