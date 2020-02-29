import tensorflow as tf
import math
import multiprocessing as mp
import random
import time

import openea.modules.train.batch as bat
from openea.modules.finding.evaluation import early_stop
from openea.modules.utils.util import task_divide
from openea.modules.base.optimizers import generate_optimizer
from openea.modules.utils.util import load_session
from openea.modules.base.initializers import init_embeddings
from openea.models.basic_model import BasicModel
from openea.modules.base.losses import get_loss_func


def formatting_attr_triples(kgs, literal_len):
    """
    Formatting attribute triples from kgs for AttrE.
    :param kgs: modules.load.kgs
    :param literal_len: [optional] Literal truncation length, taking the first literal_len characters.
    :return: attribute_triples_list1_new, attribute_triples_list2_new, char_list size
    """

    def clean_attribute_triples(triples):
        triples_new = []
        for (e, a, v) in triples:
            v = v.split('(')[0].rstrip(' ')
            v = v.replace('.', '').replace('(', '').replace(')', '').replace(',', '') \
                .replace('_', ' ').replace('-', ' ').split('"')[0]
            triples_new.append((e, a, v))
        return triples_new

    attribute_triples_list1 = clean_attribute_triples(kgs.kg1.local_attribute_triples_list)
    attribute_triples_list2 = clean_attribute_triples(kgs.kg2.local_attribute_triples_list)

    value_list = list(set([v for (_, _, v) in attribute_triples_list1 + attribute_triples_list2]))
    char_set = set()
    ch_num = {}
    for literal in value_list:
        for ch in literal:
            n = 1
            if ch in ch_num:
                n += ch_num[ch]
            ch_num[ch] = n

    ch_num = sorted(ch_num.items(), key=lambda x: x[1], reverse=True)
    ch_sum = sum([n for (_, n) in ch_num])
    for i in range(len(ch_num)):
        if ch_num[i][1] / ch_sum >= 0.0001:
            char_set.add(ch_num[i][0])
    char_list = list(char_set)
    char_id_dict = {}
    for i in range(len(char_list)):
        char_id_dict[char_list[i]] = i + 1

    value_char_ids_dict = {}
    for value in value_list:
        char_id_list = [0 for _ in range(literal_len)]
        for i in range(min(len(value), literal_len)):
            if value[i] in char_set:
                char_id_list[i] = char_id_dict[value[i]]
        value_char_ids_dict[value] = char_id_list

    attribute_triples_list1_new, attribute_triples_list2_new = list(), list()
    value_id_char_ids = list()
    value_id_cnt = 0
    for (e_id, a_id, v) in attribute_triples_list1:
        attribute_triples_list1_new.append((e_id, a_id, value_id_cnt))
        value_id_char_ids.append(value_char_ids_dict[v])
        value_id_cnt += 1

    for (e_id, a_id, v) in attribute_triples_list2:
        attribute_triples_list2_new.append((e_id, a_id, value_id_cnt))
        value_id_char_ids.append(value_char_ids_dict[v])
        value_id_cnt += 1
    return attribute_triples_list1_new, attribute_triples_list2_new, value_id_char_ids, len(char_list) + 1


def add_compositional_func(character_vectors):
    value_vector_list = tf.reduce_mean(character_vectors, axis=1)
    value_vector_list = tf.nn.l2_normalize(value_vector_list, 1)
    return value_vector_list


def n_gram_compositional_func(character_vectors, value_lens, batch_size, embed_size):
    pos_c_e_in_lstm = tf.unstack(character_vectors, num=value_lens, axis=1)
    pos_c_e_lstm = calculate_ngram_weight(pos_c_e_in_lstm, batch_size, embed_size)
    return pos_c_e_lstm


def calculate_ngram_weight(unstacked_tensor, batch_size, embed_size):
    stacked_tensor = tf.stack(unstacked_tensor, axis=1)
    stacked_tensor = tf.reverse(stacked_tensor, [1])
    index = tf.constant(len(unstacked_tensor))
    expected_result = tf.zeros([batch_size, embed_size])

    def condition(index, summation):
        return tf.greater(index, 0)

    def body(index, summation):
        precessed = tf.slice(stacked_tensor, [0, index - 1, 0], [-1, -1, -1])
        summand = tf.reduce_mean(precessed, 1)
        return tf.subtract(index, 1), tf.add(summation, summand)

    result = tf.while_loop(condition, body, [index, expected_result])
    return result[1]


class AttrE(BasicModel):

    def __init__(self):
        super().__init__()

    def init(self):
        self.attribute_triples_list1, self.attribute_triples_list2, self.value_id_char_ids, self.char_list_size = \
            formatting_attr_triples(self.kgs, self.args.literal_len)
        self._define_variables()
        self._define_embed_graph()
        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)

    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'ent_embeds',
                                              self.args.init, self.args.ent_l2_norm, dtype=tf.float32)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds',
                                              self.args.init, self.args.rel_l2_norm, dtype=tf.float32)
        with tf.variable_scope('character' + 'embeddings'):
            self.ent_embeds_ce = init_embeddings([self.kgs.entities_num, self.args.dim], 'ent_embeds_ce',
                                                 self.args.init, self.args.ent_l2_norm, dtype=tf.float32)
            self.attr_embeds = init_embeddings([self.kgs.attributes_num, self.args.dim], 'attr_embeds',
                                               self.args.init, self.args.attr_l2_norm, dtype=tf.float32)
            self.char_embeds = init_embeddings([self.char_list_size, self.args.dim], 'char_embeds',
                                               self.args.init, self.args.char_l2_norm, dtype=tf.float32)
            self.value_id_char_ids = tf.constant(self.value_id_char_ids)

    def _define_embed_graph(self):
        with tf.name_scope('triple_placeholder'):
            self.pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.pos_ts = tf.placeholder(tf.int32, shape=[None])
            self.neg_hs = tf.placeholder(tf.int32, shape=[None])
            self.neg_rs = tf.placeholder(tf.int32, shape=[None])
            self.neg_ts = tf.placeholder(tf.int32, shape=[None])

            self.pos_es = tf.placeholder(tf.int32, shape=[None])
            self.pos_as = tf.placeholder(tf.int32, shape=[None])
            self.pos_vs = tf.placeholder(tf.int32, shape=[None])
            self.neg_es = tf.placeholder(tf.int32, shape=[None])
            self.neg_as = tf.placeholder(tf.int32, shape=[None])
            self.neg_vs = tf.placeholder(tf.int32, shape=[None])

            self.joint_ents = tf.placeholder(tf.int32, shape=[None])

        with tf.name_scope('triple_lookup'):
            phs = tf.nn.embedding_lookup(self.ent_embeds, self.pos_hs)
            prs = tf.nn.embedding_lookup(self.rel_embeds, self.pos_rs)
            pts = tf.nn.embedding_lookup(self.ent_embeds, self.pos_ts)
            nhs = tf.nn.embedding_lookup(self.ent_embeds, self.neg_hs)
            nrs = tf.nn.embedding_lookup(self.rel_embeds, self.neg_rs)
            nts = tf.nn.embedding_lookup(self.ent_embeds, self.neg_ts)

            pes = tf.nn.embedding_lookup(self.ent_embeds_ce, self.pos_es)
            pas = tf.nn.embedding_lookup(self.attr_embeds, self.pos_as)
            pvs = tf.nn.embedding_lookup(self.char_embeds, tf.nn.embedding_lookup(self.value_id_char_ids, self.pos_vs))
            nes = tf.nn.embedding_lookup(self.ent_embeds_ce, self.neg_es)
            nas = tf.nn.embedding_lookup(self.attr_embeds, self.neg_as)
            nvs = tf.nn.embedding_lookup(self.char_embeds, tf.nn.embedding_lookup(self.value_id_char_ids, self.neg_vs))

            pvs = n_gram_compositional_func(pvs, self.args.literal_len, self.args.batch_size, self.args.dim)
            nvs = n_gram_compositional_func(nvs, self.args.literal_len,
                                            self.args.batch_size * self.args.neg_triple_num, self.args.dim)

            ents_se = tf.nn.embedding_lookup(self.ent_embeds, self.joint_ents)
            ents_ce = tf.nn.embedding_lookup(self.ent_embeds_ce, self.joint_ents)

        with tf.name_scope('triple_loss'):
            self.triple_loss = get_loss_func(phs, prs, pts, nhs, nrs, nts, self.args)
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)

            self.triple_loss_ce = get_loss_func(pes, pas, pvs, nes, nas, nvs, self.args)
            self.triple_optimizer_ce = generate_optimizer(self.triple_loss_ce, self.args.learning_rate,
                                                          opt=self.args.optimizer)

            cos_sim = tf.reduce_sum(tf.multiply(ents_se, ents_ce), 1, keep_dims=True)
            self.joint_loss = tf.reduce_sum(1 - cos_sim)
            self.optimizer_joint = generate_optimizer(self.joint_loss, self.args.learning_rate, opt=self.args.optimizer)

    def launch_triple_training_1epo_ce(self, epoch, triple_steps, steps_tasks, batch_queue):
        start = time.time()
        for steps_task in steps_tasks:
            mp.Process(target=bat.generate_attribute_triple_batch_queue,
                       args=(self.attribute_triples_list1, self.attribute_triples_list2,
                             set(self.attribute_triples_list1), set(self.attribute_triples_list2),
                             self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                             self.args.batch_size, steps_task,
                             batch_queue, None, None, self.args.neg_triple_num, True)).start()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            batch_pos, batch_neg = batch_queue.get()
            batch_loss, _ = self.session.run(fetches=[self.triple_loss_ce, self.triple_optimizer_ce],
                                             feed_dict={self.pos_es: [x[0] for x in batch_pos],
                                                        self.pos_as: [x[1] for x in batch_pos],
                                                        self.pos_vs: [x[2] for x in batch_pos],
                                                        self.neg_es: [x[0] for x in batch_neg],
                                                        self.neg_as: [x[1] for x in batch_neg],
                                                        self.neg_vs: [x[2] for x in batch_neg]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        random.shuffle(self.attribute_triples_list1)
        random.shuffle(self.attribute_triples_list2)
        print(
            'epoch {}, CE, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def launch_joint_training_1epo(self, epoch, entities):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(entities) / self.args.batch_size))
        for i in range(steps):
            batch_ents = list(entities)
            batch_loss, _ = self.session.run(fetches=[self.joint_loss, self.optimizer_joint],
                                             feed_dict={self.joint_ents: batch_ents})
            trained_samples_num += len(batch_ents)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        print('epoch {}, joint learning loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def run(self):
        t = time.time()
        relation_triples_num = len(self.kgs.kg1.relation_triples_list) + len(self.kgs.kg2.relation_triples_list)
        attribute_triples_num = len(self.attribute_triples_list1) + len(self.attribute_triples_list2)
        relation_triple_steps = int(math.ceil(relation_triples_num / self.args.batch_size))
        attribute_triple_steps = int(math.ceil(attribute_triples_num / self.args.batch_size))
        relation_step_tasks = task_divide(list(range(relation_triple_steps)), self.args.batch_threads_num)
        attribute_step_tasks = task_divide(list(range(attribute_triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        relation_batch_queue = manager.Queue()
        attribute_batch_queue = manager.Queue()
        entity_list = list(self.kgs.kg1.entities_list + self.kgs.kg2.entities_list)
        for i in range(1, self.args.max_epoch + 1):
            self.launch_triple_training_1epo(i, relation_triple_steps, relation_step_tasks, relation_batch_queue, None,
                                             None)
            # self.launch_triple_training_1epo_ce(i, attribute_triple_steps, attribute_step_tasks, attribute_batch_queue)
            # self.launch_joint_training_1epo(i, entity_list)
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
