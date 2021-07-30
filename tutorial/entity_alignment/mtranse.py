import math
import multiprocessing as mp
import random
import time
import gc

import tensorflow as tf
import numpy as np
from sklearn import preprocessing

import openea.modules.train.batch as bat
from openea.modules.finding.evaluation import early_stop
from openea.modules.utils.util import generate_out_folder
from openea.modules.utils.util import load_session
from openea.modules.utils.util import task_divide
from openea.modules.base.initializers import init_embeddings, orthogonal_init
from openea.modules.base.losses import margin_loss
from openea.modules.base.optimizers import generate_optimizer

from utils import MyKGs, find_alignment
from eval import valid, test
from nn_search import generate_neighbours


def search_kg1_to_kg2_1nn_neighbor(embeds1, embeds2, ents2, mapping_mat, return_sim=False, soft_nn=10):
    if mapping_mat is not None:
        embeds1 = np.matmul(embeds1, mapping_mat)
        embeds1 = preprocessing.normalize(embeds1)
    sim_mat = np.matmul(embeds1, embeds2.T)
    nearest_pairs = find_alignment(sim_mat, soft_nn)
    nns = [ents2[x[0][1]] for x in nearest_pairs]
    if return_sim:
        sim_list = []
        for pair in nearest_pairs:
            sim_list.append(sim_mat[pair[0][0], pair[0][1]])
        return nns, sim_list
    return nns


class MTransEV2:

    def set_kgs(self, kgs: MyKGs):
        self.kgs = kgs

    def set_args(self, args):
        self.args = args
        self.out_folder = generate_out_folder(self.args.output, self.args.training_data, self.args.dataset_division,
                                              self.__class__.__name__)

    def init(self):
        self._define_variables()
        self._define_embed_graph()
        self._define_align_graph()

        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)

    def __init__(self):

        self.out_folder = None
        self.args = None
        self.kgs = None

        self.session = None

        self.seed_entities1 = None
        self.seed_entities2 = None
        self.neg_ts = None
        self.neg_rs = None
        self.neg_hs = None
        self.pos_ts = None
        self.pos_rs = None
        self.pos_hs = None

        self.rel_embeds = None
        self.ent_embeds = None
        self.mapping_mat = None
        self.eye_mat = None

        self.triple_optimizer = None
        self.triple_loss = None
        self.mapping_optimizer = None
        self.mapping_loss = None

        self.mapping_mat = None

        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = False

    def _define_variables(self):
        with tf.variable_scope('KG' + 'embeddings'):
            self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'ent_embeds',
                                              self.args.init, self.args.ent_l2_norm)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds',
                                              self.args.init, self.args.rel_l2_norm)

        with tf.variable_scope('alignment' + 'mapping'):
            self.mapping_mat = orthogonal_init([self.args.dim, self.args.dim], 'mapping_matrix')
            self.eye_mat = tf.constant(np.eye(self.args.dim), dtype=tf.float32, name='eye')

    def _define_embed_graph(self):
        print("build embedding learning graph...")
        with tf.name_scope('triple_placeholder'):
            self.pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.pos_ts = tf.placeholder(tf.int32, shape=[None])
            self.neg_hs = tf.placeholder(tf.int32, shape=[None])
            self.neg_rs = tf.placeholder(tf.int32, shape=[None])
            self.neg_ts = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('triple_lookup'):
            phs = tf.nn.embedding_lookup(self.ent_embeds, self.pos_hs)
            prs = tf.nn.embedding_lookup(self.rel_embeds, self.pos_rs)
            pts = tf.nn.embedding_lookup(self.ent_embeds, self.pos_ts)
            nhs = tf.nn.embedding_lookup(self.ent_embeds, self.neg_hs)
            nrs = tf.nn.embedding_lookup(self.rel_embeds, self.neg_rs)
            nts = tf.nn.embedding_lookup(self.ent_embeds, self.neg_ts)
        with tf.name_scope('triple_loss'):
            self.triple_loss = margin_loss(phs, prs, pts, nhs, nrs, nts, self.args.embed_margin, self.args.loss_norm)
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)

    def _mapping_align_loss(self, seed_embed1, seed_embed2):
        mapped_seed_embed1 = tf.matmul(seed_embed1, self.mapping_mat)
        mapped_seed_embed1 = tf.nn.l2_normalize(mapped_seed_embed1, 1)
        with tf.name_scope('mapping_distance'):
            distance = mapped_seed_embed1 - seed_embed2
        with tf.name_scope('mapping_loss'):
            align_loss = tf.reduce_sum(tf.reduce_sum(tf.square(distance), axis=1))
        orthogonal_loss = tf.reduce_mean(
            tf.reduce_sum(tf.pow(tf.matmul(self.mapping_mat, self.mapping_mat, transpose_b=True) - self.eye_mat, 2), 1))
        return align_loss + orthogonal_loss

    def _mapping_align_marginal_loss(self, seed_embed1, seed_embed2, pos_embed1, neg_embed2):
        mapped_seed_embed1 = tf.matmul(seed_embed1, self.mapping_mat)
        mapped_seed_embed1 = tf.nn.l2_normalize(mapped_seed_embed1, 1)
        with tf.name_scope('mapping_distance'):
            distance = mapped_seed_embed1 - seed_embed2
        with tf.name_scope('mapping_loss'):
            pos_score = tf.reduce_sum(tf.square(distance), axis=1)
            align_loss = tf.reduce_sum(pos_score)
            if self.args.mapping_margin > 0.0:
                mapped_pos_embed1 = tf.matmul(pos_embed1, self.mapping_mat)
                mapped_pos_embed1 = tf.nn.l2_normalize(mapped_pos_embed1, 1)
                neg_distance = mapped_pos_embed1 - neg_embed2
                neg_score = tf.reduce_sum(tf.square(neg_distance), axis=1)
                align_loss += self.args.mapping_neg_weight * \
                              tf.reduce_sum(tf.nn.relu(tf.constant(self.args.mapping_margin) - neg_score))
            orthogonal_loss = tf.reduce_mean(tf.reduce_sum(
                tf.square(tf.matmul(self.mapping_mat, self.mapping_mat, transpose_b=True) - self.eye_mat), 1))
        return align_loss + orthogonal_loss

    def _define_align_graph(self):
        print("build alignment learning graph...")
        with tf.name_scope('seed_links_placeholder'):
            self.seed_entities1 = tf.placeholder(tf.int32, shape=[None])
            self.seed_entities2 = tf.placeholder(tf.int32, shape=[None])
            self.pos_entities1 = tf.placeholder(tf.int32, shape=[None])
            self.neg_entities2 = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('seed_links_lookup'):
            seed_embed1 = tf.nn.embedding_lookup(self.ent_embeds, self.seed_entities1)
            seed_embed2 = tf.nn.embedding_lookup(self.ent_embeds, self.seed_entities2)
            pos_embed1 = tf.nn.embedding_lookup(self.ent_embeds, self.pos_entities1)
            neg_embed2 = tf.nn.embedding_lookup(self.ent_embeds, self.neg_entities2)
        with tf.name_scope('mapping_loss'):
            self.mapping_loss = self._mapping_align_marginal_loss(seed_embed1, seed_embed2, pos_embed1, neg_embed2)
            self.mapping_optimizer = generate_optimizer(self.mapping_loss, self.args.learning_rate,
                                                        opt=self.args.optimizer)

    def _eval_valid_embeddings(self, remove_dangling=False):
        if remove_dangling:
            candidate_list = self.kgs.valid_entities2 + list(self.kgs.kg2.entities_set
                                                             - set(self.kgs.train_entities2)
                                                             - set(self.kgs.valid_entities2)
                                                             - set([x for x, _ in self.kgs.train_unlinked_entities2])
                                                             - set([x for x, _ in self.kgs.valid_unlinked_entities2]))
        else:
            candidate_list = self.kgs.valid_entities2 + list(self.kgs.kg2.entities_set
                                                             - set(self.kgs.train_entities2)
                                                             - set(self.kgs.valid_entities2))
        embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.valid_entities1).eval(session=self.session)
        embeds2 = tf.nn.embedding_lookup(self.ent_embeds, candidate_list).eval(session=self.session)
        mapping = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        return embeds1, embeds2, mapping

    def _eval_test_embeddings(self, remove_dangling=False):
        if remove_dangling:
            candidate_list = self.kgs.test_entities2 + list(self.kgs.kg2.entities_set
                                                            - set(self.kgs.train_entities2)
                                                            - set(self.kgs.valid_entities2)
                                                            - set(self.kgs.test_entities2)
                                                            - set([x for x, _ in self.kgs.train_unlinked_entities2])
                                                            - set([x for x, _ in self.kgs.valid_unlinked_entities2]))
        else:
            candidate_list = self.kgs.test_entities2 + list(self.kgs.kg2.entities_set
                                                            - set(self.kgs.train_entities2)
                                                            - set(self.kgs.valid_entities2)
                                                            - set(self.kgs.test_entities2))
        embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities1).eval(session=self.session)
        embeds2 = tf.nn.embedding_lookup(self.ent_embeds, candidate_list).eval(session=self.session)
        mapping = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        return embeds1, embeds2, mapping

    def valid_alignment(self, stop_metric, remove_dangling):
        print("\nvalidating entity alignment...")
        embeds1, embeds2, mapping = self._eval_valid_embeddings(remove_dangling=remove_dangling)
        hits, mrr_12, sim_list = valid(embeds1, embeds2, mapping, self.args.top_k,
                                       self.args.test_threads_num, metric=self.args.eval_metric,
                                       normalize=self.args.eval_norm, csls_k=0, accurate=False)
        print()
        return hits[0] if stop_metric == 'hits1' else mrr_12

    def test(self):
        embeds1, embeds2, mapping = self._eval_test_embeddings(remove_dangling=False)
        _, _, _, sim_list = test(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
                                 metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        print()

    def eval_embeddings(self, entity_list):
        embeds = tf.nn.embedding_lookup(self.ent_embeds, entity_list)
        return embeds.eval(session=self.session)

    def launch_training_1epo(self, epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2):
        self.launch_embed_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2)
        self.launch_align_training_1epo(epoch, triple_steps)

    def launch_embed_training_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1, neighbors2):
        start = time.time()
        for steps_task in steps_tasks:
            mp.Process(target=bat.generate_relation_triple_batch_queue,
                       args=(self.kgs.kg1.relation_triples_list, self.kgs.kg2.relation_triples_list,
                             self.kgs.kg1.relation_triples_set, self.kgs.kg2.relation_triples_set,
                             self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                             self.args.batch_size, steps_task,
                             batch_queue, neighbors1, neighbors2, self.args.neg_triple_num)).start()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            batch_pos, batch_neg = batch_queue.get()
            batch_loss, _ = self.session.run(fetches=[self.triple_loss, self.triple_optimizer],
                                             feed_dict={self.pos_hs: [x[0] for x in batch_pos],
                                                        self.pos_rs: [x[1] for x in batch_pos],
                                                        self.pos_ts: [x[2] for x in batch_pos],
                                                        self.neg_hs: [x[0] for x in batch_neg],
                                                        self.neg_rs: [x[1] for x in batch_neg],
                                                        self.neg_ts: [x[2] for x in batch_neg]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        random.shuffle(self.kgs.kg1.relation_triples_list)
        random.shuffle(self.kgs.kg2.relation_triples_list)
        print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.1f}s'.format(epoch, epoch_loss, time.time() - start))

    def launch_align_training_1epo(self, epoch, triple_steps):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        batch_size = 2 * len(self.kgs.train_links) // triple_steps
        kg_training_ents2 = list(self.kgs.kg2.entities_set - set(self.kgs.train_entities2))

        neg_batch_size = batch_size * self.args.mapping_neg_num

        for i in range(triple_steps):
            links_batch = random.sample(self.kgs.train_links, batch_size)
            pos_entities1 = random.sample(self.kgs.kg1.entities_list, neg_batch_size)
            neg_entities2 = random.sample(kg_training_ents2, neg_batch_size)
            batch_loss, _ = self.session.run(fetches=[self.mapping_loss, self.mapping_optimizer],
                                             feed_dict={self.seed_entities1: [x[0] for x in links_batch],
                                                        self.seed_entities2: [x[1] for x in links_batch],
                                                        self.pos_entities1: pos_entities1,
                                                        self.neg_entities2: neg_entities2})
            epoch_loss += batch_loss
            trained_samples_num += len(links_batch)
        epoch_loss /= trained_samples_num
        print('epoch {}, avg. mapping loss: {:.4f}, cost time: {:.1}s'.format(epoch, epoch_loss, time.time() - start))

    def generate_neighbors(self):
        t1 = time.time()
        assert 0.0 < self.args.truncated_epsilon < 1.0

        num1 = len(self.kgs.kg1.entities_list) // 2
        if len(self.kgs.kg1.entities_list) > 200000:
            num1 = len(self.kgs.kg1.entities_list) // 3
        if num1 > len(self.kgs.useful_entities_list1):
            kg1_random_ents = self.kgs.useful_entities_list1 + \
                              random.sample(list(set(self.kgs.kg1.entities_list) - set(self.kgs.useful_entities_list1)),
                                            num1 - len(self.kgs.useful_entities_list1))
        else:
            kg1_random_ents = self.kgs.useful_entities_list1
        embeds1 = self.eval_embeddings(kg1_random_ents)

        num2 = len(self.kgs.kg2.entities_list) // 2
        if len(self.kgs.kg2.entities_list) > 200000:
            num2 = len(self.kgs.kg2.entities_list) // 3
        if num2 > len(self.kgs.useful_entities_list2):
            kg2_random_ents = self.kgs.useful_entities_list2 + \
                              random.sample(list(set(self.kgs.kg2.entities_list) - set(self.kgs.useful_entities_list2)),
                                            num2 - len(self.kgs.useful_entities_list2))
        else:
            kg2_random_ents = self.kgs.useful_entities_list2
        embeds2 = self.eval_embeddings(kg2_random_ents)
        neighbors_num1 = int((1 - self.args.truncated_epsilon) * num1)
        neighbors_num2 = int((1 - self.args.truncated_epsilon) * num2)
        print("generating neighbors...")
        neighbors1 = generate_neighbours(embeds1, kg1_random_ents, neighbors_num1,
                                         frags_num=self.args.batch_threads_num)
        neighbors2 = generate_neighbours(embeds2, kg2_random_ents, neighbors_num2,
                                         frags_num=self.args.batch_threads_num)
        print("generating neighbors ({}, {}) costs {:.3f} s.".format(num1, num2, time.time() - t1))
        gc.collect()
        return neighbors1, neighbors2

    def run(self):
        t = time.time()

        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        training_batch_manager = mp.Manager()
        training_batch_queue = training_batch_manager.Queue()
        neighbors1, neighbors2 = dict(), dict()

        # training
        for i in range(1, self.args.max_epoch + 1):
            self.launch_training_1epo(i, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2)
            # validation
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                flag = self.valid_alignment(self.args.stop_metric, False)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break
            # truncated sampling cache
            if self.args.neg_sampling == 'truncated' and i % self.args.truncated_freq == 0:
                if neighbors1 is not None:
                    del neighbors1, neighbors2
                    gc.collect()
                neighbors1, neighbors2 = self.generate_neighbors()

        print("Training ends. Total time = {:.1f} s.".format(time.time() - t))
