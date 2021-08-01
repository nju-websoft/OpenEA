import math
import multiprocessing as mp
import random
import time

import tensorflow as tf
import numpy as np
from sklearn import preprocessing

import openea.modules.load.read as rd
import openea.modules.train.batch as bat
from openea.modules.finding.evaluation import early_stop
from openea.modules.utils.util import generate_out_folder
from openea.modules.utils.util import load_session
from openea.modules.utils.util import task_divide
from openea.modules.base.initializers import init_embeddings, orthogonal_init
from openea.modules.base.losses import margin_loss
from openea.modules.base.optimizers import generate_optimizer

from utils import MyKGs, find_alignment
from eval import valid, test, greedy_alignment, eval_margin
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
        if self.args.detection_mode == "classification":
            self._define_classification_graph()
        elif self.args.detection_mode == "margin":
            self._define_distance_margin_graph()
        elif self.args.detection_mode == "open":
            self._define_open_margin_graph()
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
                align_loss += 0.1 * tf.reduce_sum(tf.nn.relu(tf.constant(self.args.mapping_margin) - neg_score))
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
            # self.mapping_loss = self._mapping_align_loss(seed_embed1, seed_embed2)
            self.mapping_loss = self._mapping_align_marginal_loss(seed_embed1, seed_embed2, pos_embed1, neg_embed2)
            self.mapping_optimizer = generate_optimizer(self.mapping_loss, self.args.learning_rate,
                                                        opt=self.args.optimizer)

    def _define_distance_margin_graph(self):
        print("build distance margin graph...")
        with tf.name_scope('entity_placeholder'):
            self.input_ents1 = tf.placeholder(tf.int32, shape=[None])
            self.input_ents2 = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('negative_alignment_entity_placeholder'):
            self.seed_pos_ents1 = tf.placeholder(tf.int32, shape=[None])
            self.negative_ents2 = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('entity_lookup'):
            x1 = tf.nn.embedding_lookup(self.ent_embeds, self.input_ents1)
            seed_x1 = tf.nn.embedding_lookup(self.ent_embeds, self.seed_pos_ents1)
            if self.mapping_mat is not None:
                x1 = tf.matmul(x1, self.mapping_mat)
                x1 = tf.nn.l2_normalize(x1, 1)
                seed_x1 = tf.matmul(seed_x1, self.mapping_mat)
                seed_x1 = tf.nn.l2_normalize(seed_x1, 1)
            x2 = tf.nn.embedding_lookup(self.ent_embeds, self.input_ents2)
            negative_x2 = tf.nn.embedding_lookup(self.ent_embeds, self.negative_ents2)
        with tf.name_scope('dis_margin_loss'):
            dis1 = tf.reduce_sum(tf.square(x1 - x2), axis=1)
            dis2 = tf.reduce_sum(tf.square(seed_x1 - negative_x2), axis=1)
        dis_loss = tf.reduce_sum(tf.nn.relu(self.args.distance_margin - dis1))
        # + tf.reduce_sum(tf.nn.relu(self.args.distance_margin // 2 - dis2))
        self.dis_loss = 0.1 * dis_loss
        self.dis_optimizer = generate_optimizer(self.dis_loss, self.args.learning_rate, opt=self.args.optimizer)

    def _eval_valid_embeddings(self):
        candidate_list = self.kgs.valid_entities2 + list(self.kgs.kg2.entities_set
                                                         - set(self.kgs.train_entities2)
                                                         - set(self.kgs.valid_entities2))
        embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.valid_entities1).eval(session=self.session)
        embeds2 = tf.nn.embedding_lookup(self.ent_embeds, candidate_list).eval(session=self.session)
        mapping = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        return embeds1, embeds2, mapping

    def _eval_test_embeddings(self):
        candidate_list = self.kgs.test_entities2 + list(self.kgs.kg2.entities_set
                                                        - set(self.kgs.train_entities2)
                                                        - set(self.kgs.valid_entities2)
                                                        - set(self.kgs.test_entities2))
        embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities1).eval(session=self.session)
        embeds2 = tf.nn.embedding_lookup(self.ent_embeds, candidate_list).eval(session=self.session)
        mapping = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        return embeds1, embeds2, mapping

    def valid_alignment(self, stop_metric):
        print("\nevaluating alignment (relaxed setting)...")
        embeds1, embeds2, mapping = self._eval_valid_embeddings()
        hits, mrr_12, sim_list = valid(embeds1, embeds2, mapping, self.args.top_k,
                                       self.args.test_threads_num, metric=self.args.eval_metric,
                                       normalize=self.args.eval_norm, csls_k=0, accurate=False)
        print()
        return hits[0] if stop_metric == 'hits1' else mrr_12

    def get_source_and_candidates(self, source_ents_and_labels, is_test):
        total_ent_embeds = self.ent_embeds.eval(session=self.session)
        mapping_mat = self.mapping_mat.eval(session=self.session)

        source_ents = [x[0] for x in source_ents_and_labels]
        source_embeds = total_ent_embeds[np.array(source_ents),]

        if is_test:
            target_candidates = list(set(self.kgs.kg2.entities_list) -
                                     set(self.kgs.train_entities2) -
                                     set(self.kgs.valid_entities2))
        else:
            target_candidates = list(set(self.kgs.kg2.entities_list) - set(self.kgs.train_entities2))
        target_embeds = total_ent_embeds[np.array(target_candidates),]
        source_ent_y = [x[1] for x in source_ents_and_labels]
        return source_embeds, source_ents, source_ent_y, target_embeds, target_candidates, mapping_mat

    def evaluate_margin(self, source_ents_and_labels, margin, is_test=False):
        print("dangling entity detection...")

        source_embeds, source_ents, source_ent_y, target_embeds, target_candidates, mapping_mat = \
            self.get_source_and_candidates(source_ents_and_labels, is_test)
        nns, sims = search_kg1_to_kg2_1nn_neighbor(source_embeds, target_embeds, target_candidates, mapping_mat,
                                                   return_sim=True)
        dis_vec = 1 - np.array(sims)
        mean_dis = np.mean(dis_vec)
        print(mean_dis, dis_vec)
        dis_list = dis_vec.tolist()

        return eval_margin(source_ents, dis_list, source_ent_y, margin=mean_dis)

    def real_entity_alignment_evaluation(self, label11_ents, label1_num, matchable_source_ents1):
        if label11_ents is None or len(label11_ents) == 0:
            print("no predicated matchable entities")
            return 0.
        total_ent_embeds = self.ent_embeds.eval(session=self.session)
        label11_source_embeds = total_ent_embeds[np.array(label11_ents),]
        mapping_mat = self.mapping_mat.eval(session=self.session)
        label11_source_embeds = np.matmul(label11_source_embeds, mapping_mat)

        true_targets = []
        matchable_ents1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        matchable_ents2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        for e in label11_ents:
            idx = matchable_ents1.index(e)
            true_targets.append(matchable_ents2[idx])
        assert len(true_targets) == len(label11_ents)
        candidate_list = true_targets + list(self.kgs.kg2.entities_set
                                             - set(self.kgs.train_entities2)
                                             - set(self.kgs.valid_entities2)
                                             - set(true_targets))
        candidate_embeds = total_ent_embeds[np.array(candidate_list),]
        _, hits, _, _, _ = greedy_alignment(label11_source_embeds, candidate_embeds,
                                            self.args.top_k, self.args.test_threads_num,
                                            self.args.eval_metric, True, 10, False, False)
        hits1 = hits[0]
        hits10 = hits[2]
        precision = hits1 * len(label11_ents) / label1_num
        recall = hits1 * len(label11_ents) / len(matchable_source_ents1)
        f1 = 2 * precision * recall / (precision + recall)
        recall10 = hits10 * len(label11_ents) / len(matchable_source_ents1)
        print("two-step results, precision = {:.3f}, recall = {:.3f}, f1 = {:.3f}, recall@10 = {:.3f}\n".format(
            precision, recall, f1, recall10))
        return f1

    def two_step_evaluation_margin(self, matchable_source_ents1, dangling_source_ents1, is_test=False):
        print("evaluating two-step alignment (margin)...")
        label11_ents, label1_num = self.evaluate_margin(matchable_source_ents1 + dangling_source_ents1,
                                                        self.args.distance_margin, is_test=is_test)
        return self.real_entity_alignment_evaluation(label11_ents, label1_num, matchable_source_ents1)

    def test(self):
        print("\ntesting synthetic alignment...")
        if self.args.detection_mode == "margin":
            embeds1, embeds2, mapping = self._eval_test_embeddings()
        else:
            embeds1, embeds2, mapping = self._eval_test_embeddings()
        _, _, _, sim_list = test(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
                                 metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        print()
        if self.args.detection_mode == "margin":
            self.two_step_evaluation_margin(self.kgs.test_linked_entities1,
                                            self.kgs.test_unlinked_entities1, is_test=True)
        print()

    def save(self):
        ent_embeds = self.ent_embeds.eval(session=self.session)
        rel_embeds = self.rel_embeds.eval(session=self.session)
        mapping_mat = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        rd.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, mapping_mat=mapping_mat)

    def eval_kg1_ent_embeddings(self):
        embeds = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.kg1.entities_list)
        return embeds.eval(session=self.session)

    def eval_kg2_ent_embeddings(self):
        embeds = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.kg2.entities_list)
        return embeds.eval(session=self.session)

    def eval_embeddings(self, entity_list):
        embeds = tf.nn.embedding_lookup(self.ent_embeds, entity_list)
        return embeds.eval(session=self.session)

    def launch_training_1epo(self, epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2):
        self.launch_embed_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2)
        self.launch_align_training_1epo(epoch, triple_steps)
        if self.args.detection_mode == "margin":
            self.launch_distance_margin_training_1epo(epoch, triple_steps)

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

    def launch_distance_margin_training_1epo(self, epoch, triple_steps):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0

        unlinked_entities1 = self.kgs.train_unlinked_entities1
        batch_size = len(unlinked_entities1) // triple_steps
        embeds = self.ent_embeds.eval(session=self.session)
        mapping_mat = self.mapping_mat.eval(session=self.session)
        steps_num = max(1, len(unlinked_entities1) // batch_size)

        ents2_candidates = list(self.kgs.kg2.entities_set - set(self.kgs.train_entities2))
        embeds2 = embeds[ents2_candidates,]

        other_entities1 = list(
            self.kgs.kg1.entities_set - set(self.kgs.train_entities1) - set(self.kgs.train_unlinked_entities1))

        neighbors_num2 = int((1 - self.args.truncated_epsilon) * len(self.kgs.kg2.entities_set))

        for i in range(steps_num):
            batch_data1 = random.sample(unlinked_entities1, batch_size)
            unlinked_ent1 = [x[0] for x in batch_data1]
            unlinked_embeds1 = embeds[np.array(unlinked_ent1),]
            unlinked_ent12 = search_kg1_to_kg2_1nn_neighbor(unlinked_embeds1, embeds2, ents2_candidates, mapping_mat,
                                                            soft_nn=self.args.soft_nn)
            other_ents1 = random.sample(other_entities1, batch_size)
            other_ents12 = random.sample(ents2_candidates, batch_size)

            batch_loss, _ = self.session.run(fetches=[self.dis_loss, self.dis_optimizer],
                                             feed_dict={self.input_ents1: unlinked_ent1,
                                                        self.input_ents2: unlinked_ent12,
                                                        self.seed_pos_ents1: other_ents1,
                                                        self.negative_ents2: other_ents12})
            epoch_loss += batch_loss
            trained_samples_num += len(batch_data1)
        epoch_loss /= trained_samples_num
        print('epoch {}, margin loss: {:.4f}, cost time: {:.1f}s'.format(epoch, epoch_loss,
                                                                         time.time() - start))

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
                # validation via synthetic alignment
                flag = self.valid_alignment(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                # validation via two-step alignment
                if self.args.detection_mode == "margin" and i > self.args.start_class:
                    flag = self.two_step_evaluation_margin(self.kgs.valid_linked_entities1,
                                                           self.kgs.valid_unlinked_entities1)
                    self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                # early stop
                if self.early_stop or i == self.args.max_epoch:
                    break
            # truncated sampling cache
            if self.args.neg_sampling == 'truncated' and i % self.args.truncated_freq == 0:
                neighbors1, neighbors2 = self.generate_neighbors()

        print("Training ends. Total time = {:.1f} s.".format(time.time() - t))
