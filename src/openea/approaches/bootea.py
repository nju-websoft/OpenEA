import gc
import math
import multiprocessing as mp
import random
import time
import numpy as np
import tensorflow as tf

from openea.modules.finding.evaluation import early_stop
import openea.modules.train.batch as bat
from openea.approaches.aligne import AlignE
from openea.modules.utils.util import task_divide
from openea.modules.bootstrapping.alignment_finder import find_potential_alignment_mwgm, check_new_alignment
from openea.modules.base.optimizers import generate_optimizer
from openea.modules.load.kg import KG
from openea.modules.utils.util import load_session


def bootstrapping(sim_mat, unaligned_entities1, unaligned_entities2, labeled_alignment, sim_th, k):
    curr_labeled_alignment = find_potential_alignment_mwgm(sim_mat, sim_th, k)
    if curr_labeled_alignment is not None:
        labeled_alignment = update_labeled_alignment_x(labeled_alignment, curr_labeled_alignment, sim_mat)
        labeled_alignment = update_labeled_alignment_y(labeled_alignment, sim_mat)
        del curr_labeled_alignment
    if labeled_alignment is not None:
        newly_aligned_entities1 = [unaligned_entities1[pair[0]] for pair in labeled_alignment]
        newly_aligned_entities2 = [unaligned_entities2[pair[1]] for pair in labeled_alignment]
    else:
        newly_aligned_entities1, newly_aligned_entities2 = None, None
    del sim_mat
    gc.collect()
    return labeled_alignment, newly_aligned_entities1, newly_aligned_entities2


def update_labeled_alignment_x(pre_labeled_alignment, curr_labeled_alignment, sim_mat):
    labeled_alignment_dict = dict(pre_labeled_alignment)
    n1, n2 = 0, 0
    for i, j in curr_labeled_alignment:
        if labeled_alignment_dict.get(i, -1) == i and j != i:
            n2 += 1
        if i in labeled_alignment_dict.keys():
            pre_j = labeled_alignment_dict.get(i)
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


def calculate_likelihood_mat(ref_ent1, ref_ent2, labeled_alignment):
    def set2dic(alignment):
        if alignment is None:
            return None
        dic = dict()
        for i, j in alignment:
            dic[i] = j
        assert len(dic) == len(alignment)
        return dic

    t = time.time()
    ref_mat = np.zeros((len(ref_ent1), len(ref_ent2)), dtype=np.float32)
    if labeled_alignment is not None:
        alignment_dic = set2dic(labeled_alignment)
        n = 1 / len(ref_ent1)
        for ii in range(len(ref_ent1)):
            if ii in alignment_dic.keys():
                ref_mat[ii, alignment_dic.get(ii)] = 1
            else:
                for jj in range(len(ref_ent1)):
                    ref_mat[ii, jj] = n
    print("calculate likelihood matrix costs {:.2f} s".format(time.time() - t))
    return ref_mat


def generate_supervised_triples(rt_dict1, hr_dict1, rt_dict2, hr_dict2, ents1, ents2):
    assert len(ents1) == len(ents2)
    newly_triples1, newly_triples2 = list(), list()
    for i in range(len(ents1)):
        newly_triples1.extend(generate_newly_triples(ents1[i], ents2[i], rt_dict1, hr_dict1))
        newly_triples2.extend(generate_newly_triples(ents2[i], ents1[i], rt_dict2, hr_dict2))
    print("newly triples: {}, {}".format(len(newly_triples1), len(newly_triples2)))
    return newly_triples1, newly_triples2


def generate_newly_triples(ent1, ent2, rt_dict1, hr_dict1):
    newly_triples = list()
    for r, t in rt_dict1.get(ent1, set()):
        newly_triples.append((ent2, r, t))
    for h, r in hr_dict1.get(ent1, set()):
        newly_triples.append((h, r, ent2))
    return newly_triples


def generate_pos_batch(triples1, triples2, step, batch_size):
    num1 = int(len(triples1) / (len(triples1) + len(triples2)) * batch_size)
    num2 = batch_size - num1
    start1 = step * num1
    start2 = step * num2
    end1 = start1 + num1
    end2 = start2 + num2
    if end1 > len(triples1):
        end1 = len(triples1)
    if end2 > len(triples2):
        end2 = len(triples2)
    pos_triples1 = triples1[start1: end1]
    pos_triples2 = triples2[start2: end2]
    return pos_triples1, pos_triples2


def mul(tensor1, tensor2, session, num, sigmoid):
    t = time.time()
    if num < 20000:
        sim_mat = tf.matmul(tensor1, tensor2, transpose_b=True)
        if sigmoid:
            res = tf.sigmoid(sim_mat).eval(session=session)
        else:
            res = sim_mat.eval(session=session)
    else:
        res = np.matmul(tensor1.eval(session=session), tensor2.eval(session=session).T)
    print("mat mul costs: {:.3f}".format(time.time() - t))
    return res


class BootEA(AlignE):

    def __init__(self):
        super().__init__()
        self.ref_ent1 = None
        self.ref_ent2 = None

    def init(self):
        self._define_variables()
        self._define_embed_graph()
        self._define_alignment_graph()
        self._define_likelihood_graph()
        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)
        self.ref_ent1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2

        # customize parameters
        assert self.args.init == 'normal'
        assert self.args.alignment_module == 'swapping'
        assert self.args.loss == 'limited'
        assert self.args.neg_sampling == 'truncated'
        assert self.args.optimizer == 'Adagrad'
        assert self.args.eval_metric == 'inner'
        assert self.args.loss_norm == 'L2'

        assert self.args.ent_l2_norm is True
        assert self.args.rel_l2_norm is True

        assert self.args.pos_margin >= 0.0
        assert self.args.neg_margin > self.args.pos_margin

        assert self.args.neg_triple_num > 1
        assert self.args.truncated_epsilon > 0.0
        assert self.args.learning_rate >= 0.01

    def _define_alignment_graph(self):
        self.new_h = tf.placeholder(tf.int32, shape=[None])
        self.new_r = tf.placeholder(tf.int32, shape=[None])
        self.new_t = tf.placeholder(tf.int32, shape=[None])
        phs = tf.nn.embedding_lookup(self.ent_embeds, self.new_h)
        prs = tf.nn.embedding_lookup(self.rel_embeds, self.new_r)
        pts = tf.nn.embedding_lookup(self.ent_embeds, self.new_t)
        self.alignment_loss = - tf.reduce_sum(tf.log(tf.sigmoid(-tf.reduce_sum(tf.pow(phs + prs - pts, 2), 1))))
        self.alignment_optimizer = generate_optimizer(self.alignment_loss, self.args.learning_rate,
                                                      opt=self.args.optimizer)

    def _define_likelihood_graph(self):
        self.entities1 = tf.placeholder(tf.int32, shape=[None])
        self.entities2 = tf.placeholder(tf.int32, shape=[None])
        dim = len(self.kgs.valid_links) + len(self.kgs.test_entities1)
        dim1 = self.args.likelihood_slice
        self.likelihood_mat = tf.placeholder(tf.float32, shape=[dim1, dim])
        ent1_embed = tf.nn.embedding_lookup(self.ent_embeds, self.entities1)
        ent2_embed = tf.nn.embedding_lookup(self.ent_embeds, self.entities2)
        mat = tf.log(tf.sigmoid(tf.matmul(ent1_embed, ent2_embed, transpose_b=True)))
        self.likelihood_loss = -tf.reduce_sum(tf.multiply(mat, self.likelihood_mat))
        self.likelihood_optimizer = generate_optimizer(self.likelihood_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)

    def eval_ref_sim_mat(self):
        refs1_embeddings = tf.nn.embedding_lookup(self.ent_embeds, self.ref_ent1)
        refs2_embeddings = tf.nn.embedding_lookup(self.ent_embeds, self.ref_ent2)
        refs1_embeddings = tf.nn.l2_normalize(refs1_embeddings, 1).eval(session=self.session)
        refs2_embeddings = tf.nn.l2_normalize(refs2_embeddings, 1).eval(session=self.session)
        return np.matmul(refs1_embeddings, refs2_embeddings.T)

    def launch_training_k_epo(self, iter, iter_nums, triple_steps, steps_tasks, training_batch_queue, neighbors1,
                              neighbors2):
        for i in range(1, iter_nums + 1):
            epoch = (iter - 1) * iter_nums + i
            self.launch_triple_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1,
                                             neighbors2)

    def train_alignment(self, kg1: KG, kg2: KG, entities1, entities2, training_epochs):
        if entities1 is None or len(entities1) == 0:
            return
        newly_tris1, newly_tris2 = generate_supervised_triples(kg1.rt_dict, kg1.hr_dict, kg2.rt_dict, kg2.hr_dict,
                                                               entities1, entities2)
        steps = math.ceil(((len(newly_tris1) + len(newly_tris2)) / self.args.batch_size))
        if steps == 0:
            steps = 1
        for i in range(training_epochs):
            t1 = time.time()
            alignment_loss = 0
            for step in range(steps):
                newly_batch1, newly_batch2 = generate_pos_batch(newly_tris1, newly_tris2, step, self.args.batch_size)
                newly_batch1.extend(newly_batch2)
                alignment_fetches = {"loss": self.alignment_loss, "train_op": self.alignment_optimizer}
                alignment_feed_dict = {self.new_h: [tr[0] for tr in newly_batch1],
                                       self.new_r: [tr[1] for tr in newly_batch1],
                                       self.new_t: [tr[2] for tr in newly_batch1]}
                alignment_vals = self.session.run(fetches=alignment_fetches, feed_dict=alignment_feed_dict)
                alignment_loss += alignment_vals["loss"]
            alignment_loss /= (len(newly_tris1) + len(newly_tris2))
            print("alignment_loss = {:.3f}, time = {:.3f} s".format(alignment_loss, time.time() - t1))

    def likelihood(self, labeled_alignment):
        t = time.time()
        likelihood_mat = calculate_likelihood_mat(self.ref_ent1, self.ref_ent2, labeled_alignment)
        likelihood_fetches = {"likelihood_loss": self.likelihood_loss, "likelihood_op": self.likelihood_optimizer}
        likelihood_loss = 0.0
        steps = len(self.ref_ent1) // self.args.likelihood_slice
        ref_ent1_array = np.array(self.ref_ent1)
        ll = list(range(len(self.ref_ent1)))
        # print(steps)
        for i in range(steps):
            idx = random.sample(ll, self.args.likelihood_slice)
            likelihood_feed_dict = {self.entities1: ref_ent1_array[idx],
                                    self.entities2: self.ref_ent2,
                                    self.likelihood_mat: likelihood_mat[idx, :]}
            vals = self.session.run(fetches=likelihood_fetches, feed_dict=likelihood_feed_dict)
            likelihood_loss += vals["likelihood_loss"]
        print("likelihood_loss = {:.3f}, time = {:.3f} s".format(likelihood_loss, time.time() - t))

    def run(self):
        t = time.time()
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        neighbors1, neighbors2 = None, None
        labeled_align = set()
        sub_num = self.args.sub_epoch
        iter_nums = self.args.max_epoch // sub_num
        for i in range(1, iter_nums + 1):
            print("\niteration", i)
            self.launch_training_k_epo(i, sub_num, triple_steps, steps_tasks, training_batch_queue, neighbors1,
                                       neighbors2)
            if i * sub_num >= self.args.start_valid:
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == iter_nums:
                    break
            labeled_align, entities1, entities2 = bootstrapping(self.eval_ref_sim_mat(),
                                                                self.ref_ent1, self.ref_ent2, labeled_align,
                                                                self.args.sim_th, self.args.k)
            self.train_alignment(self.kgs.kg1, self.kgs.kg2, entities1, entities2, 1)
            # self.likelihood(labeled_align)
            if i * sub_num >= self.args.start_valid:
                self.valid(self.args.stop_metric)
            t1 = time.time()
            assert 0.0 < self.args.truncated_epsilon < 1.0
            neighbors_num1 = int((1 - self.args.truncated_epsilon) * self.kgs.kg1.entities_num)
            neighbors_num2 = int((1 - self.args.truncated_epsilon) * self.kgs.kg2.entities_num)
            if neighbors1 is not None:
                del neighbors1, neighbors2
            gc.collect()
            # neighbors1 = bat.generate_neighbours(self.eval_kg1_useful_ent_embeddings(),
            #                                      self.kgs.useful_entities_list1,
            #                                      neighbors_num1, self.args.batch_threads_num)
            # neighbors2 = bat.generate_neighbours(self.eval_kg2_useful_ent_embeddings(),
            #                                      self.kgs.useful_entities_list2,
            #                                      neighbors_num2, self.args.batch_threads_num)
            neighbors1 = bat.generate_neighbours_single_thread(self.eval_kg1_useful_ent_embeddings(),
                                                               self.kgs.useful_entities_list1,
                                                               neighbors_num1, self.args.test_threads_num)
            neighbors2 = bat.generate_neighbours_single_thread(self.eval_kg2_useful_ent_embeddings(),
                                                               self.kgs.useful_entities_list2,
                                                               neighbors_num2, self.args.test_threads_num)
            ent_num = len(self.kgs.kg1.entities_list) + len(self.kgs.kg2.entities_list)
            print("generating neighbors of {} entities costs {:.3f} s.".format(ent_num, time.time() - t1))
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
