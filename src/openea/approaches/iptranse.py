import math
import tensorflow as tf
import time
import sys
import random
import numpy as np
import pandas as pd
import multiprocessing as mp

from openea.modules.utils.util import task_divide
from openea.modules.finding.evaluation import early_stop
from openea.models.basic_model import BasicModel
from openea.modules.utils.util import load_session
from openea.modules.base.initializers import init_embeddings
from openea.modules.base.optimizers import generate_optimizer
import openea.modules.train.batch as bat
from openea.modules.load.kgs import KGs
from openea.modules.bootstrapping.alignment_finder import find_potential_alignment_greedily, check_new_alignment


def generate_neg_paths(pos_paths, rel_list):
    neg_paths = list()
    for (r_x, r_y, r, _) in pos_paths:
        r2 = random.sample(rel_list, 1)[0]
        neg_paths.append((r_x, r_y, r2))
    return neg_paths


def generate_newly_triples(ent1, ent2, w, rt_dict1, hr_dict1):
    newly_triples = set()
    for r, t in rt_dict1.get(ent1, set()):
        newly_triples.add((ent2, r, t, w))
    for h, r in hr_dict1.get(ent1, set()):
        newly_triples.add((h, r, ent2, w))
    return newly_triples


def generate_triples_of_latent_ents(kgs: KGs, ents1, ents2, tr_ws):
    assert len(ents1) == len(ents2)
    newly_triples = set()
    for i in range(len(ents1)):
        newly_triples |= generate_newly_triples(ents1[i], ents2[i], tr_ws[i], kgs.kg1.rt_dict, kgs.kg1.hr_dict)
        newly_triples |= generate_newly_triples(ents2[i], ents1[i], tr_ws[i], kgs.kg2.rt_dict, kgs.kg2.hr_dict)
    print("newly triples: {}".format(len(newly_triples)))
    return newly_triples


def generate_neg_triples_w(pos_triples, ents_list):
    neg_triples = list()
    for (h, r, t, w) in pos_triples:
        h2, r2, t2 = h, r, t
        choice = random.randint(0, 999)
        if choice < 500:
            h2 = random.sample(ents_list, 1)[0]
        elif choice >= 500:
            t2 = random.sample(ents_list, 1)[0]
        neg_triples.append((h2, r2, t2, w))
    return neg_triples


def generate_triple_batch(triples, batch_size, ents_list):
    if batch_size > len(triples):
        batch_size = len(triples)
    pos_triples = random.sample(triples, batch_size)
    neg_triples = generate_neg_triples_w(pos_triples, ents_list)
    return pos_triples, neg_triples


def generate_batch(kgs: KGs, paths1, paths2, batch_size, path_batch_size, step, neg_triple_num):
    pos_triples, neg_triples = bat.generate_relation_triple_batch(kgs.kg1.relation_triples_list,
                                                                  kgs.kg2.relation_triples_list,
                                                                  kgs.kg1.relation_triples_set,
                                                                  kgs.kg2.relation_triples_set,
                                                                  kgs.kg1.entities_list, kgs.kg2.entities_list,
                                                                  batch_size, step,
                                                                  None, None, neg_triple_num)
    num1 = int(len(paths1) / (len(paths1) + len(paths2)) * path_batch_size)
    num2 = path_batch_size - num1
    pos_paths1 = random.sample(paths1, num1)
    pos_paths2 = random.sample(paths2, num2)
    neg_paths1 = generate_neg_paths(pos_paths1, kgs.kg1.relations_list)
    neg_paths2 = generate_neg_paths(pos_paths2, kgs.kg2.relations_list)
    pos_paths1.extend(pos_paths2)
    neg_paths1.extend(neg_paths2)
    return pos_triples, neg_triples, pos_paths1, neg_paths1


def generate_batch_queue(kgs: KGs, paths1, paths2, batch_size, path_batch_size, steps, neg_triple_num, out_queue):
    for step in steps:
        pos_triples, neg_triples, pos_paths1, neg_paths1 = generate_batch(kgs, paths1, paths2, batch_size,
                                                                          path_batch_size, step, neg_triple_num)
        out_queue.put((pos_triples, neg_triples, pos_paths1, neg_paths1))


def generate_2steps_path(triples):
    tr = np.array([[tr[0], tr[2], tr[1]] for tr in triples])
    tr = pd.DataFrame(tr, columns=['h', 't', 'r'])
    sizes = tr.groupby(['h', 'r']).size()
    sizes.name = 'size'
    tr = tr.join(sizes, on=['h', 'r'])
    train_raw_df = tr[['h', 'r', 't', 'size']]
    two_step_df = pd.merge(train_raw_df, train_raw_df, left_on='t', right_on='h')
    print('start merge triple with path')

    two_step_df['_path_weight'] = two_step_df.size_x * two_step_df.size_y
    two_step_df = two_step_df[two_step_df['_path_weight'] < 101]
    two_step_df = pd.merge(two_step_df, train_raw_df, left_on=['h_x', 't_y'], right_on=['h', 't'], copy=False,
                           sort=False)
    # print(two_step_df[['r_x', 'r_y', 'r', '_path_weight']])
    path_mat = two_step_df[['r_x', 'r_y', 'r', '_path_weight']].values
    print("num of path:", path_mat.shape[0])
    path_list = list()
    for i in range(path_mat.shape[0]):
        path_list.append((path_mat[i][0], path_mat[i][1], path_mat[i][2], float(path_mat[i][3])))
    return path_list


class IPTransE(BasicModel):

    def __init__(self):
        super().__init__()
        self.ref_entities1, self.ref_entities2 = None, None
        self.paths1, self.paths2 = None, None

    def init(self):
        self.ref_entities1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_entities2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        self.paths1 = generate_2steps_path(self.kgs.kg1.relation_triples_list)
        self.paths2 = generate_2steps_path(self.kgs.kg2.relation_triples_list)
        self._define_variables()
        self._define_embed_graph()
        self._define_alignment_graph()
        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)

        # customize parameters
        assert self.args.alignment_module == 'sharing'
        assert self.args.init == 'normal'
        assert self.args.neg_sampling == 'uniform'
        assert self.args.optimizer == 'Adagrad'
        assert self.args.eval_metric == 'inner'
        assert self.args.loss_norm == 'L2'

        assert self.args.ent_l2_norm is True
        assert self.args.rel_l2_norm is True

        assert self.args.margin > 0.0
        assert self.args.neg_triple_num == 1
        assert self.args.sim_th > 0.0

    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'ent_embeds',
                                              self.args.init, self.args.ent_l2_norm)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds',
                                              self.args.init, self.args.rel_l2_norm)

    def _generate_transe_loss(self, phs, prs, pts, nhs, nrs, nts):
        if self.args.loss_norm == "L2":
            pos_score = tf.reduce_sum(tf.pow(phs + prs - pts, 2), 1)
            neg_score = tf.reduce_sum(tf.pow(nhs + nrs - nts, 2), 1)
        else:
            pos_score = tf.reduce_sum(tf.abs(phs + prs - pts), 1)
            neg_score = tf.reduce_sum(tf.abs(nhs + nrs - nts), 1)
        return tf.reduce_sum(tf.maximum(pos_score + self.args.margin - neg_score, 0))

    def _generate_transe_alignment_loss(self, phs, prs, pts, nhs, nrs, nts, ws):
        pos_score = tf.reduce_sum(tf.pow(phs + prs - pts, 2), 1)
        neg_score = tf.reduce_sum(tf.pow(nhs + nrs - nts, 2), 1)
        return tf.reduce_sum(ws * tf.maximum(pos_score + self.args.margin - neg_score, 0))
        # return tf.reduce_sum(ws * pos_score)

    def _generate_path_loss(self, prx, pry, pr, nrx, nry, nr, weight):
        pos_loss = tf.reduce_sum(tf.pow(prx + pry - pr, 2), 1)
        neg_loss = tf.reduce_sum(tf.pow(nrx + nry - nr, 2), 1)
        weight = tf.cast(1 / weight, dtype=tf.float32)
        return tf.reduce_sum(weight * tf.nn.relu(pos_loss + self.args.margin - neg_loss))

    def _generate_loss(self, phs, prs, pts, nhs, nrs, nts, prx, pry, pr, nrx, nry, nr, ws):
        return self._generate_transe_loss(phs, prs, pts, nhs, nrs, nts) + \
               self.args.path_parm * self._generate_path_loss(prx, pry, pr, nrx, nry, nr, ws)

    def _define_embed_graph(self):
        self.pos_hs = tf.placeholder(tf.int32, shape=[None])
        self.pos_rs = tf.placeholder(tf.int32, shape=[None])
        self.pos_ts = tf.placeholder(tf.int32, shape=[None])
        self.neg_hs = tf.placeholder(tf.int32, shape=[None])
        self.neg_rs = tf.placeholder(tf.int32, shape=[None])
        self.neg_ts = tf.placeholder(tf.int32, shape=[None])
        self.pos_rx = tf.placeholder(tf.int32, shape=[None])
        self.pos_ry = tf.placeholder(tf.int32, shape=[None])
        self.pos_r = tf.placeholder(tf.int32, shape=[None])
        self.neg_rx = tf.placeholder(tf.int32, shape=[None])
        self.neg_ry = tf.placeholder(tf.int32, shape=[None])
        self.neg_r = tf.placeholder(tf.int32, shape=[None])
        self.path_weight = tf.placeholder(tf.float32, shape=[None])

        phs = tf.nn.embedding_lookup(self.ent_embeds, self.pos_hs)
        prs = tf.nn.embedding_lookup(self.rel_embeds, self.pos_rs)
        pts = tf.nn.embedding_lookup(self.ent_embeds, self.pos_ts)

        nhs = tf.nn.embedding_lookup(self.ent_embeds, self.neg_hs)
        nrs = tf.nn.embedding_lookup(self.rel_embeds, self.neg_rs)
        nts = tf.nn.embedding_lookup(self.ent_embeds, self.neg_ts)

        prx = tf.nn.embedding_lookup(self.rel_embeds, self.pos_rx)
        pry = tf.nn.embedding_lookup(self.rel_embeds, self.pos_ry)
        pr = tf.nn.embedding_lookup(self.rel_embeds, self.pos_r)

        nrx = tf.nn.embedding_lookup(self.rel_embeds, self.neg_rx)
        nry = tf.nn.embedding_lookup(self.rel_embeds, self.neg_ry)
        nr = tf.nn.embedding_lookup(self.rel_embeds, self.neg_r)
        self.train_loss = self._generate_loss(phs, prs, pts, nhs, nrs, nts, prx, pry, pr, nrx, nry, nr,
                                              self.path_weight)
        self.optimizer = generate_optimizer(self.train_loss, self.args.learning_rate, opt=self.args.optimizer)

    def _define_alignment_graph(self):
        self.new_ph = tf.placeholder(tf.int32, shape=[None])
        self.new_pr = tf.placeholder(tf.int32, shape=[None])
        self.new_pt = tf.placeholder(tf.int32, shape=[None])
        self.new_nh = tf.placeholder(tf.int32, shape=[None])
        self.new_nr = tf.placeholder(tf.int32, shape=[None])
        self.new_nt = tf.placeholder(tf.int32, shape=[None])
        self.tr_weight = tf.placeholder(tf.float32, shape=[None])

        ph_embed = tf.nn.embedding_lookup(self.ent_embeds, self.new_ph)
        pr_embed = tf.nn.embedding_lookup(self.rel_embeds, self.new_pr)
        pt_embed = tf.nn.embedding_lookup(self.ent_embeds, self.new_pt)
        nh_embed = tf.nn.embedding_lookup(self.ent_embeds, self.new_nh)
        nr_embed = tf.nn.embedding_lookup(self.rel_embeds, self.new_nr)
        nt_embed = tf.nn.embedding_lookup(self.ent_embeds, self.new_nt)
        self.alignment_loss = self._generate_transe_alignment_loss(ph_embed, pr_embed, pt_embed, nh_embed, nr_embed,
                                                                   nt_embed, self.tr_weight)
        self.alignment_optimizer = generate_optimizer(self.alignment_loss, self.args.learning_rate,
                                                      opt=self.args.optimizer)

    def _ref_sim_mat(self):
        ref1_embeddings = tf.nn.embedding_lookup(self.ent_embeds, self.ref_entities1).eval(session=self.session)
        ref2_embeddings = tf.nn.embedding_lookup(self.ent_embeds, self.ref_entities2).eval(session=self.session)
        sim_mat = np.matmul(ref1_embeddings, ref2_embeddings.T)
        return sim_mat

    def launch_ptranse_training_1epo(self, epoch, triple_steps, steps_tasks, batch_queue):
        start = time.time()
        path_batch_size = (len(self.paths1) + len(self.paths2)) // triple_steps
        for steps_task in steps_tasks:
            mp.Process(target=generate_batch_queue,
                       args=(self.kgs, self.paths1, self.paths2, self.args.batch_size, path_batch_size,
                             steps_task, self.args.neg_triple_num, batch_queue)).start()
        fetches = {"loss": self.train_loss, "train_op": self.optimizer}
        epoch_loss = 0
        for step in range(triple_steps):
            pos_triples, neg_triples, pos_paths, neg_paths = batch_queue.get()
            feed_dict = {self.pos_hs: [x[0] for x in pos_triples],
                         self.pos_rs: [x[1] for x in pos_triples],
                         self.pos_ts: [x[2] for x in pos_triples],
                         self.neg_hs: [x[0] for x in neg_triples],
                         self.neg_rs: [x[1] for x in neg_triples],
                         self.neg_ts: [x[2] for x in neg_triples],
                         self.pos_rx: [x[0] for x in pos_paths],
                         self.pos_ry: [x[1] for x in pos_paths],
                         self.pos_r: [x[2] for x in pos_paths],
                         self.neg_rx: [x[0] for x in neg_paths],
                         self.neg_ry: [x[1] for x in neg_paths],
                         self.neg_r: [x[2] for x in neg_paths],
                         self.path_weight: [x[3] for x in pos_paths]}
            vals = self.session.run(fetches=fetches, feed_dict=feed_dict)
            epoch_loss += vals["loss"]
        epoch_loss /= self.args.batch_size
        random.shuffle(self.kgs.kg1.relation_triples_list)
        random.shuffle(self.kgs.kg2.relation_triples_list)
        print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def launch_alignment_training_1epo(self, epoch):
        t1 = time.time()
        sim_mat = self._ref_sim_mat()
        pairs = find_potential_alignment_greedily(sim_mat, self.args.sim_th)
        if pairs is None or len(pairs) == 0:
            return
        new_ent1 = [self.ref_entities1[pair[0]] for pair in pairs]
        new_ent2 = [self.ref_entities2[pair[1]] for pair in pairs]
        tr_ws = [sim_mat[pair[0], pair[1]] for pair in pairs]
        newly_triples = generate_triples_of_latent_ents(self.kgs, new_ent1, new_ent2, tr_ws)
        steps = math.ceil(((len(newly_triples)) / self.args.batch_size))
        if steps == 0:
            steps = 1
        alignment_loss = 0
        for step in range(steps):
            newly_pos_batch, newly_neg_batch = generate_triple_batch(newly_triples, self.args.batch_size,
                                                                     self.kgs.kg1.entities_list +
                                                                     self.kgs.kg2.entities_list)
            alignment_fetches = {"loss": self.alignment_loss, "train_op": self.alignment_optimizer}
            alignment_feed_dict = {self.new_ph: [tr[0] for tr in newly_pos_batch],
                                   self.new_pr: [tr[1] for tr in newly_pos_batch],
                                   self.new_pt: [tr[2] for tr in newly_pos_batch],
                                   self.new_nh: [tr[0] for tr in newly_neg_batch],
                                   self.new_nr: [tr[1] for tr in newly_neg_batch],
                                   self.new_nt: [tr[2] for tr in newly_neg_batch],
                                   self.tr_weight: [tr[3] for tr in newly_pos_batch]}
            alignment_vals = self.session.run(fetches=alignment_fetches, feed_dict=alignment_feed_dict)
            alignment_loss += alignment_vals["loss"]
        alignment_loss /= len(newly_triples)
        print('epoch {}, alignment loss: {:.4f}, cost time: {:.4f}s'.format(epoch, alignment_loss, time.time() - t1))

    def run(self):
        t = time.time()
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        for epoch in range(1, self.args.max_epoch):
            self.launch_ptranse_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue)
            if epoch >= self.args.start_valid and epoch % self.args.eval_freq == 0:
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or epoch == self.args.max_epoch:
                    break
            if epoch % self.args.bp_freq == 0:
                self.launch_alignment_training_1epo(epoch)
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
