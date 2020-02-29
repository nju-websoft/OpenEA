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
from openea.modules.base.optimizers import generate_optimizer
from openea.modules.load.kg import KG
from openea.modules.utils.util import load_session
from openea.modules.base.losses import limited_loss
from openea.approaches.bootea import generate_supervised_triples, generate_pos_batch, bootstrapping, \
    calculate_likelihood_mat
from openea.modules.base.initializers import init_embeddings
from openea.models.basic_model import BasicModel


class BootEA_TransH(BasicModel):

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

    def _calc(self, e, n):
        norm = tf.nn.l2_normalize(n, 1)
        return e - tf.reduce_sum(e * norm, 1, keep_dims=True) * norm

    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'ent_embeds',
                                              self.args.init, self.args.ent_l2_norm)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds',
                                              self.args.init, self.args.rel_l2_norm)
            self.normal_vector = init_embeddings([self.kgs.relations_num, self.args.dim],
                                                 'normal_vector', self.args.init, True)

    def _define_embed_graph(self):
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
            pos_norm_vec = tf.nn.embedding_lookup(self.normal_vector, self.pos_rs)
            neg_norm_vec = tf.nn.embedding_lookup(self.normal_vector, self.neg_rs)
            phs = self._calc(phs, pos_norm_vec)
            pts = self._calc(pts, pos_norm_vec)
            nhs = self._calc(nhs, neg_norm_vec)
            nts = self._calc(nts, neg_norm_vec)
        with tf.name_scope('triple_loss'):
            self.triple_loss = limited_loss(phs, prs, pts, nhs, nrs, nts,
                                            self.args.pos_margin, self.args.neg_margin,
                                            self.args.loss_norm, balance=self.args.neg_margin_balance)
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate, opt=self.args.optimizer)

    def _define_alignment_graph(self):
        self.new_h = tf.placeholder(tf.int32, shape=[None])
        self.new_r = tf.placeholder(tf.int32, shape=[None])
        self.new_t = tf.placeholder(tf.int32, shape=[None])
        phs = tf.nn.embedding_lookup(self.ent_embeds, self.new_h)
        prs = tf.nn.embedding_lookup(self.rel_embeds, self.new_r)
        pts = tf.nn.embedding_lookup(self.ent_embeds, self.new_t)
        self.alignment_loss = - tf.reduce_sum(tf.log(tf.sigmoid(-tf.reduce_sum(tf.pow(phs + prs - pts, 2), 1))))
        self.alignment_optimizer = generate_optimizer(self.alignment_loss, self.args.learning_rate, opt=self.args.optimizer)

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
        self.likelihood_optimizer = generate_optimizer(self.likelihood_loss, self.args.learning_rate, opt=self.args.optimizer)

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
            self.launch_training_k_epo(i, sub_num, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2)
            if i * sub_num >= self.args.start_valid:
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == iter_nums:
                    break
            labeled_align, entities1, entities2 = bootstrapping(self.eval_ref_sim_mat(),
                                                                self.ref_ent1, self.ref_ent2, labeled_align,
                                                                self.args.sim_th, self.args.k)
            self.train_alignment(self.kgs.kg1, self.kgs.kg2, entities1, entities2, 1)
            if i * sub_num >= self.args.start_valid:
                self.valid(self.args.stop_metric)
            t1 = time.time()
            assert 0.0 < self.args.truncated_epsilon < 1.0
            neighbors_num1 = int((1 - self.args.truncated_epsilon) * self.kgs.kg1.entities_num)
            neighbors_num2 = int((1 - self.args.truncated_epsilon) * self.kgs.kg2.entities_num)
            if neighbors1 is not None:
                del neighbors1, neighbors2
            gc.collect()
            neighbors1 = bat.generate_neighbours(self.eval_kg1_useful_ent_embeddings(),
                                                 self.kgs.useful_entities_list1,
                                                 neighbors_num1, self.args.batch_threads_num)
            neighbors2 = bat.generate_neighbours(self.eval_kg2_useful_ent_embeddings(),
                                                 self.kgs.useful_entities_list2,
                                                 neighbors_num2, self.args.batch_threads_num)
            ent_num = len(self.kgs.kg1.entities_list) + len(self.kgs.kg2.entities_list)
            print("generating neighbors of {} entities costs {:.3f} s.".format(ent_num, time.time() - t1))
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
