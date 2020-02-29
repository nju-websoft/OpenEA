import tensorflow as tf
import math
import multiprocessing as mp
import time
import random
import numpy as np

from openea.modules.finding.evaluation import early_stop
from openea.modules.utils.util import task_divide
from openea.modules.base.optimizers import generate_optimizer
from openea.modules.utils.util import load_session
from openea.modules.base.initializers import init_embeddings
from openea.models.basic_model import BasicModel
from openea.approaches.attr2vec import Attr2Vec
import openea.modules.train.batch as bat


class JAPE(BasicModel):

    def __init__(self):
        super().__init__()
        self.attr2vec = Attr2Vec()
        self.attr_sim_mat = None
        self.ref_entities1, self.ref_entities2 = None, None

    def init(self):
        self.ref_entities1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        self.ref_entities2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        self._define_variables()
        self._define_embed_graph()
        self._define_sim_graph()
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

        assert self.args.neg_triple_num >= 1
        assert self.args.neg_alpha >= 0.0
        assert self.args.top_attr_threshold > 0.0
        assert self.args.attr_sim_mat_threshold > 0.0
        assert self.args.attr_sim_mat_beta > 0.0

    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'ent_embeds',
                                              self.args.init, self.args.ent_l2_norm)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds',
                                              self.args.init, self.args.rel_l2_norm)

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
        with tf.name_scope('triple_loss'):
            with tf.name_scope('jape_loss_distance'):
                pos_distance = phs + prs - pts
                neg_distance = nhs + nrs - nts
            with tf.name_scope('jape_loss_score'):
                pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
                neg_score = tf.reduce_sum(tf.square(neg_distance), axis=1)
            pos_loss = tf.reduce_sum(pos_score)
            neg_loss = tf.reduce_sum(neg_score)
            self.triple_loss = pos_loss - self.args.neg_alpha * neg_loss
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate, opt=self.args.optimizer)

    def _define_sim_graph(self):
        self.entities1 = tf.placeholder(tf.int32, shape=[None])
        dim1 = self.args.sub_mat_size
        dim2 = len(self.ref_entities2)
        self.attr_sim_mat_place = tf.placeholder(tf.float32, shape=[dim1, dim2])
        ref1 = tf.nn.embedding_lookup(self.ent_embeds, self.entities1)
        ref2 = tf.nn.embedding_lookup(self.ent_embeds, self.ref_entities2)
        ref2_trans = tf.matmul(self.attr_sim_mat_place, ref2)
        ref2_trans = tf.nn.l2_normalize(ref2_trans, 1)
        self.sim_loss = self.args.attr_sim_mat_beta * tf.reduce_sum(tf.reduce_sum(tf.pow(ref1 - ref2_trans, 2), 1))
        opt_vars = [v for v in tf.trainable_variables() if v.name.startswith("relational")]
        self.sim_optimizer = generate_optimizer(self.sim_loss, self.args.learning_rate, var_list=opt_vars,
                                                opt=self.args.optimizer)

    def launch_triple_training_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1, neighbors2):
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
        print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def launch_sim_1epo(self, epoch):
        t = time.time()
        steps = len(self.ref_entities1) // self.args.sub_mat_size
        ref_ent1_array = np.array(self.ref_entities1)
        ll = list(range(len(self.ref_entities1)))
        loss = 0
        for i in range(steps):
            idx = random.sample(ll, self.args.sub_mat_size)
            feed_dict = {self.entities1: ref_ent1_array[idx], self.attr_sim_mat_place: self.attr_sim_mat[idx, :]}
            vals = self.session.run(fetches=self.sim_loss, feed_dict=feed_dict)
            loss += vals
        print('epoch {}, sim loss: {:.4f}, cost time: {:.4f}s'.format(epoch, loss, time.time() - t))

    def run_attr2vec(self):
        t = time.time()
        print("Training attribute embeddings:")
        self.attr2vec.set_args(self.args)
        self.attr2vec.set_kgs(self.kgs)
        self.attr2vec.init()
        self.attr2vec.run()
        sim_mat = self.attr2vec.eval_sim_mat()
        sim_mat[sim_mat < self.args.attr_sim_mat_threshold] = 0
        self.attr_sim_mat = sim_mat
        print("Training attributes ends. Total time = {:.3f} s.".format(time.time() - t))

    def run(self):
        self.run_attr2vec()
        print("Joint training:")
        t = time.time()
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        for i in range(1, self.args.max_epoch + 1):
            self.launch_triple_training_1epo(i, triple_steps, steps_tasks, training_batch_queue, None, None)
            self.launch_sim_1epo(i)
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
