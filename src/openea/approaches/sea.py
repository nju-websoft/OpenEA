import tensorflow as tf
import math
import multiprocessing as mp
import numpy as np
import random
import time
from sklearn import preprocessing

from openea.modules.finding.evaluation import early_stop
from openea.modules.utils.util import task_divide
from openea.modules.base.optimizers import generate_optimizer
from openea.modules.utils.util import load_session
from openea.modules.base.initializers import init_embeddings, orthogonal_init
from openea.modules.base.losses import get_loss_func
from openea.models.basic_model import BasicModel
import openea.modules.load.read as rd


class SEA(BasicModel):

    def __init__(self):
        super().__init__()

    def init(self):
        self._define_variables()
        self._define_embed_graph()
        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)

        # customize parameters
        assert self.args.loss == 'margin-based'
        assert self.args.alignment_module == 'mapping'
        assert self.args.loss == 'margin-based'
        assert self.args.neg_sampling == 'uniform'
        assert self.args.optimizer == 'Adam'
        assert self.args.eval_metric == 'inner'
        assert self.args.loss_norm == 'L2'
        assert self.args.ent_l2_norm is True
        assert self.args.rel_l2_norm is True
        assert self.args.neg_triple_num == 1

    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'ent_embeds',
                                              self.args.init, self.args.ent_l2_norm)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds',
                                              self.args.init, self.args.rel_l2_norm)
        with tf.variable_scope('kgs' + 'mapping'):
            self.mapping_mat_1 = orthogonal_init([self.args.dim, self.args.dim], 'mapping_matrix_1')
            self.eye_mat_1 = tf.constant(np.eye(self.args.dim), dtype=tf.float32, name='eye_1')
            self.mapping_mat_2 = orthogonal_init([self.args.dim, self.args.dim], 'mapping_matrix_2')
            self.eye_mat_2 = tf.constant(np.eye(self.args.dim), dtype=tf.float32, name='eye_2')

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
            self.triple_loss = get_loss_func(phs, prs, pts, nhs, nrs, nts, self.args)
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)
        with tf.name_scope('seed_links_placeholder'):
            self.labeled_entities1 = tf.placeholder(tf.int32, shape=[None])
            self.labeled_entities2 = tf.placeholder(tf.int32, shape=[None])
            self.unlabeled_entities1 = tf.placeholder(tf.int32, shape=[None])
            self.unlabeled_entities2 = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('seed_links_lookup'):
            labeled_embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.labeled_entities1)
            labeled_embeds2 = tf.nn.embedding_lookup(self.ent_embeds, self.labeled_entities2)
            unlabeled_embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.unlabeled_entities1)
            unlabeled_embeds2 = tf.nn.embedding_lookup(self.ent_embeds, self.unlabeled_entities2)
        with tf.name_scope('sup_mapping_loss'):
            mapped_12 = tf.nn.l2_normalize(tf.matmul(labeled_embeds1, self.mapping_mat_1))
            mapped_21 = tf.nn.l2_normalize(tf.matmul(labeled_embeds2, self.mapping_mat_2))
            map_loss_12 = tf.reduce_sum(tf.reduce_sum(tf.pow(labeled_embeds2 - mapped_12, 2), 1))
            map_loss_21 = tf.reduce_sum(tf.reduce_sum(tf.pow(labeled_embeds1 - mapped_21, 2), 1))
        with tf.name_scope('semi_sup_mapping_loss'):
            semi_mapped_121 = tf.nn.l2_normalize(tf.matmul(tf.matmul(unlabeled_embeds1, self.mapping_mat_1),
                                                           self.mapping_mat_2))
            semi_mapped_212 = tf.nn.l2_normalize(tf.matmul(tf.matmul(unlabeled_embeds2, self.mapping_mat_2),
                                                           self.mapping_mat_1))
            map_loss_11 = tf.reduce_sum(tf.reduce_sum(tf.pow(unlabeled_embeds1 - semi_mapped_121, 2), 1))
            map_loss_22 = tf.reduce_sum(tf.reduce_sum(tf.pow(unlabeled_embeds2 - semi_mapped_212, 2), 1))
            self.mapping_loss = self.args.alpha_1 * (map_loss_12 + map_loss_21) + \
                                self.args.alpha_2 * (map_loss_11 + map_loss_22)
            self.mapping_optimizer = generate_optimizer(self.mapping_loss, self.args.learning_rate,
                                                        opt=self.args.optimizer)

    def _eval_valid_embeddings(self):
        if len(self.kgs.valid_links) > 0:
            embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.valid_entities1).eval(session=self.session)
            embeds2 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.valid_entities2 + self.kgs.test_entities2).eval(
                session=self.session)
        else:
            embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities1).eval(session=self.session)
            embeds2 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities2).eval(session=self.session)
        mapping_mat_1 = self.mapping_mat_1.eval(session=self.session)
        return embeds1, embeds2, mapping_mat_1

    def _eval_test_embeddings(self):
        embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities1).eval(session=self.session)
        embeds2 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities2).eval(session=self.session)
        mapping_mat_1 = self.mapping_mat_1.eval(session=self.session)
        return embeds1, embeds2, mapping_mat_1

    def save(self):
        ent_embeds = self.ent_embeds.eval(session=self.session)
        rel_embeds = self.rel_embeds.eval(session=self.session)
        mapping_mat_1 = self.mapping_mat_1.eval(session=self.session)
        mapping_mat_2 = self.mapping_mat_2.eval(session=self.session)
        rd.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None,
                           mapping_mat=mapping_mat_1, rev_mapping_mat=mapping_mat_2)

    def launch_training_1epo(self, epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2):
        self.launch_triple_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2)
        self.launch_mapping_training_1epo(epoch, triple_steps)

    def launch_mapping_training_1epo(self, epoch, triple_steps):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            labeled_batch = random.sample(self.kgs.train_links, len(self.kgs.train_links) // triple_steps)
            unlabeled_batch = random.sample(self.kgs.test_links + self.kgs.valid_links,
                                            len(self.kgs.test_links + self.kgs.valid_links) // triple_steps)
            batch_loss, _ = self.session.run(fetches=[self.mapping_loss, self.mapping_optimizer],
                                             feed_dict={self.labeled_entities1: [x[0] for x in labeled_batch],
                                                        self.labeled_entities2: [x[1] for x in labeled_batch],
                                                        self.unlabeled_entities1: [x[0] for x in unlabeled_batch],
                                                        self.unlabeled_entities2: [x[1] for x in unlabeled_batch]})
            epoch_loss += batch_loss
            trained_samples_num += len(labeled_batch)
        epoch_loss /= trained_samples_num
        print('epoch {}, avg. mapping loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def run(self):
        t = time.time()
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        for i in range(1, self.args.max_epoch + 1):
            self.launch_training_1epo(i, triple_steps, steps_tasks, training_batch_queue, None, None)
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
