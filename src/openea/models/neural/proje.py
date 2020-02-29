import math
import multiprocessing as mp
import random
import time

import tensorflow as tf

import openea.modules.train.batch as bat
from openea.models.trans.transe import BasicModel
from openea.modules.base.initializers import init_embeddings
from openea.modules.utils.util import load_session
from openea.modules.base.optimizers import generate_optimizer
from openea.modules.utils.util import task_divide
from openea.modules.finding.evaluation import early_stop


class ProjE(BasicModel):

    def __init__(self):
        super().__init__()

    def init(self):
        self._define_variables()
        self._define_embed_graph()
        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)

        assert self.args.init == 'xavier'
        assert self.args.alignment_module == 'sharing'
        assert self.args.optimizer == 'Adam'
        assert self.args.eval_metric == 'inner'
        assert self.args.ent_l2_norm is True
        assert self.args.rel_l2_norm is True
        assert self.args.dnn_neg_nums > 1

    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'ent_embeds',
                                              self.args.init, self.args.ent_l2_norm)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds',
                                              self.args.init, self.args.rel_l2_norm)
        with tf.variable_scope('prob' + 'parameters'):
            self.entity_w = init_embeddings([self.kgs.entities_num, self.args.dim], 'entity_w', 'xavier', False)
            self.entity_b = init_embeddings([self.kgs.entities_num, ], 'entity_b', 'xavier', False)

    def _define_embed_graph(self):
        with tf.name_scope('triple_placeholder'):
            self.pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.pos_ts = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('triple_lookup'):
            phs = tf.nn.embedding_lookup(self.ent_embeds, self.pos_hs)
            prs = tf.nn.embedding_lookup(self.rel_embeds, self.pos_rs)
        with tf.variable_scope('input_bn', reuse=tf.AUTO_REUSE):
            bn_phs = tf.contrib.layers.batch_norm(phs, scope='bn')
            bn_prs = tf.contrib.layers.batch_norm(prs, reuse=True, scope='bn')
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            out_prs = bn_phs * tf.get_variable('mlp_w', [self.args.dim]) + \
                      bn_prs * tf.get_variable('mlp_w', [self.args.dim]) + \
                      tf.get_variable('mlp_bias', [self.args.dim])
        with tf.variable_scope('output_bn', reuse=tf.AUTO_REUSE):
            bn_out_prs = tf.contrib.layers.batch_norm(out_prs, scope='bn')
        with tf.name_scope('triple_loss'):
            triple_loss = tf.nn.nce_loss(
                weights=self.entity_w,
                biases=self.entity_b,
                labels=tf.reshape(self.pos_ts, [-1, 1]),
                inputs=bn_out_prs,
                num_sampled=self.args.dnn_neg_nums,
                num_classes=self.kgs.entities_num,
                partition_strategy='div')
            self.triple_loss = tf.reduce_sum(triple_loss)
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)

    def launch_triple_training_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1,
                                    neighbors2):
        start = time.time()
        for steps_task in steps_tasks:
            mp.Process(target=bat.generate_pos_batch_queue,
                       args=(self.kgs.kg1.relation_triples_list, self.kgs.kg2.relation_triples_list,
                             self.args.batch_size, steps_task, batch_queue)).start()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            batch_pos = batch_queue.get()
            batch_loss, _ = self.session.run(fetches=[self.triple_loss, self.triple_optimizer],
                                             feed_dict={self.pos_hs: [x[0] for x in batch_pos],
                                                        self.pos_rs: [x[1] for x in batch_pos],
                                                        self.pos_ts: [x[2] for x in batch_pos]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        random.shuffle(self.kgs.kg1.relation_triples_list)
        random.shuffle(self.kgs.kg2.relation_triples_list)
        print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

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
