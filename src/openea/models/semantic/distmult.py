import multiprocessing as mp
import random
import time

import tensorflow as tf

import openea.modules.train.batch as bat
from openea.models.basic_model import BasicModel
from openea.modules.base.initializers import init_embeddings
from openea.modules.base.optimizers import generate_optimizer
from openea.modules.utils.util import generate_out_folder
from openea.modules.utils.util import load_session


class DistMult(BasicModel):

    def set_kgs(self, kgs):
        self.kgs = kgs

    def set_args(self, args):
        self.args = args
        self.out_folder = generate_out_folder(self.args.output, self.args.training_data, self.args.dataset_division, self.__class__.__name__)

    def init(self):
        self._define_variables()
        self._define_mapping_variables()
        self._define_embed_graph()
        self._define_mapping_graph()
        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)

    def __init__(self):
        super().__init__()
        self.metric = 'inner'

    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'ent_embeds', self.args.init,
                                              self.args.ent_l2_norm)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds', self.args.init,
                                              self.args.rel_l2_norm)

    def _calc(self, h, t, r):
        return h * r * t

    def _define_embed_graph(self):
        with tf.name_scope('triple_placeholder'):
            self.hs = tf.placeholder(tf.int32, shape=[None])
            self.rs = tf.placeholder(tf.int32, shape=[None])
            self.ts = tf.placeholder(tf.int32, shape=[None])
            self.label = tf.placeholder(tf.float32, shape=[None])
        with tf.name_scope('triple_lookup'):
            phs = tf.nn.embedding_lookup(self.ent_embeds, self.hs)
            prs = tf.nn.embedding_lookup(self.rel_embeds, self.rs)
            pts = tf.nn.embedding_lookup(self.ent_embeds, self.ts)
        with tf.name_scope('triple_loss'):
            res = tf.reduce_sum(self._calc(phs, pts, prs), 1, keep_dims=False)
            self.triple_loss = tf.reduce_mean(tf.nn.softplus(- self.label * res))
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate, opt='Adagrad')

    def launch_triple_training_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1,
                                    neighbors2):
        start = time.time()
        for steps_task in steps_tasks:
            mp.Process(target=bat.generate_triple_label_batch,
                       args=(self.kgs.kg1.relation_triples_list, self.kgs.kg2.relation_triples_list,
                             self.kgs.kg1.relation_triples_set, self.kgs.kg2.relation_triples_set,
                             self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                             self.args.batch_size, steps_task,
                             batch_queue, neighbors1, neighbors2,
                             self.args.neg_triple_num)).start()
        epoch_loss = 0
        trained_pos_triples = 0
        for i in range(triple_steps):
            batch, label = batch_queue.get()
            batch_loss, _ = self.session.run(fetches=[self.triple_loss, self.triple_optimizer],
                                             feed_dict={self.hs: [x[0] for x in batch],
                                                        self.rs: [x[1] for x in batch],
                                                        self.ts: [x[2] for x in batch],
                                                        self.label: label})
            trained_pos_triples += len(batch)
            epoch_loss += batch_loss
        # print(trained_pos_triples)
        # epoch_loss /= trained_pos_triples
        random.shuffle(self.kgs.kg1.relation_triples_list)
        random.shuffle(self.kgs.kg2.relation_triples_list)
        print('epoch {}, triple loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))
