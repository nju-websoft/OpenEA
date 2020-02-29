import tensorflow as tf

from openea.models.trans.transe import TransE
from openea.modules.base.initializers import init_embeddings
from openea.modules.base.losses import get_loss_func
from openea.modules.base.optimizers import generate_optimizer


class TransR(TransE):

    def __init__(self):
        super().__init__()

    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim],
                                              'ent_embeds', self.args.init, self.args.ent_l2_norm)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim],
                                              'rel_embeds', self.args.init, self.args.rel_l2_norm)
            self.rel_matrix = init_embeddings([self.kgs.relations_num, self.args.dim * self.args.dim],
                                              'rel_matrix', self.args.init, False)

    def _define_embed_graph(self):
        with tf.name_scope('triple_placeholder'):
            self.pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.pos_ts = tf.placeholder(tf.int32, shape=[None])
            self.neg_hs = tf.placeholder(tf.int32, shape=[None])
            self.neg_rs = tf.placeholder(tf.int32, shape=[None])
            self.neg_ts = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('triple_lookup'):
            phs = tf.reshape(tf.nn.embedding_lookup(self.ent_embeds, self.pos_hs), [-1, self.args.dim, 1])
            prs = tf.nn.embedding_lookup(self.rel_embeds, self.pos_rs)
            pts = tf.reshape(tf.nn.embedding_lookup(self.ent_embeds, self.pos_ts), [-1, self.args.dim, 1])
            nhs = tf.reshape(tf.nn.embedding_lookup(self.ent_embeds, self.neg_hs), [-1, self.args.dim, 1])
            nrs = tf.nn.embedding_lookup(self.rel_embeds, self.neg_rs)
            nts = tf.reshape(tf.nn.embedding_lookup(self.ent_embeds, self.neg_ts), [-1, self.args.dim, 1])
            p_matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, self.pos_rs),
                                  [-1, self.args.dim, self.args.dim])
            n_matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, self.neg_rs),
                                  [-1, self.args.dim, self.args.dim])
            phs = tf.reshape(tf.matmul(p_matrix, phs), [-1, self.args.dim])
            pts = tf.reshape(tf.matmul(p_matrix, pts), [-1, self.args.dim])
            phs = tf.nn.l2_normalize(phs, 1)
            pts = tf.nn.l2_normalize(pts, 1)
            nhs = tf.reshape(tf.matmul(n_matrix, nhs), [-1, self.args.dim])
            nts = tf.reshape(tf.matmul(n_matrix, nts), [-1, self.args.dim])
            nhs = tf.nn.l2_normalize(nhs, 1)
            nts = tf.nn.l2_normalize(nts, 1)
        with tf.name_scope('triple_loss'):
            self.triple_loss = get_loss_func(phs, prs, pts, nhs, nrs, nts, self.args)
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)
