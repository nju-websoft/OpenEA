import tensorflow as tf

from openea.models.trans.transe import TransE
from openea.modules.base.initializers import init_embeddings
from openea.modules.base.losses import margin_loss
from openea.modules.base.optimizers import generate_optimizer


class TransH(TransE):

    def __init__(self):
        super().__init__()

    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim],
                                              'ent_embeds', self.args.init, self.args.ent_l2_norm)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim],
                                              'rel_embeds', self.args.init, self.args.rel_l2_norm)
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
            self.triple_loss = margin_loss(phs, prs, pts, nhs, nrs, nts, self.args.margin, self.args.loss_norm)
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate, opt=self.args.optimizer)

    @staticmethod
    def _calc(e, n):
        norm = tf.nn.l2_normalize(n, 1)
        return e - tf.reduce_sum(e * norm, 1, keep_dims=True) * norm
