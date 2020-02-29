import tensorflow as tf

from openea.models.trans.transe import TransE
from openea.modules.base.initializers import init_embeddings
from openea.modules.base.losses import get_loss_func
from openea.modules.base.optimizers import generate_optimizer


class TransD(TransE):

    def __init__(self):
        super().__init__()

    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim],
                                              'ent_embeds', self.args.init, self.args.ent_l2_norm)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim],
                                              'rel_embeds', self.args.init, self.args.rel_l2_norm)
            self.ent_transfer = init_embeddings([self.kgs.entities_num, self.args.dim],
                                                'ent_transfer', self.args.init, self.args.ent_l2_norm)
            self.rel_transfer = init_embeddings([self.kgs.relations_num, self.args.dim],
                                                'rel_transfer', self.args.init, self.args.rel_l2_norm)

    def _define_embed_graph(self):
        with tf.name_scope('triple_placeholder'):
            self.pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.pos_ts = tf.placeholder(tf.int32, shape=[None])
            self.neg_hs = tf.placeholder(tf.int32, shape=[None])
            self.neg_rs = tf.placeholder(tf.int32, shape=[None])
            self.neg_ts = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('triple_lookup'):
            phe = tf.nn.embedding_lookup(self.ent_embeds, self.pos_hs)
            pte = tf.nn.embedding_lookup(self.ent_embeds, self.pos_ts)
            pre = tf.nn.embedding_lookup(self.rel_embeds, self.pos_rs)
            pht = tf.nn.embedding_lookup(self.ent_transfer, self.pos_hs)
            ptt = tf.nn.embedding_lookup(self.ent_transfer, self.pos_ts)
            prt = tf.nn.embedding_lookup(self.rel_transfer, self.pos_rs)
            nhe = tf.nn.embedding_lookup(self.ent_embeds, self.neg_hs)
            nte = tf.nn.embedding_lookup(self.ent_embeds, self.neg_ts)
            nre = tf.nn.embedding_lookup(self.rel_embeds, self.neg_rs)
            nht = tf.nn.embedding_lookup(self.ent_transfer, self.neg_hs)
            ntt = tf.nn.embedding_lookup(self.ent_transfer, self.neg_ts)
            nrt = tf.nn.embedding_lookup(self.rel_transfer, self.neg_rs)
        with tf.name_scope('projection'):
            phe = self._calc(phe, pht, prt)
            pte = self._calc(pte, ptt, prt)
            nhe = self._calc(nhe, nht, nrt)
            nte = self._calc(nte, ntt, nrt)
        with tf.name_scope('triple_loss'):
            self.triple_loss = get_loss_func(phe, pre, pte, nhe, nre, nte, self.args)
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)

    def _calc(self, e, t, r):
        return tf.nn.l2_normalize(e + tf.reduce_sum(e * t, 1, keep_dims=True) * r, 1)
