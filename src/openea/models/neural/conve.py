import math

import tensorflow as tf

from openea.models.neural.proje import ProjE
from openea.modules.base.optimizers import generate_optimizer
from openea.modules.utils.util import load_session


def dim_factorization(d):
    half = int(math.sqrt(d)) + 1
    while d % half > 0:
        half -= 1
    x = half
    y = d // half
    assert x * y == d
    print("dim factorization", x, y)
    return x, y


class ConvE(ProjE):

    def __init__(self):
        super().__init__()
        self.kernel_size = (3, 3)
        print("kernel_size", self.kernel_size)

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

    def _define_embed_graph(self):
        with tf.name_scope('triple_placeholder'):
            self.pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.pos_ts = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('triple_lookup'):
            x, y = dim_factorization(self.args.dim)
            phs = tf.reshape(tf.nn.embedding_lookup(self.ent_embeds, self.pos_hs), [-1, 1, x, y])
            prs = tf.reshape(tf.nn.embedding_lookup(self.rel_embeds, self.pos_rs), [-1, 1, x, y])

        stacked_inputs = tf.concat([phs, prs], 2)
        stacked_inputs = tf.layers.batch_normalization(stacked_inputs)
        stacked_inputs = tf.nn.dropout(stacked_inputs, self.args.output_keep_prob)
        with tf.variable_scope('cnn'):
            ocnn = tf.layers.conv2d(stacked_inputs, self.args.filter_num, self.kernel_size,
                                    padding='same', use_bias=True, data_format='channels_first')

        ocnn = tf.layers.batch_normalization(ocnn, axis=1)
        ocnn = tf.nn.relu(ocnn)
        ocnn = tf.nn.dropout(ocnn, self.args.output_keep_prob)
        ocnn = tf.reshape(ocnn, [-1, self.args.filter_num * self.args.dim * 2])
        ocnn = tf.contrib.layers.fully_connected(ocnn, self.args.dim)
        ocnn = tf.layers.batch_normalization(ocnn)

        with tf.name_scope('triple_loss'):
            triple_loss = tf.nn.nce_loss(
                weights=self.entity_w,
                biases=self.entity_b,
                labels=tf.reshape(self.pos_ts, [-1, 1]),
                inputs=ocnn,
                num_sampled=self.args.dnn_neg_nums,
                num_classes=self.kgs.entities_num,
                partition_strategy='div',
            )

            self.triple_loss = tf.reduce_sum(triple_loss)
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)
