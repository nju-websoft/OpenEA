import numpy as np
import tensorflow as tf

from openea.modules.base.initializers import orthogonal_init
from openea.modules.base.losses import mapping_loss
from openea.modules.base.optimizers import generate_optimizer


def add_mapping_module(model):
    with tf.name_scope('seed_links_placeholder'):
        model.seed_entities1 = tf.placeholder(tf.int32, shape=[None])
        model.seed_entities2 = tf.placeholder(tf.int32, shape=[None])
    with tf.name_scope('seed_links_lookup'):
        tes1 = tf.nn.embedding_lookup(model.ent_embeds, model.seed_entities1)
        tes2 = tf.nn.embedding_lookup(model.ent_embeds, model.seed_entities2)
    with tf.name_scope('mapping_loss'):
        model.mapping_loss = model.args.alpha * mapping_loss(tes1, tes2, model.mapping_mat, model.eye_mat)
        model.mapping_optimizer = generate_optimizer(model.mapping_loss, model.args.learning_rate,
                                                     opt=model.args.optimizer)


def add_mapping_variables(model):
    with tf.variable_scope('kgs' + 'mapping'):
        model.mapping_mat = orthogonal_init([model.args.dim, model.args.dim], 'mapping_matrix')
        model.eye_mat = tf.constant(np.eye(model.args.dim), dtype=tf.float32, name='eye')
