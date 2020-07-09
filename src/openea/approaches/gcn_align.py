import math
import multiprocessing as mp
import random
import time

import tensorflow as tf
import numpy as np

import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh

import openea.modules.load.read as rd
import openea.modules.train.batch as bat
from openea.modules.utils.util import load_session
from openea.modules.finding.evaluation import valid, test, early_stop
from openea.models.basic_model import BasicModel
from openea.modules.base.optimizers import generate_optimizer
from openea.modules.utils.util import merge_dic

'''
Refactoring based on https://github.com/1049451037/GCN-Align
'''
_LAYER_UIDS = {}


# ******************************inits************************
def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def trunc_normal(shape, name=None, normalize=True):
    initial = tf.Variable(tf.truncated_normal(shape, stddev=1.0 / math.sqrt(shape[0])))
    if not normalize:
        return initial
    return tf.nn.l2_normalize(initial, 1)


# *******************************layers**************************
def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    print(x)
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def load_attr(ent_num, kgs):
    cnt = {}
    entity_attributes_dict = merge_dic(kgs.kg1.entity_attributes_dict, kgs.kg2.entity_attributes_dict)
    for _, vs in entity_attributes_dict.items():
        for v in vs:
            if v not in cnt:
                cnt[v] = 1
            else:
                cnt[v] += 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    print(fre)
    attr2id = {}
    num = int(0.7 * len(cnt))
    for i in range(num):
        attr2id[fre[i][0]] = i
    attr = np.zeros((ent_num, num), dtype=np.float32)
    for ent, vs in entity_attributes_dict.items():
        for v in vs:
            if v in attr2id:
                attr[ent][attr2id[v]] = 1.0
    return attr


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer. (featureless=True and transform=False) is not supported for now."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, transform=True, init=glorot, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.transform = transform

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                if input_dim == output_dim and not self.transform and not featureless:
                    continue
                self.vars['weights_' + str(i)] = init([input_dim, output_dim],
                                                      name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.dropout:
            if self.sparse_inputs:
                x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
            else:
                x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if 'weights_' + str(i) in self.vars:
                if not self.featureless:
                    pre_sup = dot(x, self.vars['weights_' + str(i)], sparse=self.sparse_inputs)
                else:
                    pre_sup = self.vars['weights_' + str(i)]
            else:
                pre_sup = x
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


# *******************************************************************
# ****************************metrics***********************************
def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def get_placeholder_by_name(name):
    try:
        return tf.get_default_graph().get_tensor_by_name(name + ":0")
    except:
        return tf.placeholder(tf.int32, name=name)


def align_loss(outlayer, ILL, gamma, k):
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
    neg_left = get_placeholder_by_name("neg_left")  # tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right = get_placeholder_by_name("neg_right")  # tf.placeholder(tf.int32, [t * k], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    D = A + gamma
    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
    neg_left = get_placeholder_by_name("neg2_left")  # tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg_right = get_placeholder_by_name("neg2_right")  # tf.placeholder(tf.int32, [t * k], "neg2_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
    return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)


# ***************************models****************************************
class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, args, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.args = args
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.args.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=self.args.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=self.args.sparse_inputs,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=self.args.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, args, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.args = args
        # *************add***************

        # ************************************
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.args.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.args.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.args.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN_Align_Unit(Model):
    def __init__(self, args, placeholders, input_dim, output_dim, ILL, sparse_inputs=False, featureless=True, **kwargs):
        super(GCN_Align_Unit, self).__init__(**kwargs)
        self.args = args

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.placeholders = placeholders
        self.ILL = ILL
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.args.learning_rate)
        self.build()


    def _loss(self):
        self.loss += align_loss(self.outputs, self.ILL, self.args.gamma, self.args.neg_triple_num)

    def _accuracy(self):
        pass

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            featureless=self.featureless,
                                            sparse_inputs=self.sparse_inputs,
                                            transform=False,
                                            init=trunc_normal,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.output_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=False,
                                            transform=False,
                                            logging=self.logging))


class GCN_Utils:
    def __init__(self, args, kgs):
        self.args = args
        self.kgs = kgs

    @staticmethod
    def sparse_to_tuple(sparse_mx):
        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx

    @staticmethod
    def normalize_adj(adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def preprocess_adj(self, adj):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        adj_normalized = self.normalize_adj(adj + sp.eye(adj.shape[0]))
        return self.sparse_to_tuple(adj_normalized)

    @staticmethod
    def construct_feed_dict(features, support, placeholders):
        """Construct feed dictionary for GCN-Align."""
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
        return feed_dict

    def chebyshev_polynomials(self, adj, k):
        """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
        print("Calculating Chebyshev polynomials up to order {}...".format(k))

        adj_normalized = self.normalize_adj(adj)
        laplacian = sp.eye(adj.shape[0]) - adj_normalized
        largest_eigval, _ = eigsh(laplacian, 1, which='LM')
        scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

        t_k = list()
        t_k.append(sp.eye(adj.shape[0]))
        t_k.append(scaled_laplacian)

        def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
            s_lap = sp.csr_matrix(scaled_lap, copy=True)
            return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

        for i in range(2, k + 1):
            t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

        return self.sparse_to_tuple(t_k)

    @staticmethod
    def func(triples):
        head = {}
        cnt = {}
        for tri in triples:
            if tri[1] not in cnt:
                cnt[tri[1]] = 1
                head[tri[1]] = {tri[0]}
            else:
                cnt[tri[1]] += 1
                head[tri[1]].add(tri[0])
        r2f = {}
        for r in cnt:
            r2f[r] = len(head[r]) / cnt[r]
        return r2f

    @staticmethod
    def ifunc(triples):
        tail = {}
        cnt = {}
        for tri in triples:
            if tri[1] not in cnt:
                cnt[tri[1]] = 1
                tail[tri[1]] = {tri[2]}
            else:
                cnt[tri[1]] += 1
                tail[tri[1]].add(tri[2])
        r2if = {}
        for r in cnt:
            r2if[r] = len(tail[r]) / cnt[r]
        return r2if

    def get_weighted_adj(self, e, KG):
        r2f = self.func(KG)
        r2if = self.ifunc(KG)
        M = {}
        for tri in KG:
            if tri[0] == tri[2]:
                continue
            if (tri[0], tri[2]) not in M:
                M[(tri[0], tri[2])] = max(r2if[tri[1]], 0.3)
            else:
                M[(tri[0], tri[2])] += max(r2if[tri[1]], 0.3)
            if (tri[2], tri[0]) not in M:
                M[(tri[2], tri[0])] = max(r2f[tri[1]], 0.3)
            else:
                M[(tri[2], tri[0])] += max(r2f[tri[1]], 0.3)
        row = []
        col = []
        data = []
        for key in M:
            row.append(key[1])
            col.append(key[0])
            data.append(M[key])
        return sp.coo_matrix((data, (row, col)), shape=(e, e))

    def get_ae_input(self, attr):
        return self.sparse_to_tuple(sp.coo_matrix(attr))

    def load_data(self, attr):
        ae_input = self.get_ae_input(attr)
        triples = self.kgs.kg1.relation_triples_list + self.kgs.kg2.relation_triples_list
        adj = self.get_weighted_adj(self.kgs.entities_num, triples)
        train = np.array(self.kgs.train_links)
        return adj, ae_input, train


class GCN_Align(BasicModel):
    def __init__(self):
        super().__init__()
        self.attr = None
        self.opt = 'SGD'
        self.act_func = tf.nn.relu
        self.dropout = 0.0
        # *****************************add*******************************************************
        self.struct_loss = None
        self.struct_optimizer = None
        self.vec_ae = None
        self.vec_se = None
        self.num_supports = None
        self.utils = None
        self.adj = None
        self.ae_input = None
        self.train = None
        self.e = None
        self.support = None
        self.adj = None
        self.ph_ae = None
        self.ph_se = None
        self.model_ae = None
        self.model_se = None
        self.feed_dict_se = None
        self.feed_dict_ae = None

    def init(self):
        assert self.args.alignment_module == 'mapping'
        assert self.args.neg_triple_num > 1
        assert self.args.learning_rate >= 0.01

        self.num_supports = self.args.support_number
        self.utils = GCN_Utils(self.args, self.kgs)
        self.attr = load_attr(self.kgs.entities_num, self.kgs)
        self.adj, self.ae_input, self.train = self.utils.load_data(self.attr)
        self.e = self.ae_input[2][0]
        self.support = [self.utils.preprocess_adj(self.adj)]
        self.ph_ae = {
            "support": [tf.sparse_placeholder(tf.float32) for _ in range(self.args.support_number)],
            "features": tf.sparse_placeholder(tf.float32),
            "dropout": tf.placeholder_with_default(0., shape=()),
            "num_features_nonzero": tf.placeholder_with_default(0, shape=())
        }
        self.ph_se = {
            "support": [tf.sparse_placeholder(tf.float32) for _ in range(self.args.support_number)],
            "features": tf.placeholder(tf.float32),
            "dropout": tf.placeholder_with_default(0., shape=()),
            "num_features_nonzero": tf.placeholder_with_default(0, shape=())
        }
        self.model_ae = GCN_Align_Unit(self.args, self.ph_ae, input_dim=self.ae_input[2][1],
                                       output_dim=self.args.ae_dim, ILL=self.train,
                                       sparse_inputs=True, featureless=False, logging=False)
        self.model_se = GCN_Align_Unit(self.args, self.ph_se, input_dim=self.e, output_dim=self.args.se_dim,
                                       ILL=self.train, sparse_inputs=False,
                                       featureless=True, logging=False)

        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)

    def train_embeddings(self, loss, optimizer, output):
        # **t=train_number k=neg_num
        neg_num = self.args.neg_triple_num
        train_num = len(self.kgs.train_links)
        train_links = np.array(self.kgs.train_links)
        pos = np.ones((train_num, neg_num)) * (train_links[:, 0].reshape((train_num, 1)))
        neg_left = pos.reshape((train_num * neg_num,))
        pos = np.ones((train_num, neg_num)) * (train_links[:, 1].reshape((train_num, 1)))
        neg2_right = pos.reshape((train_num * neg_num,))
        neg2_left = None
        neg_right = None
        feed_dict_se = None
        feed_dict_ae = None

        for i in range(1, self.args.max_epoch + 1):
            start = time.time()
            if i % 10 == 1:
                neg2_left = np.random.choice(self.e, train_num * neg_num)
                neg_right = np.random.choice(self.e, train_num * neg_num)
            feed_dict_ae = self.utils.construct_feed_dict(self.ae_input, self.support, self.ph_ae)
            feed_dict_ae.update({self.ph_ae['dropout']: self.args.dropout})
            feed_dict_ae.update({'neg_left:0': neg_left, 'neg_right:0': neg_right,
                                 'neg2_left:0': neg2_left, 'neg2_right:0': neg2_right})
            feed_dict_se = self.utils.construct_feed_dict(1., self.support, self.ph_se)
            feed_dict_se.update({self.ph_se['dropout']: self.args.dropout})
            feed_dict_se.update({'neg_left:0': neg_left, 'neg_right:0': neg_right,
                                 'neg2_left:0': neg2_left, 'neg2_right:0': neg2_right})
            batch_loss1, _ = self.session.run(fetches=[self.model_ae.loss, self.model_ae.opt_op],
                                              feed_dict=feed_dict_ae)
            batch_loss2, _ = self.session.run(fetches=[self.model_se.loss, self.model_se.opt_op],
                                              feed_dict=feed_dict_se)

            batch_loss = batch_loss1 + batch_loss2
            print('epoch {}, avg. relation triple loss: {:.4f}, cost time: {:.4f}s'.format(i, batch_loss,
                                                                                           time.time() - start))

            # ********************no early stop********************************************
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                self.feed_dict_se = feed_dict_se
                self.feed_dict_ae = feed_dict_ae
                flag = self.valid_(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break
        vec_se = self.session.run(output, feed_dict=feed_dict_se)
        vec_ae = self.session.run(self.model_ae.outputs, feed_dict=feed_dict_ae)
        self.vec_se = vec_se
        self.vec_ae = vec_ae
        return vec_se, vec_ae

    def test(self, save=True):
        if self.args.test_method == "sa":
            beta = self.args.beta
            embeddings = np.concatenate([self.vec_se * beta, self.vec_ae * (1.0 - beta)], axis=1)
        else:
            embeddings = self.vec_se
        embeds1 = np.array([embeddings[e] for e in self.kgs.test_entities1])
        embeds2 = np.array([embeddings[e] for e in self.kgs.test_entities2])
        rest_12, _, _ = test(embeds1, embeds2, None, self.args.top_k, self.args.test_threads_num,
                             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        test(embeds1, embeds2, None, self.args.top_k, self.args.test_threads_num,
             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=self.args.csls, accurate=True)
        if save:
            ent_ids_rest_12 = [(self.kgs.test_entities1[i], self.kgs.test_entities2[j]) for i, j in rest_12]
            rd.save_results(self.out_folder, ent_ids_rest_12)

    def save(self):
        ent_embeds = self.vec_se
        attr_embeds = self.vec_ae
        rd.save_embeddings(self.out_folder, self.kgs, ent_embeds, None, attr_embeds, mapping_mat=None)

    def valid_(self, stop_metric):
        se = self.session.run(self.model_se.outputs, feed_dict=self.feed_dict_se)
        if self.args.test_method == "sa":
            ae = self.session.run(self.model_ae.outputs, feed_dict=self.feed_dict_ae)
            beta = self.args.beta
            embeddings = np.concatenate([se*beta, ae*(1.0-beta)], axis=1)
        else:
            embeddings = se
        embeds1 = np.array([embeddings[e] for e in self.kgs.valid_entities1])
        embeds2 = np.array([embeddings[e] for e in self.kgs.valid_entities2 + self.kgs.test_entities2])
        hits1_12, mrr_12 = valid(embeds1, embeds2, None, self.args.top_k, self.args.test_threads_num,
                                 metric=self.args.eval_metric)
        if stop_metric == 'hits1':
            return hits1_12
        return mrr_12

    def run(self):
        t = time.time()
        self.train_embeddings(self.struct_loss, self.struct_optimizer, self.model_se.outputs)
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
