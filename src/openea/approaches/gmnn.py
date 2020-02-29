import time

import tensorflow as tf
import numpy as np

import multiprocessing
import codecs
import os
from openea.modules.utils.util import load_session
from openea.modules.finding.evaluation import valid, test, early_stop
from openea.models.basic_model import BasicModel
from tensorflow.python.ops import nn_ops
from scipy import spatial

'''
Refactoring based on https://github.com/syxu828/Crosslingula-KG-Matching
'''

_LAYER_UIDS = {}
eps = 1e-6


# *********************help gennerate train file***********

def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return [ls]
    if n > ls_len:
        return [ls]
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return


def find_near(all_links, id_1_vec, id_2_vec, zero_vec_id1, zero_vec_id2, cand_size):
    result_dict = dict()
    for item in all_links:
        id_1 = item[0]
        vec_1 = id_1_vec[id_1]

        if id_1 not in zero_vec_id1:
            cand_id_2_score_map = {}
            for id_2_cand in id_2_vec.keys():
                if id_2_cand in zero_vec_id2:
                    continue
                cos_sim = 1 - spatial.distance.cosine(vec_1, id_2_vec[id_2_cand])
                cand_id_2_score_map[id_2_cand] = cos_sim

            cand_id_2_score_items = cand_id_2_score_map.items()
            cand_id_2_score_items = sorted(cand_id_2_score_items, key=lambda d: d[1], reverse=True)
            result_list = []
            for idx in range(cand_size):
                result_list.append(cand_id_2_score_items[idx][0])
            result_dict[id_1] = result_list
    return result_dict


# *********************use to load embedding****************
def load_word_embedding(embedding_path, word_idx):
    with codecs.open(embedding_path, "r", "utf-8") as f:
        vecs = list()
        for line in f:
            line = line.strip()
            if len(line.split(" ")) == 2:
                continue
            info = line.split(' ')
            word = info[0]
            vec = [float(v) for v in info[1:]]
            if len(vec) != 300:
                continue
            vecs.append(vec)
            word_idx[word] = len(word_idx.keys()) + 1  # + 1 is due to that we already have an unknown word
    return np.array(vecs)


def write_word_idx(word_idx, path):
    dir = path[:path.rfind('/')]
    if not os.path.exists(dir):
        os.makedirs(dir)

    with codecs.open(path, 'w', 'utf-8') as f:
        for word in word_idx:
            f.write(str(word) + " " + str(word_idx[word]) + '\n')


def read_word_idx_from_file(path, if_key_is_int=False):
    word_idx = {}
    with codecs.open(path, 'r', 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            info = line.strip().split(" ")
            if len(info) != 2:
                word_idx[' '] = int(info[0])
            else:
                if if_key_is_int:
                    word_idx[int(info[0])] = int(info[1])
                else:
                    word_idx[info[0]] = int(info[1])
    return word_idx


def zeros(shape, name=None):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def cosine_distance(y1, y2):
    # y1 [....,a, 1, d]
    # y2 [....,1, b, d]
    cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps))
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps))
    return cosine_numerator / y1_norm / y2_norm


def cal_relevancy_matrix(node_1_repres, node_2_repres, watch=None):
    # [batch_size, 1, single_graph_1_nodes_size, node_embedding_dim]
    node_1_repres_tmp = tf.expand_dims(node_1_repres, 1)

    # [batch_size, single_graph_2_nodes_size, 1, node_embedding_dim]
    node_2_repres_tmp = tf.expand_dims(node_2_repres, 2)

    # [batch_size, single_graph_2_nodes_size, single_graph_1_nodes_size]
    relevancy_matrix = cosine_distance(node_1_repres_tmp, node_2_repres_tmp)

    watch["node_1_repres_tmp"] = node_1_repres
    watch["node_2_repres_tmp"] = node_2_repres
    watch["relevancy_matrix"] = relevancy_matrix

    return relevancy_matrix


def mask_relevancy_matrix(relevancy_matrix, graph_1_mask, graph_2_mask):
    # relevancy_matrix: [batch_size, single_graph_2_nodes_size, single_graph_1_nodes_size]
    # graph_1_mask: [batch_size, single_graph_1_nodes_size]
    # graph_2_mask: [batch_size, single_graph_2_nodes_size]
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(graph_1_mask, 1))
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(graph_2_mask, 2))

    # [batch_size, single_graph_2_nodes_size, single_graph_1_nodes_size]
    return relevancy_matrix


def multi_perspective_expand_for_2D(in_tensor, decompose_params):
    # [batch_size, 'x', dim]
    in_tensor = tf.expand_dims(in_tensor, axis=1)
    # [1, decompse_dim, dim]
    decompose_params = tf.expand_dims(decompose_params, axis=0)
    # [batch_size, decompse_dim, dim]
    return tf.multiply(in_tensor, decompose_params)


def cal_maxpooling_matching(node_1_rep, node_2_rep, decompose_params):
    # node_1_rep: [batch_size, single_graph_1_nodes_size, dim]
    # node_2_rep: [batch_size, single_graph_2_nodes_size, dim]
    # decompose_params: [decompose_dim, dim]
    def singel_instance(x):
        # p: [single_graph_1_nodes_size, dim], q: [single_graph_2_nodes_size, dim]
        p = x[0]
        q = x[1]

        # [single_graph_1_nodes_size, decompose_dim, dim]
        p = multi_perspective_expand_for_2D(p, decompose_params)

        # [single_graph_2_nodes_size, decompose_dim, dim]
        q = multi_perspective_expand_for_2D(q, decompose_params)

        # [single_graph_1_nodes_size, 1, decompose_dim, dim]
        p = tf.expand_dims(p, 1)

        # [1, single_graph_2_nodes_size, decompose_dim, dim]
        q = tf.expand_dims(q, 0)

        # [single_graph_1_nodes_size, single_graph_2_nodes_size, decompose]
        return cosine_distance(p, q)

    elems = (node_1_rep, node_2_rep)

    # [batch_size, single_graph_1_nodes_size, single_graph_2_nodes_size, decompse_dim]
    matching_matrix = tf.map_fn(singel_instance, elems, dtype=tf.float32)

    # [batch_size, single_graph_1_nodes_size, 2 * decompse_dim]
    return tf.concat(axis=2, values=[tf.reduce_max(matching_matrix, axis=2), tf.reduce_mean(matching_matrix, axis=2)])


def cal_max_node_2_representation(node_2_rep, relevancy_matrix):
    layer_utils = LayerUtils()
    # [batch_size, single_graph_1_nodes_size]
    atten_positions = tf.argmax(relevancy_matrix, axis=2, output_type=tf.int32)
    max_node_2_reps = layer_utils.collect_representation(node_2_rep, atten_positions)

    # [batch_size, single_graph_1_nodes_size, dim]
    return max_node_2_reps


def multi_perspective_match(feature_dim, rep_1, rep_2, options=None, scope_name='mp-match', reuse=False):
    """
        :param repres1: [batch_size, len, feature_dim]
        :param repres2: [batch_size, len, feature_dim]
        :return:
    """
    layer_utils = LayerUtils
    input_shape = tf.shape(rep_1)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    matching_result = []
    with tf.variable_scope(scope_name, reuse=reuse):
        match_dim = 0
        if options['with_cosine']:
            cosine_value = layer_utils.cosine_distance(rep_1, rep_2, cosine_norm=False)
            cosine_value = tf.reshape(cosine_value, [batch_size, seq_length, 1])
            matching_result.append(cosine_value)
            match_dim += 1

        if options['with_mp_cosine']:
            mp_cosine_params = tf.get_variable("mp_cosine", shape=[options['cosine_MP_dim'], feature_dim],
                                               dtype=tf.float32)
            mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
            mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
            rep_1_flat = tf.expand_dims(rep_1, axis=2)
            rep_2_flat = tf.expand_dims(rep_2, axis=2)
            mp_cosine_matching = layer_utils.cosine_distance(tf.multiply(rep_1_flat, mp_cosine_params),
                                                             rep_2_flat, cosine_norm=False)
            matching_result.append(mp_cosine_matching)
            match_dim += options['cosine_MP_dim']

    matching_result = tf.concat(axis=2, values=matching_result)
    return matching_result, match_dim


def match_graph_1_with_graph_2(node_1_rep, node_2_rep, node_1_mask, node_2_mask, node_rep_dim, options=None,
                               watch=None):
    '''
    :param node_1_rep:
    :param node_2_rep:
    :param node_1_mask:
    :param node_2_mask:
    :param node_rep_dim: dim of node representation
    :param with_maxpool_match:
    :param with_max_attentive_match:
    :param options:
    :return:
    '''

    with_maxpool_match = options["with_maxpool_match"]
    with_max_attentive_match = options["with_max_attentive_match"]

    # an array of [batch_size, single_graph_1_nodes_size]
    all_graph_2_aware_representations = []
    dim = 0
    with tf.variable_scope('match_graph_1_with_graph_2'):
        # [batch_size, single_graph_1_nodes_size, single_graph_2_nodes_size]
        relevancy_matrix = cal_relevancy_matrix(node_2_rep, node_1_rep, watch=watch)
        relevancy_matrix = mask_relevancy_matrix(relevancy_matrix, node_2_mask, node_1_mask)

        all_graph_2_aware_representations.append(tf.reduce_max(relevancy_matrix, axis=2, keepdims=True))
        all_graph_2_aware_representations.append(tf.reduce_mean(relevancy_matrix, axis=2, keepdims=True))
        dim += 2

        if with_maxpool_match:
            maxpooling_decomp_params = tf.get_variable("maxpooling_matching_decomp",
                                                       shape=[options['cosine_MP_dim'], node_rep_dim],
                                                       dtype=tf.float32)

            # [batch_size, single_graph_1_nodes_size, 2 * decompse_dim]
            maxpooling_rep = cal_maxpooling_matching(node_1_rep, node_2_rep, maxpooling_decomp_params)
            maxpooling_rep = tf.multiply(maxpooling_rep, tf.expand_dims(node_1_mask, -1))
            all_graph_2_aware_representations.append(maxpooling_rep)
            dim += 2 * options['cosine_MP_dim']

        if with_max_attentive_match:
            # [batch_size, single_graph_1_nodes_size, dim]
            max_att = cal_max_node_2_representation(node_2_rep, relevancy_matrix)

            # [batch_size, single_graph_1_nodes_size, match_dim]
            (max_attentive_rep, match_dim) = multi_perspective_match(node_rep_dim, node_1_rep, max_att, options=options,
                                                                     scope_name='mp-match-max-att')
            max_attentive_rep = tf.multiply(max_attentive_rep, tf.expand_dims(node_1_mask, -1))
            all_graph_2_aware_representations.append(max_attentive_rep)
            dim += match_dim

        # [batch_size, single_graph_1_nodes_size, dim]
        all_graph_2_aware_representations = tf.concat(axis=2, values=all_graph_2_aware_representations)

    return all_graph_2_aware_representations, dim


def highway_layer(in_val, output_size, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
    #     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = tf.nn.tanh(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = trans * gate + in_val * (1.0 - gate)
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs


def multi_highway_layer(in_val, output_size, num_layers, scope=None):
    scope_name = 'highway_layer'
    if scope is not None: scope_name = scope
    for i in range(num_layers):
        cur_scope_name = scope_name + "-{}".format(i)
        in_val = highway_layer(in_val, output_size, scope=cur_scope_name)
    return in_val


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def random(shape, name=None):
    # tf.get_variable('W_train',
    #                 shape=[self.word_vocab_size, self.word_embedding_dim],
    # initializer=tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


class LayerUtils:
    def __init__(self):
        pass

    def my_lstm_layer(self, input_reps, lstm_dim, input_lengths=None, scope_name=None, reuse=False, is_training=True,
                      dropout_rate=0.2, use_cudnn=True):
        """
        :param inputs: [batch_size, seq_len, feature_dim]
        :param lstm_dim:
        :param scope_name:
        :param reuse:
        :param is_training:
        :param dropout_rate:
        :return:
        """
        input_reps = self.dropout_layer(input_reps, dropout_rate, is_training=is_training)
        with tf.variable_scope(scope_name, reuse=reuse):
            if use_cudnn:
                inputs = tf.transpose(input_reps, [1, 0, 2])
                lstm = tf.contrib.cudnn_rnn.CudnnLSTM(1, lstm_dim, direction="bidirectional",
                                                      name="{}_cudnn_bi_lstm".format(scope_name),
                                                      dropout=dropout_rate if is_training else 0)
                outputs, _ = lstm(inputs)
                outputs = tf.transpose(outputs, [1, 0, 2])
                f_rep = outputs[:, :, 0:lstm_dim]
                b_rep = outputs[:, :, lstm_dim:2 * lstm_dim]
            else:
                context_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim)
                context_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim)
                if is_training:
                    context_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_fw,
                                                                         output_keep_prob=(1 - dropout_rate))
                    context_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(context_lstm_cell_bw,
                                                                         output_keep_prob=(1 - dropout_rate))
                context_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_fw])
                context_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_bw])

                (f_rep, b_rep), _ = tf.nn.bidirectional_dynamic_rnn(
                    context_lstm_cell_fw, context_lstm_cell_bw, input_reps, dtype=tf.float32,
                    sequence_length=input_lengths)  # [batch_size, question_len, context_lstm_dim]
                outputs = tf.concat(axis=2, values=[f_rep, b_rep])
        return f_rep, b_rep, outputs

    @staticmethod
    def dropout_layer(input_reps, dropout_rate, is_training=True):
        if is_training:
            output_repr = tf.nn.dropout(input_reps, (1 - dropout_rate))
        else:
            output_repr = input_reps
        return output_repr

    @staticmethod
    def cosine_distance(y1, y2, cosine_norm=True, eps=1e-6):
        # cosine_norm = True
        # y1 [....,a, 1, d]
        # y2 [....,1, b, d]
        cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
        if not cosine_norm:
            return tf.tanh(cosine_numerator)
        y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps))
        y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps))
        return cosine_numerator / y1_norm / y2_norm

    @staticmethod
    def euclidean_distance(y1, y2, eps=1e-6):
        distance = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1 - y2), axis=-1), eps))
        return distance

    @staticmethod
    def cross_entropy(logits, truth, mask=None):
        # logits: [batch_size, passage_len]
        # truth: [batch_size, passage_len]
        # mask: [batch_size, passage_len]
        if mask is not None: logits = tf.multiply(logits, mask)
        xdev = tf.subtract(logits, tf.expand_dims(tf.reduce_max(logits, 1), -1))
        log_predictions = tf.subtract(xdev, tf.expand_dims(tf.log(tf.reduce_sum(tf.exp(xdev), -1)), -1))
        result = tf.multiply(truth, log_predictions)  # [batch_size, passage_len]
        if mask is not None:
            result = tf.multiply(result, mask)  # [batch_size, passage_len]
        return tf.multiply(-1.0, tf.reduce_sum(result, -1))  # [batch_size]

    @staticmethod
    def projection_layer(in_val, input_size, output_size, activation_func=tf.tanh, scope=None):
        # in_val: [batch_size, passage_len, dim]
        input_shape = tf.shape(in_val)
        batch_size = input_shape[0]
        passage_len = input_shape[1]
        #     feat_dim = input_shape[2]
        in_val = tf.reshape(in_val, [batch_size * passage_len, input_size])
        with tf.variable_scope(scope or "projection_layer"):
            full_w = tf.get_variable("full_w", [input_size, output_size], dtype=tf.float32)
            full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
            outputs = activation_func(tf.nn.xw_plus_b(in_val, full_w, full_b))
        outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
        return outputs  # [batch_size, passage_len, output_size]

    @staticmethod
    def highway_layer(in_val, output_size, activation_func=tf.tanh, scope=None):
        # in_val: [batch_size, passage_len, dim]
        input_shape = tf.shape(in_val)
        batch_size = input_shape[0]
        passage_len = input_shape[1]
        #     feat_dim = input_shape[2]
        in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
        with tf.variable_scope(scope or "highway_layer"):
            highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
            highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
            full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
            full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
            trans = activation_func(tf.nn.xw_plus_b(in_val, full_w, full_b))
            gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
            outputs = tf.add(tf.multiply(trans, gate), tf.multiply(in_val, tf.subtract(1.0, gate)), "y")
        outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
        return outputs

    def multi_highway_layer(self, in_val, output_size, num_layers, activation_func=tf.tanh, scope_name=None,
                            reuse=False):
        with tf.variable_scope(scope_name, reuse=reuse):
            for i in range(num_layers):
                cur_scope_name = scope_name + "-{}".format(i)
                in_val = self.highway_layer(in_val, output_size, activation_func=activation_func, scope=cur_scope_name)
        return in_val

    def collect_representation(self, representation, positions):
        # representation: [batch_size, node_num, feature_dim]
        # positions: [batch_size, neigh_num]
        return self.collect_probs(representation, positions)

    def collect_final_step_of_lstm(self, lstm_representation, lengths):
        # lstm_representation: [batch_size, passsage_length, dim]
        # lengths: [batch_size]
        lengths = tf.maximum(lengths, tf.zeros_like(lengths, dtype=tf.int32))

        batch_size = tf.shape(lengths)[0]
        batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
        indices = tf.stack((batch_nums, lengths), axis=1)  # shape (batch_size, 2)
        result = tf.gather_nd(lstm_representation, indices, name='last-forwar-lstm')
        return result  # [batch_size, dim]

    def collect_probs(self, probs, positions):
        # probs [batch_size, chunks_size]
        # positions [batch_size, pair_size]
        batch_size = tf.shape(probs)[0]
        pair_size = tf.shape(positions)[1]
        batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
        batch_nums = tf.reshape(batch_nums, shape=[-1, 1])  # [batch_size, 1]
        batch_nums = tf.tile(batch_nums, multiples=[1, pair_size])  # [batch_size, pair_size]

        indices = tf.stack((batch_nums, positions), axis=2)  # shape (batch_size, pair_size, 2)
        pair_probs = tf.gather_nd(probs, indices)
        # pair_probs = tf.reshape(pair_probs, shape=[batch_size, pair_size])
        return pair_probs

    def calcuate_attention(self, in_value_1, in_value_2, feature_dim1, feature_dim2, scope_name='att',
                           att_type='symmetric', att_dim=20, remove_diagnoal=False, mask1=None, mask2=None,
                           is_training=False, dropout_rate=0.2):
        input_shape = tf.shape(in_value_1)
        batch_size = input_shape[0]
        len_1 = input_shape[1]
        len_2 = tf.shape(in_value_2)[1]

        in_value_1 = self.dropout_layer(in_value_1, dropout_rate, is_training=is_training)
        in_value_2 = self.dropout_layer(in_value_2, dropout_rate, is_training=is_training)
        with tf.variable_scope(scope_name):
            # calculate attention ==> a: [batch_size, len_1, len_2]
            atten_w1 = tf.get_variable("atten_w1", [feature_dim1, att_dim], dtype=tf.float32)
            if feature_dim1 == feature_dim2:
                atten_w2 = atten_w1
            else:
                atten_w2 = tf.get_variable("atten_w2", [feature_dim2, att_dim], dtype=tf.float32)
            atten_value_1 = tf.matmul(tf.reshape(in_value_1, [batch_size * len_1, feature_dim1]),
                                      atten_w1)  # [batch_size*len_1, feature_dim]
            atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, att_dim])
            atten_value_2 = tf.matmul(tf.reshape(in_value_2, [batch_size * len_2, feature_dim2]),
                                      atten_w2)  # [batch_size*len_2, feature_dim]
            atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, att_dim])

            if att_type == 'additive':
                atten_b = tf.get_variable("atten_b", [att_dim], dtype=tf.float32)
                atten_v = tf.get_variable("atten_v", [1, att_dim], dtype=tf.float32)
                atten_value_1 = tf.expand_dims(atten_value_1, axis=2,
                                               name="atten_value_1")  # [batch_size, len_1, 'x', feature_dim]
                atten_value_2 = tf.expand_dims(atten_value_2, axis=1,
                                               name="atten_value_2")  # [batch_size, 'x', len_2, feature_dim]
                atten_value = atten_value_1 + atten_value_2  # + tf.expand_dims(tf.expand_dims(tf.expand_dims(atten_b, axis=0), axis=0), axis=0)
                atten_value = nn_ops.bias_add(atten_value, atten_b)
                atten_value = tf.tanh(atten_value)  # [batch_size, len_1, len_2, feature_dim]
                atten_value = tf.reshape(atten_value, [-1,
                                                       att_dim]) * atten_v  # tf.expand_dims(atten_v, axis=0) # [batch_size*len_1*len_2, feature_dim]
                atten_value = tf.reduce_sum(atten_value, axis=-1)
                atten_value = tf.reshape(atten_value, [batch_size, len_1, len_2])
            else:
                atten_value_1 = tf.tanh(atten_value_1)
                # atten_value_1 = tf.nn.relu(atten_value_1)
                atten_value_2 = tf.tanh(atten_value_2)
                # atten_value_2 = tf.nn.relu(atten_value_2)
                diagnoal_params = tf.get_variable("diagnoal_params", [1, 1, att_dim], dtype=tf.float32)
                atten_value_1 = atten_value_1 * diagnoal_params
                atten_value = tf.matmul(atten_value_1, atten_value_2, transpose_b=True)  # [batch_size, len_1, len_2]

            # normalize
            if remove_diagnoal:
                diagnoal = tf.ones([len_1], tf.float32)  # [len1]
                diagnoal = 1.0 - tf.diag(diagnoal)  # [len1, len1]
                diagnoal = tf.expand_dims(diagnoal, axis=0)  # ['x', len1, len1]
                atten_value = atten_value * diagnoal
            if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
            if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))
            atten_value = tf.nn.softmax(atten_value, name='atten_value')  # [batch_size, len_1, len_2]
            if remove_diagnoal: atten_value = atten_value * diagnoal
            if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
            if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))

        return atten_value

    def weighted_sum(self, atten_scores, in_values):
        """
        :param atten_scores: # [batch_size, len1, len2]
        :param in_values: [batch_size, len2, dim]
        :return:
        """
        return tf.matmul(atten_scores, in_values)

    def cal_relevancy_matrix(self, in_question_repres, in_passage_repres):
        in_question_repres_tmp = tf.expand_dims(in_question_repres, 1)  # [batch_size, 1, question_len, dim]
        in_passage_repres_tmp = tf.expand_dims(in_passage_repres, 2)  # [batch_size, passage_len, 1, dim]
        relevancy_matrix = self.cosine_distance(in_question_repres_tmp,
                                                in_passage_repres_tmp)  # [batch_size, passage_len, question_len]
        return relevancy_matrix

    def mask_relevancy_matrix(self, relevancy_matrix, question_mask, passage_mask):
        # relevancy_matrix: [batch_size, passage_len, question_len]
        # question_mask: [batch_size, question_len]
        # passage_mask: [batch_size, passsage_len]
        if question_mask is not None:
            relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(question_mask, 1))
        relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(passage_mask, 2))
        return relevancy_matrix

    @staticmethod
    def compute_gradients(tensor, var_list):
        grads = tf.gradients(tensor, var_list)
        return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]


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
        """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
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
            outputs = self._call(inputs)
            return outputs


class Dense(Layer):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, dropout=0.,
                 act=tf.nn.relu, placeholders=None, bias=True, featureless=False,
                 sparse_inputs=False, name='', **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout
        self.name = name
        self.act = act
        self.featureless = featureless
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim

        # helper variable for sparse dropout
        self.sparse_inputs = sparse_inputs
        if sparse_inputs:
            self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.get_variable('weights', shape=(input_dim, output_dim),
                                                   dtype=tf.float32,
                                                   initializer=tf.contrib.layers.xavier_initializer(),
                                                   regularizer=tf.contrib.layers.l2_regularizer(0.0000))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

    def _call(self, inputs):
        x = inputs

        # x = tf.nn.dropout(x, self.dropout)

        # transform
        output = tf.matmul(x, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class UniformNeighborSampler(Layer):
    """
       Uniformly samples neighbors.
       Assumes that adj lists are padded with random re-sampling
    """

    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        adj_lists = tf.transpose(tf.transpose(adj_lists))
        adj_lists = tf.slice(adj_lists, [0, 0], [-1, num_samples])
        return adj_lists


class GatedMeanAggregator(Layer):
    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0, bias=True, act=tf.nn.relu,
                 name=None, concat=False, **kwargs):
        super(GatedMeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if concat:
            self.output_dim = 2 * output_dim

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

            self.vars['gate_weights'] = glorot([2 * output_dim, 2 * output_dim],
                                               name='gate_weights')
            self.vars['gate_bias'] = zeros([2 * output_dim], name='bias')

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)

        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)

        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])

        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        gate = tf.concat([from_self, from_neighs], axis=1)
        gate = tf.matmul(gate, self.vars["gate_weights"]) + self.vars["gate_bias"]
        gate = tf.nn.relu(gate)

        return gate * self.act(output)


class GCNAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used self vector and neighbor vectors.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0., bias=True, act=tf.nn.relu, name=None, concat=False, mode="train", **kwargs):
        super(GCNAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.mode = mode

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['weights'] = glorot([neigh_input_dim, output_dim],
                                          name='neigh_weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        if self.mode == "train":
            neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
            self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)

        means = tf.reduce_mean(tf.concat([neigh_vecs, tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)

        # [nodes] x [out_dim]
        output = tf.matmul(means, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output), self.output_dim


class MeanAggregator(Layer):
    """Aggregates via mean followed by matmul and non-linearity."""

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0, bias=True, act=tf.nn.relu,
                 name=None, concat=False, mode="train", if_use_high_way=False, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.mode = mode
        self.if_use_high_way = if_use_high_way

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        self.neigh_input_dim = neigh_input_dim

        if concat:
            self.output_dim = 2 * output_dim
        else:
            self.output_dim = output_dim

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')

            # self.vars['neigh_weights'] = random([neigh_input_dim, output_dim], name='neigh_weights')
            # self.vars['self_weights'] = random([input_dim, output_dim], name='neigh_weights')

            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        self.input_dim = input_dim

        self.output_dim = output_dim

        if self.concat:
            self.output_dim = output_dim * 2

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        if self.mode == "train":
            neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - self.dropout)
            self_vecs = tf.nn.dropout(self_vecs, 1 - self.dropout)

        # reduce_mean performs better than mean_pool
        neigh_means = tf.reduce_mean(neigh_vecs, axis=1)
        # neigh_means = mean_pool(neigh_vecs, neigh_len)

        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])

        if self.if_use_high_way:
            with tf.variable_scope("fw_hidden_highway"):
                fw_hidden = multi_highway_layer(from_neighs, self.neigh_input_dim, 1)

        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output), self.output_dim


class AttentionAggregator(Layer):
    """ Attention-based aggregator """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0, bias=True, act=tf.nn.relu,
                 name=None, concat=False, mode="train", **kwargs):
        super(AttentionAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.mode = mode

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if neigh_input_dim == None:
            neigh_input_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim

        with tf.variable_scope(self.name + name + '_vars'):
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

            self.q_dense_layer = Dense(input_dim=input_dim, output_dim=input_dim, bias=False, sparse_inputs=False,
                                       name="q")
            self.k_dense_layer = Dense(input_dim=input_dim, output_dim=input_dim, bias=False, sparse_inputs=False,
                                       name="k")
            self.v_dense_layer = Dense(input_dim=input_dim, output_dim=input_dim, bias=False, sparse_inputs=False,
                                       name="v")

            self.output_dense_layer = Dense(input_dim=input_dim, output_dim=output_dim, bias=False, sparse_inputs=False,
                                            name="output_transform")

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        q = self.q_dense_layer(self_vecs)

        neigh_vecs = tf.concat([tf.expand_dims(self_vecs, axis=1), neigh_vecs], axis=1)
        neigh_len = tf.shape(neigh_vecs)[1]
        neigh_vecs = tf.reshape(neigh_vecs, [-1, self.input_dim])

        k = self.k_dense_layer(neigh_vecs)
        v = self.v_dense_layer(neigh_vecs)

        k = tf.reshape(k, [-1, neigh_len, self.input_dim])
        v = tf.reshape(v, [-1, neigh_len, self.input_dim])

        logits = tf.reduce_sum(tf.multiply(tf.expand_dims(q, axis=1), k), axis=-1)
        # if self.bias:
        #     logits += self.vars['bias']

        weights = tf.nn.softmax(logits, name="attention_weights")

        attention_output = tf.reduce_sum(tf.multiply(tf.expand_dims(weights, axis=-1), v), axis=1)

        attention_output = self.output_dense_layer(attention_output)

        return attention_output, self.output_dim


class MaxPoolingAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions."""

    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0., bias=True, act=tf.nn.relu, name=None, concat=False, mode="train", **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        self.mode = mode
        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if concat:
            self.output_dim = 2 * output_dim

        if model_size == "small":
            hidden_dim = self.hidden_dim = 50
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 50

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim, output_dim=hidden_dim, act=tf.nn.relu,
                                     dropout=dropout, sparse_inputs=False, logging=self.logging))

        with tf.variable_scope(self.name + name + '_vars'):

            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim], name='neigh_weights')

            self.vars['self_weights'] = glorot([input_dim, output_dim], name='self_weights')

            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

        if self.concat:
            self.output_dim = output_dim * 2

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]

        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_max(neigh_h, axis=1)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output), self.output_dim


class SeqAggregator(Layer):
    """ Aggregates via a standard LSTM.
    """

    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0., bias=True, act=tf.nn.relu, name=None, concat=False, mode="train", **kwargs):
        super(SeqAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.mode = mode
        self.output_dim = output_dim

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 128
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 256

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                name='neigh_weights')

            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        dims = tf.shape(neigh_vecs)
        batch_size = dims[0]
        initial_state = self.cell.zero_state(batch_size, tf.float32)
        used = tf.sign(tf.reduce_max(tf.abs(neigh_vecs), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.maximum(length, tf.constant(1.))
        length = tf.cast(length, tf.int32)

        with tf.variable_scope(self.name) as scope:
            try:
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                    self.cell, neigh_vecs,
                    initial_state=initial_state, dtype=tf.float32, time_major=False,
                    sequence_length=length)
            except ValueError:
                scope.reuse_variables()
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                    self.cell, neigh_vecs,
                    initial_state=initial_state, dtype=tf.float32, time_major=False,
                    sequence_length=length)
        batch_size = tf.shape(rnn_outputs)[0]
        max_len = tf.shape(rnn_outputs)[1]
        out_size = int(rnn_outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_len + (length - 1)
        flat = tf.reshape(rnn_outputs, [-1, out_size])
        neigh_h = tf.gather(flat, index)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        output = tf.add_n([from_self, from_neighs])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphMatchNN(object):
    def __init__(self, mode, conf, pretrained_word_embeddings):
        self.training_op = None
        self.loss = None
        self.mode = mode
        self.word_vocab_size = conf.word_vocab_size
        self.l2_lambda = conf.l2_lambda
        self.word_embedding_dim = conf.hidden_layer_dim
        self.encoder_hidden_dim = conf.encoder_hidden_dim

        # the setting for the GCN
        self.num_layers = conf.num_layers
        self.graph_encode_direction = conf.graph_encode_direction
        self.hidden_layer_dim = conf.hidden_layer_dim
        self.concat = conf.concat

        self.y_true = tf.placeholder(tf.float32, [None, 2], name="true_labels")

        # the following place holders are for the first graph
        self.fw_adj_info_first = tf.placeholder(tf.int32, [None, None])  # the fw adj info for each node
        self.bw_adj_info_first = tf.placeholder(tf.int32, [None, None])  # the bw adj info for each node
        self.feature_info_first = tf.placeholder(tf.int32, [None, None])  # the feature info for each node
        self.feature_len_first = tf.placeholder(tf.int32, [None])  # the feature len for each node
        self.batch_nodes_first = tf.placeholder(tf.int32, [None, None])  # the nodes for the first batch
        self.batch_mask_first = tf.placeholder(tf.float32, [None, None])  # the mask for the first batch
        self.looking_table_first = tf.placeholder(tf.int32, [None])  # the looking table for the first batch
        self.entity_index_first = tf.placeholder(tf.int32, [None])  # the entity node index in each graph

        self.fw_adj_info_second = tf.placeholder(tf.int32, [None, None])  # the fw adj info for each node
        self.bw_adj_info_second = tf.placeholder(tf.int32, [None, None])  # the bw adj info for each node
        self.feature_info_second = tf.placeholder(tf.int32, [None, None])  # the feature info for each node
        self.feature_len_second = tf.placeholder(tf.int32, [None])  # the feature len for each node
        self.batch_nodes_second = tf.placeholder(tf.int32, [None, None])  # the nodes for the first batch
        self.batch_mask_second = tf.placeholder(tf.float32, [None, None])  # the mask for the second batch
        self.looking_table_second = tf.placeholder(tf.int32, [None])  # the looking table for the second batch
        self.entity_index_second = tf.placeholder(tf.int32, [None])  # the entity node index in each graph

        self.with_match_highway = conf.with_match_highway
        self.with_gcn_highway = conf.with_gcn_highway
        self.if_use_multiple_gcn_1_state = conf.if_use_multiple_gcn_1_state
        self.if_use_multiple_gcn_2_state = conf.if_use_multiple_gcn_2_state

        self.pretrained_word_embeddings = pretrained_word_embeddings
        self.pretrained_word_size = conf.pretrained_word_size
        self.learned_word_size = conf.learned_word_size

        self.sample_size_per_layer_first = tf.shape(self.fw_adj_info_first)[1]
        self.sample_size_per_layer_second = tf.shape(self.fw_adj_info_second)[1]
        self.batch_size = tf.shape(self.y_true)[0]
        self.dropout = conf.dropout

        self.fw_aggregators_first = []
        self.bw_aggregators_first = []
        self.aggregator_dim_first = conf.aggregator_dim_first
        self.gcn_window_size_first = conf.gcn_window_size_first
        self.gcn_layer_size_first = conf.gcn_layer_size_first

        self.fw_aggregators_second = []
        self.bw_aggregators_second = []
        self.aggregator_dim_second = conf.aggregator_dim_second
        self.gcn_window_size_second = conf.gcn_window_size_second
        self.gcn_layer_size_second = conf.gcn_layer_size_second

        self.if_pred_on_dev = False
        self.learning_rate = conf.learning_rate

        self.agg_sim_method = conf.agg_sim_method

        self.agg_type_first = conf.gcn_type_first
        self.agg_type_second = conf.gcn_type_second

        self.node_vec_method = conf.node_vec_method
        self.pred_method = conf.pred_method
        self.watch = {}
        # *******************new add*********************
        self.layer_utils = LayerUtils()
        self.conf = conf
        self.conf.options = dict()
        self.conf.options = {"aggregation_layer_num": 1, "with_full_match": True, "with_maxpool_match": True,
                             "with_max_attentive_match": True, "with_attentive_match": True, "with_cosine": True,
                             "with_mp_cosine": True, "highway_layer_num": 1, "with_highway": True,
                             "with_match_highway": True, "with_aggregation_highway": True, "use_cudnn": False,
                             "aggregation_lstm_dim": 100, "with_moving_average": False,
                             'cosine_MP_dim': conf.cosine_MP_dim}
        self.temp = None

    def _build_graph(self):
        node_1_mask = self.batch_mask_first
        node_2_mask = self.batch_mask_second
        node_1_looking_table = self.looking_table_first
        node_2_looking_table = self.looking_table_second

        node_2_aware_representations = []
        node_2_aware_dim = 0
        node_1_aware_representations = []
        node_1_aware_dim = 0

        pad_word_embedding = tf.zeros([1, self.word_embedding_dim])  # this is for the PAD symbol
        self.word_embeddings = tf.concat([pad_word_embedding,
                                          tf.get_variable('pretrained_embedding',
                                                          shape=[self.pretrained_word_size, self.word_embedding_dim],
                                                          initializer=tf.constant_initializer(
                                                              self.pretrained_word_embeddings), trainable=True),
                                          tf.get_variable('W_train',
                                                          shape=[self.learned_word_size, self.word_embedding_dim],
                                                          initializer=tf.contrib.layers.xavier_initializer(),
                                                          trainable=True)], 0)

        self.watch['word_embeddings'] = self.word_embeddings

        # ============ encode node feature by looking up word embedding =============
        with tf.variable_scope('node_rep_gen'):
            # [node_size, hidden_layer_dim]
            feature_embedded_chars_first = tf.nn.embedding_lookup(self.word_embeddings, self.feature_info_first)
            graph_1_size = tf.shape(feature_embedded_chars_first)[0]

            feature_embedded_chars_second = tf.nn.embedding_lookup(self.word_embeddings, self.feature_info_second)
            graph_2_size = tf.shape(feature_embedded_chars_second)[0]

            if self.node_vec_method == "lstm":
                cell = self.build_encoder_cell(1, self.hidden_layer_dim)

                outputs, hidden_states = tf.nn.dynamic_rnn(cell=cell, inputs=feature_embedded_chars_first,
                                                           sequence_length=self.feature_len_first, dtype=tf.float32)
                node_1_rep = self.layer_utils.collect_final_step_of_lstm(outputs, self.feature_len_first - 1)

                outputs, hidden_states = tf.nn.dynamic_rnn(cell=cell, inputs=feature_embedded_chars_second,
                                                           sequence_length=self.feature_len_second, dtype=tf.float32)
                node_2_rep = self.layer_utils.collect_final_step_of_lstm(outputs, self.feature_len_second - 1)

            elif self.node_vec_method == "word_emb":
                node_1_rep = tf.reshape(feature_embedded_chars_first, [graph_1_size, -1])
                node_2_rep = tf.reshape(feature_embedded_chars_second, [graph_2_size, -1])

            self.watch["node_1_rep_initial"] = node_1_rep
            self.watch['node_2_rep_initial'] = node_2_rep

        # ============ encode node feature by GCN =============
        with tf.variable_scope('first_gcn') as first_gcn_scope:
            # shape of node embedding: [batch_size, single_graph_nodes_size, node_embedding_dim]
            # shape of node size: [batch_size]
            gcn_1_res = self.gcn_encode(self.batch_nodes_first,
                                        node_1_rep,
                                        self.fw_adj_info_first, self.bw_adj_info_first,
                                        input_node_dim=self.word_embedding_dim,
                                        output_node_dim=self.aggregator_dim_first,
                                        fw_aggregators=self.fw_aggregators_first,
                                        bw_aggregators=self.bw_aggregators_first,
                                        window_size=self.gcn_window_size_first,
                                        layer_size=self.gcn_layer_size_first,
                                        scope="first_gcn",
                                        agg_type=self.agg_type_first,
                                        sample_size_per_layer=self.sample_size_per_layer_first,
                                        keep_inter_state=self.if_use_multiple_gcn_1_state)

            node_1_rep = gcn_1_res[0]
            node_1_rep_dim = gcn_1_res[3]

            gcn_2_res = self.gcn_encode(self.batch_nodes_second,
                                        node_2_rep,
                                        self.fw_adj_info_second,
                                        self.bw_adj_info_second,
                                        input_node_dim=self.word_embedding_dim,
                                        output_node_dim=self.aggregator_dim_first,
                                        fw_aggregators=self.fw_aggregators_first,
                                        bw_aggregators=self.bw_aggregators_first,
                                        window_size=self.gcn_window_size_first,
                                        layer_size=self.gcn_layer_size_first,
                                        scope="first_gcn",
                                        agg_type=self.agg_type_first,
                                        sample_size_per_layer=self.sample_size_per_layer_second,
                                        keep_inter_state=self.if_use_multiple_gcn_1_state)

            node_2_rep = gcn_2_res[0]
            node_2_rep_dim = gcn_2_res[3]

        self.watch["node_1_rep_first_GCN"] = node_1_rep
        self.watch["node_1_mask"] = node_1_mask
        self.watch["node_2_rep_first_GCN"] = node_2_rep

        # mask
        node_1_rep = tf.multiply(node_1_rep, tf.expand_dims(node_1_mask, 2))
        node_2_rep = tf.multiply(node_2_rep, tf.expand_dims(node_2_mask, 2))

        self.watch["node_1_rep_first_GCN_masked"] = node_1_rep

        if self.pred_method == "node_level":
            entity_1_rep = tf.reshape(tf.nn.embedding_lookup(tf.transpose(node_1_rep, [1, 0, 2]), tf.constant(0)),
                                      [-1, node_1_rep_dim])
            entity_2_rep = tf.reshape(tf.nn.embedding_lookup(tf.transpose(node_2_rep, [1, 0, 2]), tf.constant(0)),
                                      [-1, node_2_rep_dim])

            entity_1_2_diff = entity_1_rep - entity_2_rep
            entity_1_2_sim = entity_1_rep * entity_2_rep

            aggregation = tf.concat([entity_1_rep, entity_2_rep, entity_1_2_diff, entity_1_2_sim], axis=1)
            aggregation_dim = 4 * node_1_rep_dim

            w_0 = tf.get_variable("w_0", [aggregation_dim, aggregation_dim / 2], dtype=tf.float32)
            b_0 = tf.get_variable("b_0", [aggregation_dim / 2], dtype=tf.float32)
            w_1 = tf.get_variable("w_1", [aggregation_dim / 2, 2], dtype=tf.float32)
            b_1 = tf.get_variable("b_1", [2], dtype=tf.float32)

            # ====== Prediction Layer ===============
            logits = tf.matmul(aggregation, w_0) + b_0
            logits = tf.tanh(logits)
            logits = tf.matmul(logits, w_1) + b_1

        elif self.pred_method == "graph_level":
            # if the prediction method is graph_level, we perform the graph matching based prediction

            assert node_1_rep_dim == node_2_rep_dim
            input_dim = node_1_rep_dim

            with tf.variable_scope('node_level_matching') as matching_scope:
                # ========= node level matching ===============
                (match_reps, match_dim) = match_graph_1_with_graph_2(node_1_rep, node_2_rep, node_1_mask, node_2_mask,
                                                                     input_dim,
                                                                     options=self.conf.options, watch=self.watch)

                matching_scope.reuse_variables()

                node_2_aware_representations.append(match_reps)
                node_2_aware_dim += match_dim

                (match_reps, match_dim) = match_graph_1_with_graph_2(node_2_rep, node_1_rep, node_2_mask, node_1_mask,
                                                                     input_dim,
                                                                     options=self.conf.options, watch=self.watch)

                node_1_aware_representations.append(match_reps)
                node_1_aware_dim += match_dim

            # TODO: add one more MP matching over the graph representation
            # with tf.variable_scope('context_MP_matching'):
            #     for i in range(options['context_layer_num']):
            #         with tf.variable_scope('layer-{}',format(i)):

            # [batch_size, single_graph_nodes_size, node_2_aware_dim]
            node_2_aware_representations = tf.concat(axis=2, values=node_2_aware_representations)

            # [batch_size, single_graph_nodes_size, node_1_aware_dim]
            node_1_aware_representations = tf.concat(axis=2, values=node_1_aware_representations)

            # if self.mode == "train":
            #     node_2_aware_representations = tf.nn.dropout(node_2_aware_representations, (1 - options['dropout_rate']))
            #     node_1_aware_representations = tf.nn.dropout(node_1_aware_representations, (1 - options['dropout_rate']))

            # ========= Highway layer ==============
            if self.with_match_highway:
                with tf.variable_scope("left_matching_highway"):
                    node_2_aware_representations = multi_highway_layer(node_2_aware_representations, node_2_aware_dim,
                                                                       self.conf.options['highway_layer_num'])
                with tf.variable_scope("right_matching_highway"):
                    node_1_aware_representations = multi_highway_layer(node_1_aware_representations, node_1_aware_dim,
                                                                       self.conf.options['highway_layer_num'])

            self.watch["node_1_rep_match"] = node_2_aware_representations

            # ========= Aggregation Layer ==============
            aggregation_representation = []
            aggregation_dim = 0

            node_2_aware_aggregation_input = node_2_aware_representations
            node_1_aware_aggregation_input = node_1_aware_representations

            self.watch["node_1_rep_match_layer"] = node_2_aware_aggregation_input

            with tf.variable_scope('aggregation_layer'):
                # TODO: now we only have 1 aggregation layer;
                #  need to change this part if support more aggregation layers
                # [batch_size, single_graph_nodes_size, node_2_aware_dim]
                node_2_aware_aggregation_input = tf.multiply(node_2_aware_aggregation_input,
                                                             tf.expand_dims(node_1_mask, axis=-1))

                # [batch_size, single_graph_nodes_size, node_1_aware_dim]
                node_1_aware_aggregation_input = tf.multiply(node_1_aware_aggregation_input,
                                                             tf.expand_dims(node_2_mask, axis=-1))

                if self.agg_sim_method == "GCN":
                    # [batch_size*single_graph_nodes_size, node_2_aware_dim]
                    node_2_aware_aggregation_input = tf.reshape(node_2_aware_aggregation_input,
                                                                shape=[-1, node_2_aware_dim])

                    # [batch_size*single_graph_nodes_size, node_1_aware_dim]
                    node_1_aware_aggregation_input = tf.reshape(node_1_aware_aggregation_input,
                                                                shape=[-1, node_1_aware_dim])

                    # [node_1_size, node_2_aware_dim]
                    node_1_rep = tf.concat(
                        [tf.nn.embedding_lookup(node_2_aware_aggregation_input, node_1_looking_table),
                         tf.zeros([1, node_2_aware_dim])], 0)

                    # [node_2_size, node_1_aware_dim]
                    node_2_rep = tf.concat(
                        [tf.nn.embedding_lookup(node_1_aware_aggregation_input, node_2_looking_table),
                         tf.zeros([1, node_1_aware_dim])], 0)

                    gcn_1_res = self.gcn_encode(self.batch_nodes_first,
                                                node_1_rep,
                                                self.fw_adj_info_first,
                                                self.bw_adj_info_first,
                                                input_node_dim=node_2_aware_dim,
                                                output_node_dim=self.aggregator_dim_second,
                                                fw_aggregators=self.fw_aggregators_second,
                                                bw_aggregators=self.bw_aggregators_second,
                                                window_size=self.gcn_window_size_second,
                                                layer_size=self.gcn_layer_size_second,
                                                scope="second_gcn",
                                                agg_type=self.agg_type_second,
                                                sample_size_per_layer=self.sample_size_per_layer_first,
                                                keep_inter_state=self.if_use_multiple_gcn_2_state)

                    max_graph_1_rep = gcn_1_res[1]
                    mean_graph_1_rep = gcn_1_res[2]
                    graph_1_rep_dim = gcn_1_res[3]

                    gcn_2_res = self.gcn_encode(self.batch_nodes_second,
                                                node_2_rep,
                                                self.fw_adj_info_second,
                                                self.bw_adj_info_second,
                                                input_node_dim=node_1_aware_dim,
                                                output_node_dim=self.aggregator_dim_second,
                                                fw_aggregators=self.fw_aggregators_second,
                                                bw_aggregators=self.bw_aggregators_second,
                                                window_size=self.gcn_window_size_second,
                                                layer_size=self.gcn_layer_size_second,
                                                scope="second_gcn",
                                                agg_type=self.agg_type_second,
                                                sample_size_per_layer=self.sample_size_per_layer_second,
                                                keep_inter_state=self.if_use_multiple_gcn_2_state)

                    max_graph_2_rep = gcn_2_res[1]
                    mean_graph_2_rep = gcn_2_res[2]
                    graph_2_rep_dim = gcn_2_res[3]

                    assert graph_1_rep_dim == graph_2_rep_dim

                    if self.if_use_multiple_gcn_2_state:
                        graph_1_reps = gcn_1_res[5]
                        graph_2_reps = gcn_2_res[5]
                        inter_dims = gcn_1_res[6]
                        for idx in range(len(graph_1_reps)):
                            (max_graph_1_rep_tmp, mean_graph_1_rep_tmp) = graph_1_reps[idx]
                            (max_graph_2_rep_tmp, mean_graph_2_rep_tmp) = graph_2_reps[idx]
                            inter_dim = inter_dims[idx]
                            aggregation_representation.append(max_graph_1_rep_tmp)
                            aggregation_representation.append(mean_graph_1_rep_tmp)
                            aggregation_representation.append(max_graph_2_rep_tmp)
                            aggregation_representation.append(mean_graph_2_rep_tmp)
                            aggregation_dim += 4 * inter_dim

                    else:
                        aggregation_representation.append(max_graph_1_rep)
                        aggregation_representation.append(mean_graph_1_rep)
                        aggregation_representation.append(max_graph_2_rep)
                        aggregation_representation.append(mean_graph_2_rep)
                        aggregation_dim = 4 * graph_1_rep_dim

                    # aggregation_representation = tf.concat(aggregation_representation, axis=1)

                    gcn_2_window_size = int(len(aggregation_representation) / 4)
                    aggregation_dim = aggregation_dim / gcn_2_window_size

                    w_0 = tf.get_variable("w_0", [aggregation_dim, aggregation_dim / 2], dtype=tf.float32)
                    b_0 = tf.get_variable("b_0", [aggregation_dim / 2], dtype=tf.float32)
                    w_1 = tf.get_variable("w_1", [aggregation_dim / 2, 2], dtype=tf.float32)
                    b_1 = tf.get_variable("b_1", [2], dtype=tf.float32)

                    weights = tf.get_variable("gcn_2_window_weights", [gcn_2_window_size], dtype=tf.float32)

                    # shape: [gcn_2_window_size, batch_size, 2]
                    logits = []
                    for layer_idx in range(gcn_2_window_size):
                        max_graph_1_rep = aggregation_representation[layer_idx * 4 + 0]
                        mean_graph_1_rep = aggregation_representation[layer_idx * 4 + 1]
                        max_graph_2_rep = aggregation_representation[layer_idx * 4 + 2]
                        mean_graph_2_rep = aggregation_representation[layer_idx * 4 + 3]

                        aggregation_representation_single = tf.concat(
                            [max_graph_1_rep, mean_graph_1_rep, max_graph_2_rep, mean_graph_2_rep], axis=1)

                        # ====== Prediction Layer ===============
                        logit = tf.matmul(aggregation_representation_single, w_0) + b_0
                        logit = tf.tanh(logit)
                        logit = tf.matmul(logit, w_1) + b_1
                        logits.append(logit)

                    if len(logits) != 1:
                        logits = tf.reshape(tf.concat(logits, axis=0), [gcn_2_window_size, -1, 2])
                        logits = tf.transpose(logits, [1, 0, 2])
                        logits = tf.multiply(logits, tf.expand_dims(weights, axis=-1))
                        logits = tf.reduce_sum(logits, axis=1)
                    else:
                        logits = tf.reshape(logits, [-1, 2])

        # ====== Highway layer ============
        # if options['with_aggregation_highway']:

        with tf.name_scope("loss"):
            self.y_pred = tf.nn.softmax(logits)
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true, logits=logits,
                                                                              name="xentropy_loss")) / tf.cast(
                self.batch_size, tf.float32)

        # ============  Training Objective ===========================
        if self.mode == "train" and not self.if_pred_on_dev:
            optimizer = tf.train.AdamOptimizer()
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1)
            self.training_op = optimizer.apply_gradients(zip(clipped_gradients, params))

    def build_encoder_cell(self, num_layers, hidden_size):
        if num_layers == 1:
            cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            if self.mode == "train" and self.dropout > 0.0:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.dropout)
            return cell
        else:
            cell_list = []
            for i in range(num_layers):
                single_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
                if self.mode == "train" and self.dropout > 0.0:
                    single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, self.dropout)
                cell_list.append(single_cell)
            return tf.contrib.rnn.MultiRNNCell(cell_list)

    def gcn_encode(self, batch_nodes, embedded_node_rep, fw_adj_info, bw_adj_info, input_node_dim, output_node_dim,
                   fw_aggregators, bw_aggregators, window_size, layer_size, scope, agg_type, sample_size_per_layer,
                   keep_inter_state=False):
        with tf.variable_scope(scope):
            single_graph_nodes_size = tf.shape(batch_nodes)[1]
            # ============ encode graph structure ==========
            fw_sampler = UniformNeighborSampler(fw_adj_info)
            bw_sampler = UniformNeighborSampler(bw_adj_info)
            nodes = tf.reshape(batch_nodes, [-1, ])

            # the fw_hidden and bw_hidden is the initial node embedding
            # [node_size, dim_size]
            fw_hidden = tf.nn.embedding_lookup(embedded_node_rep, nodes)
            bw_hidden = tf.nn.embedding_lookup(embedded_node_rep, nodes)

            # [node_size, adj_size]
            fw_sampled_neighbors = fw_sampler((nodes, sample_size_per_layer))
            bw_sampled_neighbors = bw_sampler((nodes, sample_size_per_layer))

            inter_fw_hiddens = []
            inter_bw_hiddens = []
            inter_dims = []

            if scope == "first_gcn":
                self.watch["node_1_rep_in_first_gcn"] = []

            fw_hidden_dim = input_node_dim
            # layer is the index of convolution and hop is used to combine information
            for layer in range(layer_size):
                self.watch["node_1_rep_in_first_gcn"].append(fw_hidden)

                if len(fw_aggregators) <= layer:
                    fw_aggregators.append([])
                if len(bw_aggregators) <= layer:
                    bw_aggregators.append([])
                for hop in range(window_size):
                    if hop > 6:
                        fw_aggregator = fw_aggregators[layer][6]
                    elif len(fw_aggregators[layer]) > hop:
                        fw_aggregator = fw_aggregators[layer][hop]
                    else:
                        if agg_type == "GCN":
                            fw_aggregator = GCNAggregator(fw_hidden_dim, output_node_dim, concat=self.concat,
                                                          dropout=self.dropout, mode=self.mode)
                        elif agg_type == "mean_pooling":
                            fw_aggregator = MeanAggregator(fw_hidden_dim, output_node_dim, concat=self.concat,
                                                           dropout=self.dropout, if_use_high_way=self.with_gcn_highway,
                                                           mode=self.mode)
                        elif agg_type == "max_pooling":
                            fw_aggregator = MaxPoolingAggregator(fw_hidden_dim, output_node_dim, concat=self.concat,
                                                                 dropout=self.dropout, mode=self.mode)
                        elif agg_type == "lstm":
                            fw_aggregator = SeqAggregator(fw_hidden_dim, output_node_dim, concat=self.concat,
                                                          dropout=self.dropout, mode=self.mode)
                        elif agg_type == "att":
                            fw_aggregator = AttentionAggregator(fw_hidden_dim, output_node_dim, concat=self.concat,
                                                                dropout=self.dropout, mode=self.mode)

                        fw_aggregators[layer].append(fw_aggregator)

                    # [node_size, adj_size, word_embedding_dim]
                    if layer == 0 and hop == 0:
                        neigh_vec_hidden = tf.nn.embedding_lookup(embedded_node_rep, fw_sampled_neighbors)
                    else:
                        neigh_vec_hidden = tf.nn.embedding_lookup(
                            tf.concat([fw_hidden, tf.zeros([1, fw_hidden_dim])], 0), fw_sampled_neighbors)

                    # if self.with_gcn_highway:
                    #     # we try to forget something when introducing the neighbor information
                    #     with tf.variable_scope("fw_hidden_highway"):
                    #         fw_hidden = multi_highway_layer(fw_hidden, fw_hidden_dim, options['highway_layer_num'])

                    bw_hidden_dim = fw_hidden_dim

                    fw_hidden, fw_hidden_dim = fw_aggregator((fw_hidden, neigh_vec_hidden))

                    if keep_inter_state:
                        inter_fw_hiddens.append(fw_hidden)
                        inter_dims.append(fw_hidden_dim)

                    if self.graph_encode_direction == "bi":
                        if hop > 6:
                            bw_aggregator = bw_aggregators[layer][6]
                        elif len(bw_aggregators[layer]) > hop:
                            bw_aggregator = bw_aggregators[layer][hop]
                        else:
                            if agg_type == "GCN":
                                bw_aggregator = GCNAggregator(bw_hidden_dim, output_node_dim, concat=self.concat,
                                                              dropout=self.dropout, mode=self.mode)
                            elif agg_type == "mean_pooling":
                                bw_aggregator = MeanAggregator(bw_hidden_dim, output_node_dim, concat=self.concat,
                                                               dropout=self.dropout,
                                                               if_use_high_way=self.with_gcn_highway, mode=self.mode)
                            elif agg_type == "max_pooling":
                                bw_aggregator = MaxPoolingAggregator(bw_hidden_dim, output_node_dim, concat=self.concat,
                                                                     dropout=self.dropout, mode=self.mode)
                            elif agg_type == "lstm":
                                bw_aggregator = SeqAggregator(bw_hidden_dim, output_node_dim, concat=self.concat,
                                                              dropout=self.dropout, mode=self.mode)
                            elif agg_type == "att":
                                bw_aggregator = AttentionAggregator(bw_hidden_dim, output_node_dim, concat=self.concat,
                                                                    mode=self.mode, dropout=self.dropout)

                            bw_aggregators[layer].append(bw_aggregator)

                        if layer == 0 and hop == 0:
                            neigh_vec_hidden = tf.nn.embedding_lookup(embedded_node_rep, bw_sampled_neighbors)
                        else:
                            neigh_vec_hidden = tf.nn.embedding_lookup(
                                tf.concat([bw_hidden, tf.zeros([1, fw_hidden_dim])], 0), bw_sampled_neighbors)

                        if self.with_gcn_highway:
                            with tf.variable_scope("bw_hidden_highway"):
                                bw_hidden = multi_highway_layer(bw_hidden, fw_hidden_dim,
                                                                self.conf.options['highway_layer_num'])

                        bw_hidden, bw_hidden_dim = bw_aggregator((bw_hidden, neigh_vec_hidden))

                        if keep_inter_state:
                            inter_bw_hiddens.append(bw_hidden)

            node_dim = fw_hidden_dim

            # hidden stores the representation for all nodes
            fw_hidden = tf.reshape(fw_hidden, [-1, single_graph_nodes_size, node_dim])
            if self.graph_encode_direction == "bi":
                bw_hidden = tf.reshape(bw_hidden, [-1, single_graph_nodes_size, node_dim])
                hidden = tf.concat([fw_hidden, bw_hidden], axis=2)
                graph_dim = 2 * node_dim
            else:
                hidden = fw_hidden
                graph_dim = node_dim

            hidden = tf.nn.relu(hidden)
            max_pooled = tf.reduce_max(hidden, 1)
            mean_pooled = tf.reduce_mean(hidden, 1)
            res = [hidden]

            max_graph_embedding = tf.reshape(max_pooled, [-1, graph_dim])
            mean_graph_embedding = tf.reshape(mean_pooled, [-1, graph_dim])
            res.append(max_graph_embedding)
            res.append(mean_graph_embedding)
            res.append(graph_dim)

            if keep_inter_state:
                inter_node_reps = []
                inter_graph_reps = []
                inter_graph_dims = []
                # process the inter hidden states
                for _ in range(len(inter_fw_hiddens)):
                    inter_fw_hidden = inter_fw_hiddens[_]
                    inter_bw_hidden = inter_bw_hiddens[_]
                    inter_dim = inter_dims[_]
                    inter_fw_hidden = tf.reshape(inter_fw_hidden, [-1, single_graph_nodes_size, inter_dim])

                    if self.graph_encode_direction == "bi":
                        inter_bw_hidden = tf.reshape(inter_bw_hidden, [-1, single_graph_nodes_size, inter_dim])
                        inter_hidden = tf.concat([inter_fw_hidden, inter_bw_hidden], axis=2)
                        inter_graph_dim = inter_dim * 2
                    else:
                        inter_hidden = inter_fw_hidden
                        inter_graph_dim = inter_dim

                    inter_node_rep = tf.nn.relu(inter_hidden)
                    inter_node_reps.append(inter_node_rep)
                    inter_graph_dims.append(inter_graph_dim)

                    max_pooled_tmp = tf.reduce_max(inter_node_rep, 1)
                    mean_pooled_tmp = tf.reduce_max(inter_node_rep, 1)
                    max_graph_embedding = tf.reshape(max_pooled_tmp, [-1, inter_graph_dim])
                    mean_graph_embedding = tf.reshape(mean_pooled_tmp, [-1, inter_graph_dim])
                    inter_graph_reps.append((max_graph_embedding, mean_graph_embedding))

                res.append(inter_node_reps)
                res.append(inter_graph_reps)
                res.append(inter_graph_dims)

            return res

    def act(self, sess, mode, dict, if_pred_on_dev, eval):
        self.if_pred_on_dev = if_pred_on_dev

        feed_dict = {
            self.y_true: np.array(dict['y']),
            self.fw_adj_info_first: np.array(dict['fw_adj_info_first']),
            self.bw_adj_info_first: np.array(dict['bw_adj_info_first']),
            self.feature_info_first: np.array(dict['feature_info_first']),
            self.feature_len_first: np.array(dict['feature_len_first']),
            self.batch_nodes_first: np.array(dict['batch_nodes_first']),
            self.batch_mask_first: np.array(dict['batch_mask_first']),
            self.looking_table_first: np.array(dict['looking_table_first']),

            self.fw_adj_info_second: np.array(dict['fw_adj_info_second']),
            self.bw_adj_info_second: np.array(dict['bw_adj_info_second']),
            self.feature_info_second: np.array(dict['feature_info_second']),
            self.feature_len_second: np.array(dict['feature_len_second']),
            self.batch_nodes_second: np.array(dict['batch_nodes_second']),
            self.batch_mask_second: np.array(dict['batch_mask_second']),
            self.looking_table_second: np.array(dict['looking_table_second']),
        }
        if mode == "train" and not if_pred_on_dev:
            output_feeds = [self.watch, self.training_op, self.loss]
        elif mode == "test" or if_pred_on_dev:
            output_feeds = [self.y_pred]

        if eval:
            embeds1 = sess.run(self.watch["node_1_rep_initial"], feed_dict)
            embeds2 = sess.run(self.watch["node_2_rep_initial"], feed_dict)
            return embeds1, embeds2
        results = sess.run(output_feeds, feed_dict)
        return results


class GMNN(BasicModel):
    def __init__(self):
        super().__init__()
        self.attr = None
        self.opt = 'SGD'
        self.act_func = tf.nn.relu
        self.dropout = 0.0
        # *****************************add*******************************************************
        self.struct_loss = None
        self.struct_optimizer = None
        self.train = None
        # **********GMNN add new arguments**************************
        self.use_pretrained_embedding = None
        self.hidden_layer_dim = None
        self.pretraind_word_embedding = None
        self.word_idx = None
        self.embedding_path = None
        self.pretraind_word_size = 0
        self.word_embedding_dime = None
        self.unknown_word = None
        self.conf = None
        self.id_dict1 = dict()
        self.id_dict2 = dict()
        self.assign_gen = dict()
        self.save_g1_batch = None
        self.save_g2_batch = None
        self.save_label_batch = None
        self.temp_model = None
        self.graphs_1_test = None
        self.graphs_2_test = None
        self.labels_test = None
        self.saver = None

    def init(self):

        # ****************************init GMNN arguments****************
        self.use_pretrained_embedding = self.args.use_pretrained_embedding
        self.hidden_layer_dim = self.args.hidden_layer_dim
        self.pretraind_word_embedding = np.array([])
        self.word_idx = dict()
        self.embedding_path = self.args.embedding_path
        self.word_embedding_dim = self.args.word_embedding_dim
        self.unknown_word = "**UNK**"
        self.conf = self.args

        for key, value in self.kgs.kg1.entities_id_dict.items():
            temp = key.split("/")[-1].lower()
            temp = temp.replace("(", "")
            temp = temp.replace(")", "")
            self.id_dict1[int(value)] = temp.replace("_", " ")
            # self.id_dict1[int(value)] = key.split("/")[-1].replace("_", " ")
        for key, value in self.kgs.kg2.entities_id_dict.items():
            temp = key.split("/")[-1].lower()
            temp = temp.replace("(", "")
            temp = temp.replace(")", "")
            self.id_dict2[int(value)] = temp.replace("_", " ")
            # self.id_dict2[int(value)] = key.split("/")[-1].replace("_", " ")
        for item in self.kgs.train_links:
            self.assign_gen[int(item[0])] = int(item[1])
        for item in self.kgs.valid_links:
            self.assign_gen[int(item[0])] = int(item[1])
        for item in self.kgs.test_links:
            self.assign_gen[int(item[0])] = int(item[1])

    def extract_sub_graph(self, id_list, id_features, fw_graph):
        g_ids_features_list = []
        g_adj_list = []
        for _ in range(len(id_list)):
            id = id_list[_]
            g_ids_features = {}
            g_adj = {}
            id_mapping = {}
            hop_1_ids = [id]

            if id in fw_graph:
                fw_neighbor_info = fw_graph[id]
                for (rel_id, obj_id) in fw_neighbor_info:
                    if obj_id not in hop_1_ids:
                        hop_1_ids.append(obj_id)

            for subj_id in hop_1_ids:
                if subj_id not in id_mapping:
                    id_mapping[subj_id] = len(id_mapping)

                subj_mapped_id = id_mapping[subj_id]

                if subj_mapped_id not in g_ids_features:
                    feature = id_features[subj_id]
                    g_ids_features[subj_mapped_id] = feature

                if subj_mapped_id not in g_adj:
                    g_adj[subj_mapped_id] = []

                if subj_id in fw_graph:
                    fw_neighbor_info = fw_graph[subj_id]
                    for (rel_id, obj_id) in fw_neighbor_info:
                        if obj_id not in hop_1_ids:
                            continue

                        if obj_id not in id_mapping:
                            id_mapping[obj_id] = len(id_mapping)
                        obj_mapped_id = id_mapping[obj_id]

                        if obj_mapped_id not in g_adj[subj_mapped_id]:
                            g_adj[subj_mapped_id].append(obj_mapped_id)

                        if obj_mapped_id not in g_ids_features:
                            feature = id_features[obj_id]
                            g_ids_features[obj_mapped_id] = feature

            g_ids_features_list.append(g_ids_features)
            g_adj_list.append(g_adj)

        return g_ids_features_list, g_adj_list

    def can_gen(self, method):
        def if_zero_vec(zvec):
            tag = True
            for _ in range(len(zvec)):
                if zvec[_] != 0.0:
                    tag = False
                    break
            return tag

        w_vec = {}

        with codecs.open(self.args.embedding_path, "r", encoding="utf-8") as emb_fr:
            lines = emb_fr.readlines()
            for line in lines:
                info = line.strip().split(" ")
                w = info[0]
                vec = [float(x) for x in info[1:]]
                w_vec[w] = vec
        id_1_vec = {}
        zero_vec_id1 = set()
        for key, value in self.id_dict1.items():
            vec = [0.0 for _ in range(300)]
            split_value = value.split(" ")
            for item in split_value:
                item = item.lower()
                if item in w_vec:
                    temp_v = w_vec[item]
                    vec = [vec[_] + temp_v[_] for _ in range(300)]
            vec = [vec[_] for _ in range(300)]
            id_1_vec[key] = vec
            if if_zero_vec(vec):
                zero_vec_id1.add(key)

        id_2_vec = {}
        zero_vec_id2 = set()
        for key, value in self.id_dict2.items():
            vec = [0.0 for _ in range(300)]
            split_value = value.split(" ")
            for item in split_value:
                if item in w_vec:
                    temp_v = w_vec[item]
                    vec = [vec[_] + temp_v[_] for _ in range(300)]
            vec = [vec[_] for _ in range(300)]
            id_2_vec[key] = vec
            if if_zero_vec(vec):
                zero_vec_id2.add(key)
        real_links = None
        if method == "train":
            real_links = self.kgs.train_links
        elif method == "dev":
            real_links = self.kgs.valid_links
        elif method == "test":
            real_links = self.kgs.test_links
        # *********************************add multiprocessing*****************************
        final_result = []
        pool_result = []
        tasks = div_list(real_links, 8)
        pool = multiprocessing.Pool(processes=8)
        for task in tasks:
            pool_result.append(pool.apply_async(find_near,
                                                (task, id_1_vec, id_2_vec, zero_vec_id1,
                                                 zero_vec_id2, self.args.cand_size)))
        pool.close()
        pool.join()
        for item in pool_result:
            temp_pool_result = item.get()
            final_result.append(temp_pool_result)

        # *********************************************************************************
        with open(self.args.training_data + method + "_cand_list" + str(self.args.cand_size), "w",
                  encoding="utf-8") as f:
            for item in final_result:
                for key in item.keys():
                    f.write(str(key) + ": ")
                    for value in range(len(item[key])):
                        f.write(str(item[key][value]) + " ")
                    f.write('\n')

    def gen_data(self, method):
        real_links = None
        if method == "train":
            real_links = self.kgs.train_links
        elif method == "dev":
            real_links = self.kgs.valid_links
        elif method == "test":
            real_links = self.kgs.test_links

        with codecs.open(self.args.training_data + method + "_cand_list" + str(self.args.cand_size),
                         "r", encoding="utf-8") as f, codecs.open(self.args.training_data
                                                                  + method + "_examples" + str(self.args.cand_size),
                                                                  "w", encoding="utf-8") as fw:
            id_1_2_map = {}
            for item in real_links:
                id_1_2_map[item[0]] = item[1]
            lines = f.readlines()
            for line in lines:
                info = line.strip().split(":")
                id_1 = int(info[0])
                gold_id_2 = id_1_2_map[id_1]
                cands = info[1].strip().split(" ")
                cands = cands[1:self.args.cand_size]
                fw.write(str(id_1) + "\t" + str(self.assign_gen[id_1]) + "\t" + str(1) + "\n")
                for cand in cands:
                    cand = int(cand)
                    label = 0
                    if cand == gold_id_2:
                        label = 1
                    fw.write(str(id_1) + "\t" + str(cand) + "\t" + str(label) + "\n")

    def build_graph(self, triple_file_path):
        with codecs.open(triple_file_path, 'r', 'utf-8') as f:
            fw_graph = {}
            bw_graph = {}
            lines = f.readlines()
            for line in lines:
                info = line.strip().split("\t")
                subj_id = int(info[0])
                rel_id = int(info[1])
                obj_id = int(info[2])

                if subj_id not in fw_graph:
                    fw_graph[subj_id] = []
                fw_graph[subj_id].append((rel_id, obj_id))

                if obj_id not in bw_graph:
                    bw_graph[obj_id] = []
                bw_graph[obj_id].append((rel_id, subj_id))

            return fw_graph, bw_graph

    def extract_node_feature(self, feature_path):
        with codecs.open(feature_path, 'r', 'utf-8') as f:
            res = {}
            lines = f.readlines()
            for line in lines:
                info = line.strip().split('\t')
                id = int(info[0])
                if len(info) != 2:
                    feature = "**UNK**"
                else:
                    feature = info[1].lower()
                res[id] = feature
            return res

    def gen_graph(self, relation_list, kg_features):
        fw_graph = dict()
        bw_graph = dict()
        features = kg_features
        for item in relation_list:
            if int(item[0]) not in fw_graph:
                fw_graph[int(item[0])] = []
            fw_graph[int(item[0])].append((int(item[1]), int(item[2])))
            if int(item[2]) not in bw_graph:
                bw_graph[int(item[2])] = []
            bw_graph[int(item[2])].append((int(item[1]), int(item[0])))

        id_list = list(features.keys())
        features_list, adj_list = self.extract_sub_graph(id_list, features, fw_graph)
        res_map = {}
        for _ in range(len(id_list)):
            id = id_list[_]
            feature = features_list[_]
            adj = adj_list[_]
            jo = {}
            jo['g_ids_features'] = feature
            jo['g_adj'] = adj
            res_map[id] = jo
        return res_map

    def read_data(self, input_path, word_idx, if_increase_dct):
        g1_map = self.gen_graph(self.kgs.kg1.relation_triples_list, self.id_dict1)
        g2_map = self.gen_graph(self.kgs.kg2.relation_triples_list, self.id_dict2)
        graphs_1 = []
        graphs_2 = []
        labels = []
        with open(input_path, 'r') as fr:
            lines = fr.readlines()
            for _ in range(len(lines)):
                line = lines[_].strip()
                info = line.split("\t")
                id_1 = int(info[0])
                id_2 = int(info[1])
                label = int(info[2])

                graph_1 = g1_map[id_1]
                graph_2 = g2_map[id_2]

                graph_1['g_id'] = id_1
                graph_2['g_id'] = id_2

                graphs_1.append(graph_1)
                graphs_2.append(graph_2)
                labels.append(label)

                if if_increase_dct:
                    features = [graph_1['g_ids_features'], graph_2['g_ids_features']]
                    for f in features:
                        for id in f:
                            for w in f[id].split():
                                if w not in word_idx:
                                    word_idx[w] = len(word_idx) + 1

        return graphs_1, graphs_2, labels

    def batch_graph(self, graphs):
        g_ids_features = {}
        g_fw_adj = {}
        g_bw_adj = {}
        g_nodes = []

        for g in graphs:
            id_adj = g['g_adj']
            features = g['g_ids_features']

            nodes = []

            # we first add all nodes into batch_graph and create a mapping from graph id to batch_graph id, this mapping will be
            # used in the creation of fw_adj and bw_adj

            id_gid_map = {}
            offset = len(g_ids_features.keys())
            for id in features.keys():
                id = int(id)
                g_ids_features[offset + id] = features[id]
                id_gid_map[id] = offset + id
                nodes.append(offset + id)
            g_nodes.append(nodes)

            for id in id_adj:
                adj = id_adj[id]
                id = int(id)
                g_id = id_gid_map[id]
                if g_id not in g_fw_adj:
                    g_fw_adj[g_id] = []
                for t in adj:
                    t = int(t)
                    g_t = id_gid_map[t]
                    g_fw_adj[g_id].append(g_t)
                    if g_t not in g_bw_adj:
                        g_bw_adj[g_t] = []
                    g_bw_adj[g_t].append(g_id)

        node_size = len(g_ids_features.keys())
        for id in range(node_size):
            if id not in g_fw_adj:
                g_fw_adj[id] = []
            if id not in g_bw_adj:
                g_bw_adj[id] = []

        graph = {}
        graph['g_ids_features'] = g_ids_features
        graph['g_nodes'] = g_nodes
        graph['g_fw_adj'] = g_fw_adj
        graph['g_bw_adj'] = g_bw_adj

        return graph

    def vectorize_label(self, labels):
        lv = []
        for label in labels:
            if label == 0 or label == '0':
                lv.append([1, 0])
            elif label == 1 or label == '1':
                lv.append([0, 1])
            else:
                print("error in vectoring the label")
        lv = np.array(lv)
        return lv

    def vectorize_batch_graph(self, graph, word_idx):
        # vectorize the graph feature and normalize the adj info
        id_features = graph['g_ids_features']
        gv = {}
        nv = []
        n_len_v = []
        word_max_len = 0
        for id in id_features:
            feature = id_features[id]
            word_max_len = max(word_max_len, len(feature.split()))
        # word_max_len = min(word_max_len, conf.word_size_max)

        for id in graph['g_ids_features']:
            feature = graph['g_ids_features'][id]
            fv = []
            for token in feature.split():
                if len(token) == 0:
                    continue
                if token in word_idx:
                    fv.append(word_idx[token])
                else:
                    fv.append(word_idx[self.args.unknown_word])

            if len(fv) > word_max_len:
                n_len_v.append(word_max_len)
            else:
                n_len_v.append(len(fv))

            for _ in range(word_max_len - len(fv)):
                fv.append(0)
            fv = fv[:word_max_len]
            nv.append(fv)

        # add an all-zero vector for the PAD node
        nv.append([0 for _ in range(word_max_len)])
        n_len_v.append(0)

        gv['g_ids_features'] = np.array(nv)
        gv['g_ids_feature_lens'] = np.array(n_len_v)

        # ============== vectorize adj info ======================
        g_fw_adj = graph['g_fw_adj']
        g_fw_adj_v = []

        degree_max_size = 0
        for id in g_fw_adj:
            degree_max_size = max(degree_max_size, len(g_fw_adj[id]))
        g_bw_adj = graph['g_bw_adj']
        for id in g_bw_adj:
            degree_max_size = max(degree_max_size, len(g_bw_adj[id]))
        degree_max_size = min(degree_max_size, self.args.sample_size_per_layer)

        for id in g_fw_adj:
            adj = g_fw_adj[id]
            for _ in range(degree_max_size - len(adj)):
                adj.append(len(g_fw_adj.keys()))
            adj = adj[:degree_max_size]
            assert len(adj) == degree_max_size
            g_fw_adj_v.append(adj)

        # PAD node directs to the PAD node
        g_fw_adj_v.append([len(g_fw_adj.keys()) for _ in range(degree_max_size)])

        g_bw_adj_v = []
        for id in g_bw_adj:
            adj = g_bw_adj[id]
            for _ in range(degree_max_size - len(adj)):
                adj.append(len(g_bw_adj.keys()))
            adj = adj[:degree_max_size]
            assert len(adj) == degree_max_size
            g_bw_adj_v.append(adj)

        # PAD node directs to the PAD node
        g_bw_adj_v.append([len(g_bw_adj.keys()) for _ in range(degree_max_size)])

        # ============== vectorize nodes info ====================
        g_nodes = graph['g_nodes']
        graph_max_size = 0
        for nodes in g_nodes:
            graph_max_size = max(graph_max_size, len(nodes))

        g_node_v = []
        g_node_mask = []
        entity_index = []
        for nodes in g_nodes:
            mask = [1 for _ in range(len(nodes))]
            for _ in range(graph_max_size - len(nodes)):
                nodes.append(len(g_fw_adj.keys()))
                mask.append(0)
            nodes = nodes[:graph_max_size]
            mask = mask[:graph_max_size]
            g_node_v.append(nodes)
            g_node_mask.append(mask)
            entity_index.append(0)

        g_looking_table = []
        global_count = 0
        for mask in g_node_mask:
            for item in mask:
                if item == 1:
                    g_looking_table.append(global_count)
                global_count += 1

        gv['g_nodes'] = np.array(g_node_v)
        gv['g_bw_adj'] = np.array(g_bw_adj_v)
        gv['g_fw_adj'] = np.array(g_fw_adj_v)
        gv['g_mask'] = np.array(g_node_mask)
        gv['g_looking_table'] = np.array(g_looking_table)
        gv['entity_index'] = entity_index

        return gv

    def feed_dict(self, g1_v_batch, g2_v_batch, label_v_batch):
        dict = {}
        dict['fw_adj_info_first'] = g1_v_batch['g_fw_adj']
        dict['bw_adj_info_first'] = g1_v_batch['g_bw_adj']
        dict['feature_info_first'] = g1_v_batch['g_ids_features']
        dict['feature_len_first'] = g1_v_batch['g_ids_feature_lens']
        dict['batch_nodes_first'] = g1_v_batch['g_nodes']
        dict['batch_mask_first'] = g1_v_batch['g_mask']
        dict['looking_table_first'] = g1_v_batch['g_looking_table']

        dict['fw_adj_info_second'] = g2_v_batch['g_fw_adj']
        dict['bw_adj_info_second'] = g2_v_batch['g_bw_adj']
        dict['feature_info_second'] = g2_v_batch['g_ids_features']
        dict['feature_len_second'] = g2_v_batch['g_ids_feature_lens']
        dict['batch_nodes_second'] = g2_v_batch['g_nodes']
        dict['batch_mask_second'] = g2_v_batch['g_mask']
        dict['looking_table_second'] = g2_v_batch['g_looking_table']

        dict['y'] = label_v_batch
        return dict

    def construct_embedding(self, method):
        construct_list = list()
        if method == "dev" or method == "valid":
            construct_list = self.kgs.valid_links
        elif method == "test":
            construct_list = self.kgs.test_links
        elif method == "kg1":
            kg2_len = len(self.kgs.kg2.entities_list)
            for i in range(len(self.kgs.kg1.entities_list)):
                temp = [self.kgs.kg1.entities_list[i], self.kgs.kg2.entities_list[i % kg2_len]]
                construct_list.append(temp)
        elif method == "kg2":
            kg1_len = len(self.kgs.kg1.entities_list)
            for i in range(len(self.kgs.kg2.entities_list)):
                temp = [self.kgs.kg2.entities_list[i], self.kgs.kg1.entities_list[i % kg1_len]]
                construct_list.append(temp)
        max_len = 0
        dev1 = list()
        dev2 = list()
        for item in construct_list:
            words = self.id_dict1[item[0]]
            temp_dev1 = list()
            temp_words = words.split()
            max_len = max(max_len, len(temp_words))
            for i in range(len(temp_words)):
                word = temp_words[i].lower()
                if word in self.word_idx.keys():
                    temp_dev1.append(self.word_idx[word])
                else:
                    temp_dev1.append(0)
            dev1.append(temp_dev1)

            words = self.id_dict2[item[1]]
            temp_dev2 = list()
            temp_words = words.split()
            max_len = max(max_len, len(temp_words))
            for i in range(len(temp_words)):
                word = temp_words[i].lower()
                if word in self.word_idx.keys():
                    temp_dev2.append(self.word_idx[word])
                else:
                    temp_dev2.append(0)
            dev2.append(temp_dev2)
        dev1_len = list()
        dev2_len = list()
        for i in range(len(dev1)):
            if len(dev1[i]) < max_len:
                dev1_len.append(len(dev1[i]))
                for j in range(max_len - len(dev1[i])):
                    dev1[i].append(0)
            else:
                dev1_len.append(max_len)
            if len(dev2[i]) < max_len:
                dev2_len.append(len(dev2[i]))
                for j in range(max_len - len(dev2[i])):
                    dev2[i].append(0)
            else:
                dev2_len.append(max_len)
        self.save_g1_batch['g_ids_features'] = np.array(dev1)
        self.save_g2_batch['g_ids_features'] = np.array(dev2)
        self.save_g1_batch['g_ids_feature_lens'] = np.array(dev1_len)
        self.save_g2_batch['g_ids_feature_lens'] = np.array(dev2_len)
        dict = self.feed_dict(self.save_g1_batch, self.save_g2_batch, self.save_label_batch)
        return dict

    def train_embeddings(self, ):
        train_data_path = self.args.training_data + "train_examples" + str(self.args.cand_size)
        dev_data_path = self.args.training_data + "dev_examples" + str(self.args.cand_size)
        test_data_path = self.args.training_data + "test_examples" + str(self.args.cand_size)
        graphs_1_train, graphs_2_train, labels_train = self.read_data(train_data_path, self.word_idx, True)
        graphs_1_dev, graphs_2_dev, labels_dev = self.read_data(dev_data_path, self.word_idx, True)
        self.graphs_1_test, self.graphs_2_test, self.labels_test = self.read_data(test_data_path, self.word_idx, True)

        self.args.word_vocab_size = len(self.word_idx)
        self.args.pretrained_word_size = self.pretraind_word_size
        self.args.learned_word_size = len(self.word_idx) - self.pretraind_word_size

        model = GraphMatchNN("train", self.args, self.pretraind_word_embedding)
        model._build_graph()
        self.saver = tf.train.Saver(max_to_keep=None)
        self.temp_model = model
        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)
        g1_v_batch = None
        g2_v_batch = None
        label_v_batch = None
        for i in range(1, self.args.max_epoch + 1):
            start_time = time.time()
            n_train = len(graphs_1_train)
            temp_order = list(range(n_train))
            np.random.shuffle(temp_order)

            loss_sum = 0.0
            for start in range(0, n_train, self.args.batch_size):
                end = min(start + self.args.batch_size, n_train)
                graphs_1 = []
                graphs_2 = []
                labels = []
                for _ in range(start, end):
                    idx = temp_order[_]
                    graphs_1.append(graphs_1_train[idx])
                    graphs_2.append(graphs_2_train[idx])
                    labels.append(labels_train[idx])

                batch_graph_1 = self.batch_graph(graphs_1)
                batch_graph_2 = self.batch_graph(graphs_2)

                g1_v_batch = self.vectorize_batch_graph(batch_graph_1, self.word_idx)
                g2_v_batch = self.vectorize_batch_graph(batch_graph_2, self.word_idx)
                label_v_batch = self.vectorize_label(labels)
                train_dict = self.feed_dict(g1_v_batch, g2_v_batch, label_v_batch)

                watch, _, batch_loss = model.act(self.session, "train", train_dict, False, False)
                loss_sum += batch_loss

            print('epoch {}, avg. relation triple loss: {:.4f}, cost time: {:.4f}s'.format(i, loss_sum,
                                                                                           time.time() - start_time))

            # ********************no early stop********************************************
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                self.save_g1_batch = g1_v_batch
                self.save_g2_batch = g2_v_batch
                self.save_label_batch = label_v_batch
                flag = self.valid_(self.args.stop_metric, model, graphs_1_dev, graphs_2_dev, labels_dev)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break

        return

    def test(self, save=True):
        t = time.time()
        n_dev = len(self.graphs_1_test)
        dev_batch_size = self.args.dev_batch_size
        golds = []
        predicted_res = []
        g1_ori_ids = []
        g2_ori_ids = []
        for start in range(0, n_dev, dev_batch_size):
            end = min(start + dev_batch_size, n_dev)
            graphs_1 = []
            graphs_2 = []
            labels = []
            for _ in range(start, end):
                graphs_1.append(self.graphs_1_test[_])
                graphs_2.append(self.graphs_2_test[_])
                labels.append(self.labels_test[_])
                golds.append(self.labels_test[_])

                g1_ori_ids.append(self.graphs_1_test[_]['g_id'])
                g2_ori_ids.append(self.graphs_2_test[_]['g_id'])

            batch_graph_1 = self.batch_graph(graphs_1)
            batch_graph_2 = self.batch_graph(graphs_2)

            g1_v_batch = self.vectorize_batch_graph(batch_graph_1, self.word_idx)
            g2_v_batch = self.vectorize_batch_graph(batch_graph_2, self.word_idx)
            label_v_batch = self.vectorize_label(labels)

            dev_dict = self.feed_dict(g1_v_batch, g2_v_batch, label_v_batch)
            predicted = self.temp_model.act(self.session, "train", dev_dict, True, False)[0]
            for _ in range(0, end - start):
                predicted_res.append(predicted[_][1])  # add the prediction result into the bag
        count = 0.0
        correct_50 = 0.0
        correct_10 = 0.0
        correct_5 = 0.0
        correct_1 = 0.0
        mr = 0.0
        mrr = 0.00
        cand_size = self.args.cand_size
        assert len(predicted_res) % cand_size == 0
        assert len(predicted_res) == len(g1_ori_ids)
        assert len(g1_ori_ids) == len(g2_ori_ids)
        number = int(len(predicted_res) / cand_size)
        incorrect_pairs = []
        for _ in range(number):
            idx_score = {}
            for idx in range(cand_size):
                idx_score[_ * cand_size + idx] = predicted_res[_ * cand_size + idx]
            idx_score_items = idx_score.items()
            idx_score_items = sorted(idx_score_items, key=lambda d: d[1], reverse=True)

            id_1 = g1_ori_ids[_ * cand_size]
            id_2 = g2_ori_ids[_ * cand_size]
            for sub_idx in range(min(100, len(idx_score_items))):
                idx = idx_score_items[sub_idx][0]
                if golds[idx] == 1:
                    mr += (sub_idx + 1)
                    mrr += 1 / (sub_idx + 1)
                    if sub_idx < 50:
                        correct_50 += 1
                    if sub_idx < 10:
                        correct_10 += 1
                    if sub_idx < 5:
                        correct_5 += 1
                    if sub_idx == 0:
                        correct_1 += 1.0
                    else:
                        incorrect_pairs.append((id_1, id_2))
                    break
            count += 1.0
        mr /= count
        mrr /= count
        hit50 = correct_50 / count * 100
        hit10 = correct_10 / count * 100
        hit5 = correct_5 / count * 100
        hit1 = correct_1 / count * 100
        top_k = [1, 5, 10, 50]
        hits = [hit1, hit5, hit10, hit50]
        cost = time.time() - t
        print("accurate results: hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, time = {:.3f} s ".
              format(top_k, hits, mr, mrr, cost))

    def save(self):
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)
        self.saver.save(self.session, self.out_folder + "gmnn_model", global_step=0)
        print("Results saved!")

    def valid_(self, stop_metric, model, graphs_1_dev, graphs_2_dev, labels_dev):
        t = time.time()
        n_dev = len(graphs_1_dev)
        dev_batch_size = self.args.dev_batch_size
        golds = []
        predicted_res = []
        g1_ori_ids = []
        g2_ori_ids = []
        for start in range(0, n_dev, dev_batch_size):
            end = min(start + dev_batch_size, n_dev)
            graphs_1 = []
            graphs_2 = []
            labels = []
            for _ in range(start, end):
                graphs_1.append(graphs_1_dev[_])
                graphs_2.append(graphs_2_dev[_])
                labels.append(labels_dev[_])
                golds.append(labels_dev[_])

                g1_ori_ids.append(graphs_1_dev[_]['g_id'])
                g2_ori_ids.append(graphs_2_dev[_]['g_id'])

            batch_graph_1 = self.batch_graph(graphs_1)
            batch_graph_2 = self.batch_graph(graphs_2)

            g1_v_batch = self.vectorize_batch_graph(batch_graph_1, self.word_idx)
            g2_v_batch = self.vectorize_batch_graph(batch_graph_2, self.word_idx)
            label_v_batch = self.vectorize_label(labels)

            dev_dict = self.feed_dict(g1_v_batch, g2_v_batch, label_v_batch)
            predicted = model.act(self.session, "train", dev_dict, True, False)[0]
            for _ in range(0, end - start):
                predicted_res.append(predicted[_][1])  # add the prediction result into the bag
        count = 0.0
        correct_50 = 0.0
        correct_10 = 0.0
        correct_5 = 0.0
        correct_1 = 0.0

        mr = 0.0
        mrr = 0.00
        cand_size = self.args.cand_size
        assert len(predicted_res) % cand_size == 0
        assert len(predicted_res) == len(g1_ori_ids)
        assert len(g1_ori_ids) == len(g2_ori_ids)
        number = int(len(predicted_res) / cand_size)
        incorrect_pairs = []
        for _ in range(number):
            idx_score = {}
            for idx in range(cand_size):
                idx_score[_ * cand_size + idx] = predicted_res[_ * cand_size + idx]
            idx_score_items = idx_score.items()
            idx_score_items = sorted(idx_score_items, key=lambda d: d[1], reverse=True)

            id_1 = g1_ori_ids[_ * cand_size]
            id_2 = g2_ori_ids[_ * cand_size]
            for sub_idx in range(min(100, len(idx_score_items))):
                idx = idx_score_items[sub_idx][0]
                if golds[idx] == 1:
                    mr += (sub_idx + 1)
                    mrr += 1 / (sub_idx + 1)
                    if sub_idx < 50:
                        correct_50 += 1
                    if sub_idx < 10:
                        correct_10 += 1
                    if sub_idx < 5:
                        correct_5 += 1
                    if sub_idx == 0:
                        correct_1 += 1.0
                    else:
                        incorrect_pairs.append((id_1, id_2))
                    break
            count += 1.0
        mr /= count
        mrr /= count
        hit50 = correct_50 / count * 100
        hit10 = correct_10 / count * 100
        hit5 = correct_5 / count * 100
        hit1 = correct_1 / count * 100
        top_k = [1, 5, 10, 50]
        hits = [hit1, hit5, hit10, hit50]
        cost = time.time() - t
        print("quick results: hits@{} = {}%, time = {:.3f} s ".format(top_k, hits, cost))

        if stop_metric == 'hits1':
            return hit1
        return mrr

    def run(self):
        if self.use_pretrained_embedding:
            self.pretraind_word_embedding = load_word_embedding(self.embedding_path, self.word_idx)
            self.pretraind_word_size = len(self.pretraind_word_embedding)
            self.args.hidden_layer_dim = self.args.pretrained_word_embedding_dim
        self.word_idx[self.unknown_word] = len(self.word_idx.keys()) + 1
        if self.args.build_train_examples:
            self.can_gen("train")
            self.gen_data("train")
            self.can_gen("dev")
            self.gen_data("dev")
            self.can_gen("test")
            self.gen_data("test")
        t = time.time()
        self.train_embeddings()
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
