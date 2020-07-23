import math
import multiprocessing as mp
import random
import time
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.sparse import csr_matrix

from openea.models.trans.transe import BasicModel
from openea.modules.utils.util import load_session
from openea.modules.finding.evaluation import early_stop


class BasicReader(object):
    def read(self, data_path='data/dbp_wd_15k_V1/mapping/0_3/'):
        # add shortcuts
        kgs = self.kgs

        kg1 = pd.DataFrame(kgs.kg1.relation_triples_list, columns=['h_id', 'r_id', 't_id'])
        kg2 = pd.DataFrame(kgs.kg2.relation_triples_list, columns=['h_id', 'r_id', 't_id'])

        kb = pd.concat([kg1, kg2], ignore_index=True)

        # self._eid_1 = pd.Series(eid_1)
        # self._eid_2 = pd.Series(eid_2)

        self._ent_num = kgs.entities_num
        self._rel_num = kgs.relations_num
        # self._ent_id = e_map
        # self._rel_id = r_map
        self._ent_mapping = pd.DataFrame(list(kgs.train_links), columns=['kb_1', 'kb_2'])
        self._rel_mapping = pd.DataFrame({}, columns=['kb_1', 'kb_2'])
        self._ent_testing = pd.DataFrame(list(kgs.test_links), columns=['kb_1', 'kb_2'])
        self._rel_testing = pd.DataFrame({}, columns=['kb_1', 'kb_2'])

        # add reverse edges
        rev_kb = kb[['t_id', 'r_id', 'h_id']].values
        rev_kb[:, 1] += self._rel_num
        rev_kb = pd.DataFrame(rev_kb, columns=['h_id', 'r_id', 't_id'])
        self._rel_num *= 2
        kb = pd.concat([kb, rev_kb], ignore_index=True)
        # print(kb)
        # print(kb[len(kb)//2:])

        self._kb = kb
        # we first tag the entities that have algined entities according to entity_mapping
        self.add_align_infor()
        # we then connect two KGs by creating new triples involving aligned entities.
        self.add_weight()

    def add_align_infor(self):
        kb = self._kb

        ent_mapping = self._ent_mapping
        rev_e_m = ent_mapping.rename(columns={'kb_1': 'kb_2', 'kb_2': 'kb_1'})
        rel_mapping = self._rel_mapping
        rev_r_m = rel_mapping.rename(columns={'kb_1': 'kb_2', 'kb_2': 'kb_1'})

        ent_mapping = pd.concat([ent_mapping, rev_e_m], ignore_index=True)
        rel_mapping = pd.concat([rel_mapping, rev_r_m], ignore_index=True)

        ent_mapping = pd.Series(ent_mapping.kb_2.values, index=ent_mapping.kb_1.values)
        rel_mapping = pd.Series(rel_mapping.kb_2.values, index=rel_mapping.kb_1.values)

        self._e_m = ent_mapping
        self._r_m = rel_mapping

        kb['ah_id'] = kb.h_id
        kb['ar_id'] = kb.r_id
        kb['at_id'] = kb.t_id

        h_mask = kb.h_id.isin(ent_mapping)
        r_mask = kb.r_id.isin(rel_mapping)
        t_mask = kb.t_id.isin(ent_mapping)

        kb['ah_id'][h_mask] = ent_mapping.loc[kb['ah_id'][h_mask].values]
        kb['ar_id'][r_mask] = rel_mapping.loc[kb['ar_id'][r_mask].values]
        kb['at_id'][t_mask] = ent_mapping.loc[kb['at_id'][t_mask].values]

        self._kb = kb

    def add_weight(self):
        kb = self._kb[['h_id', 'r_id', 't_id', 'ah_id', 'ar_id', 'at_id']]

        kb['w_h'] = 0
        kb['w_r'] = 0
        kb['w_t'] = 0

        h_mask = ~(kb.h_id == kb.ah_id)
        r_mask = ~(kb.r_id == kb.ar_id)
        t_mask = ~(kb.t_id == kb.at_id)

        kb.loc[h_mask, 'w_h'] = 1
        kb.loc[r_mask, 'w_r'] = 1
        kb.loc[t_mask, 'w_t'] = 1

        akb = kb[['ah_id', 'ar_id', 'at_id', 'w_h', 'w_r', 'w_t']]
        akb = akb.rename(columns={'ah_id': 'h_id', 'ar_id': 'r_id', 'at_id': 't_id'})

        ahkb = kb[h_mask][['ah_id', 'r_id', 't_id', 'w_h', 'w_r', 'w_t']].rename(columns={'ah_id': 'h_id'})
        arkb = kb[r_mask][['h_id', 'ar_id', 't_id', 'w_h', 'w_r', 'w_t']].rename(columns={'ar_id': 'r_id'})
        atkb = kb[t_mask][['h_id', 'r_id', 'at_id', 'w_h', 'w_r', 'w_t']].rename(columns={'at_id': 't_id'})
        ahrkb = kb[h_mask & r_mask][['ah_id', 'ar_id', 't_id', 'w_h', 'w_r', 'w_t']].rename(
            columns={'ah_id': 'h_id', 'ar_id': 'r_id'})
        ahtkb = kb[h_mask & t_mask][['ah_id', 'r_id', 'at_id', 'w_h', 'w_r', 'w_t']].rename(
            columns={'ah_id': 'h_id', 'at_id': 't_id'})
        artkb = kb[r_mask & t_mask][['h_id', 'ar_id', 'at_id', 'w_h', 'w_r', 'w_t']].rename(
            columns={'ar_id': 'r_id', 'at_id': 't_id'})
        ahrtkb = kb[h_mask & r_mask & t_mask][['ah_id', 'ar_id', 'at_id', 'w_h', 'w_r', 'w_t']].rename(
            columns={'ah_id': 'h_id',
                     'ar_id': 'r_id',
                     'at_id': 't_id'})

        kb['w_h'] = 0
        kb['w_r'] = 0
        kb['w_t'] = 0

        kb = pd.concat(
            [akb, ahkb, arkb, atkb, ahrkb, ahtkb, artkb, ahrtkb, kb[['h_id', 'r_id', 't_id', 'w_h', 'w_r', 'w_t']]],
            ignore_index=True).drop_duplicates()

        self._kb = kb.reset_index(drop=True)


# sampler
class BasicSampler(object):
    def sample_paths(self, repeat_times=2):
        opts = self._options

        kb = self._kb.copy()

        kb = kb[['h_id', 'r_id', 't_id']]

        # sampling triples with the h_id-(r_id,t_id) form.

        rtlist = np.unique(kb[['r_id', 't_id']].values, axis=0)

        rtdf = pd.DataFrame(rtlist, columns=['r_id', 't_id'])

        rtdf = rtdf.reset_index().rename({'index': 'tail_id'}, axis='columns')

        rtkb = kb.merge(
            rtdf, left_on=['r_id', 't_id'], right_on=['r_id', 't_id'])

        htail = np.unique(rtkb[['h_id', 'tail_id']].values, axis=0)

        htailmat = csr_matrix((np.ones(len(htail)), (htail[:, 0], htail[:, 1])),
                              shape=(self._ent_num, rtlist.shape[0]))

        # calulate corss-KG bias at first
        em = pd.concat(
            [self._ent_mapping.kb_1, self._ent_mapping.kb_2]).values

        rtkb['across'] = rtkb.t_id.isin(em)
        rtkb.loc[rtkb.across, 'across'] = opts.beta
        rtkb.loc[rtkb.across == 0, 'across'] = 1 - opts.beta

        rtailkb = rtkb[['h_id', 't_id', 'tail_id', 'across']]

        def gen_tail_dict(x):
            return x.tail_id.values, x.across.values / x.across.sum()

        rtailkb = rtailkb.groupby('h_id').apply(gen_tail_dict)

        rtailkb = pd.DataFrame({'tails': rtailkb})

        # start sampling

        hrt = np.repeat(kb.values, repeat_times, axis=0)

        # for starting triples
        def perform_random(x):
            return np.random.choice(x.tails[0], 1, p=x.tails[1].astype(np.float))

        # else
        def perform_random2(x):
            # calculate depth bias
            pre_c = htailmat[np.repeat(x.pre, x.tails[0].shape[0]), x.tails[0]]
            pre_c[pre_c == 0] = opts.alpha
            pre_c[pre_c == 1] = 1 - opts.alpha
            p = x.tails[1].astype(np.float).reshape(
                [-1, ]) * pre_c.A.reshape([-1, ])
            p = p / p.sum()
            return np.random.choice(x.tails[0], 1, p=p)

        # print(rtailkb.loc[hrt[:, 2]])
        rt_x = rtailkb.loc[hrt[:, 2]].apply(perform_random, axis=1)
        rt_x = rtlist[np.concatenate(rt_x.values)]

        rts = [hrt, rt_x]
        print('hrt', 'rt_x', len(hrt), len(rt_x))
        c_length = 5
        while c_length < opts.max_length:
            curr = rtailkb.loc[rt_x[:, 1]]
            print(len(curr), len(hrt[:, 0]))
            curr.loc[:, 'pre'] = hrt[:, 0]

            rt_x = curr.apply(perform_random2, axis=1)
            rt_x = rtlist[np.concatenate(rt_x.values)]

            rts.append(rt_x)
            c_length += 2

        data = np.concatenate(rts, axis=1)
        data = pd.DataFrame(data)

        self._train_data = data
        # print("save paths to:", '%spaths_%.1f_%.1f' % (opts.data_path, opts.alpha, opts.beta))
        # data.to_csv('%spaths_%.1f_%.1f' % (opts.data_path, opts.alpha, opts.beta))


class RSN4EA(BasicReader, BasicSampler, BasicModel):
    def __init__(self):
        super().__init__()

    def init(self):
        self._options = opts = self.args
        opts.data_path = opts.training_data

        self.read(data_path=self._options.data_path)

        sequence_datapath = '%spaths_%.1f_%.1f' % (
            self._options.data_path, self._options.alpha, self._options.beta)

        if not os.path.exists(sequence_datapath):
            self.sample_paths()
        else:
            print('load existing training sequences')
            self._train_data = pd.read_csv('%spaths_%.1f_%.1f' % (
                self._options.data_path, self._options.alpha, self._options.beta), index_col=0)

        self._define_variables()
        self._define_embed_graph()
        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)

    def _define_variables(self):
        options = self._options
        hidden_size = options.hidden_size

        self._entity_embedding = tf.get_variable(
            'entity_embedding',
            [self._ent_num, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer(uniform=False)
        )
        self._relation_embedding = tf.get_variable(
            'relation_embedding',
            [self._rel_num, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer(uniform=False)
        )

        self.ent_embeds, self.rel_embeds = self._entity_embedding, self._relation_embedding

        self._rel_w = tf.get_variable(
            "relation_softmax_w",
            [self._rel_num, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer(uniform=False)
        )
        self._rel_b = tf.get_variable(
            "relation_softmax_b",
            [self._rel_num],
            initializer=tf.constant_initializer(0)
        )
        self._ent_w = tf.get_variable(
            "entity_softmax_w",
            [self._ent_num, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer(uniform=False)
        )
        self._ent_b = tf.get_variable(
            "entity_softmax_b",
            [self._ent_num],
            initializer=tf.constant_initializer(0)
        )

        self.entity_w, self._entity_b = self._ent_w, self._ent_b

        self._lr = tf.Variable(options.learning_rate, trainable=False)

        self._optimizer = tf.train.AdamOptimizer(options.learning_rate)  # , beta2=0.98, epsilon=1e-9

    def bn(self, inputs, is_train=True, reuse=True):
        return tf.contrib.layers.batch_norm(inputs,
                                            center=True,
                                            scale=True,
                                            is_training=is_train,
                                            reuse=reuse,
                                            scope='bn',
                                            )

    def lstm_cell(self, drop=True, keep_prob=0.5, num_layers=2, hidden_size=None):
        if not hidden_size:
            hidden_size = self._options.hidden_size

        def basic_lstm_cell():
            return tf.contrib.rnn.LSTMCell(
                num_units=hidden_size,
                initializer=tf.orthogonal_initializer,
                forget_bias=1,
                reuse=tf.get_variable_scope().reuse,
                activation=tf.identity
            )

        def drop_cell():
            return tf.contrib.rnn.DropoutWrapper(
                basic_lstm_cell(),
                output_keep_prob=keep_prob
            )

        if drop:
            gen_cell = drop_cell
        else:
            gen_cell = basic_lstm_cell

        if num_layers == 0:
            return gen_cell()

        cell = tf.contrib.rnn.MultiRNNCell(
            [gen_cell() for _ in range(num_layers)],
            state_is_tuple=True,
        )
        return cell

    def sampled_loss(self, inputs, labels, w, b, weight=1, is_entity=False):
        num_sampled = min(self._options.num_samples, w.shape[0] // 3)

        labels = tf.reshape(labels, [-1, 1])

        losses = tf.nn.nce_loss(
            weights=w,
            biases=b,
            labels=labels,
            inputs=tf.reshape(inputs, [-1, w.get_shape().as_list()[1]]),
            num_sampled=num_sampled,
            num_classes=w.shape[0],
            partition_strategy='div',
        )
        return losses * weight

    def logits(self, inputs, w, b):
        return tf.nn.bias_add(tf.matmul(inputs, tf.transpose(w)), b)

    # shuffle data
    def sample(self, data):
        choices = np.random.choice(len(data), size=len(data), replace=False)
        return data.iloc[choices]

    # build an RSN of length l
    def build_sub_graph(self, length=15, reuse=False):
        options = self._options
        hidden_size = options.hidden_size
        batch_size = options.batch_size

        seq = tf.placeholder(
            tf.int32, [batch_size, length], name='seq' + str(length))

        e_em, r_em = self._entity_embedding, self._relation_embedding

        # seperately read, and then recover the order
        ent = seq[:, :-1:2]
        rel = seq[:, 1::2]

        ent_em = tf.nn.embedding_lookup(e_em, ent)
        rel_em = tf.nn.embedding_lookup(r_em, rel)

        em_seq = []
        for i in range(length - 1):
            if i % 2 == 0:
                em_seq.append(ent_em[:, i // 2])
            else:
                em_seq.append(rel_em[:, i // 2])

        # seperately bn
        with tf.variable_scope('input_bn'):
            if not reuse:
                bn_em_seq = [tf.reshape(self.bn(em_seq[i], reuse=(
                        i is not 0)), [-1, 1, hidden_size]) for i in range(length - 1)]
            else:
                bn_em_seq = [tf.reshape(
                    self.bn(em_seq[i], reuse=True), [-1, 1, hidden_size]) for i in range(length - 1)]

        bn_em_seq = tf.concat(bn_em_seq, axis=1)

        ent_bn_em = bn_em_seq[:, ::2]

        with tf.variable_scope('rnn', reuse=reuse):

            cell = self.lstm_cell(True, options.keep_prob, options.num_layers)

            outputs, state = tf.nn.dynamic_rnn(cell, bn_em_seq, dtype=tf.float32)

        # with tf.variable_scope('transformer', reuse=reuse):
        #     outputs = transformer_model(input_tensor=bn_em_seq,
        #                                 hidden_size=hidden_size,
        #                                 intermediate_size=hidden_size*4,
        #                                 num_attention_heads=8)

        rel_outputs = outputs[:, 1::2, :]
        outputs = [outputs[:, i, :] for i in range(length - 1)]

        ent_outputs = outputs[::2]

        # RSN
        res_rel_outputs = tf.contrib.layers.fully_connected(rel_outputs, hidden_size, biases_initializer=None,
                                                            activation_fn=None) + \
                          tf.contrib.layers.fully_connected(
                              ent_bn_em, hidden_size, biases_initializer=None, activation_fn=None)

        # recover the order
        res_rel_outputs = [res_rel_outputs[:, i, :] for i in range((length - 1) // 2)]
        outputs = []
        for i in range(length - 1):
            if i % 2 == 0:
                outputs.append(ent_outputs[i // 2])
            else:
                outputs.append(res_rel_outputs[i // 2])

        # output bn
        with tf.variable_scope('output_bn'):
            if reuse:
                bn_outputs = [tf.reshape(
                    self.bn(outputs[i], reuse=True), [-1, 1, hidden_size]) for i in range(length - 1)]
            else:
                bn_outputs = [tf.reshape(self.bn(outputs[i], reuse=(
                        i is not 0)), [-1, 1, hidden_size]) for i in range(length - 1)]

        def cal_loss(bn_outputs, seq):
            losses = []

            masks = np.random.choice([0., 1.0], size=batch_size, p=[0.5, 0.5])
            weight = tf.random_shuffle(tf.cast(masks, tf.float32))
            for i, output in enumerate(bn_outputs):
                if i % 2 == 0:
                    losses.append(self.sampled_loss(
                        output, seq[:, i + 1], self._rel_w, self._rel_b, weight=weight, is_entity=i))
                else:
                    losses.append(self.sampled_loss(
                        output, seq[:, i + 1], self._ent_w, self._ent_b, weight=weight, is_entity=i))
            losses = tf.stack(losses, axis=1)
            return losses

        seq_loss = cal_loss(bn_outputs, seq)

        losses = tf.reduce_sum(seq_loss) / batch_size

        return losses, seq

    # build the main graph
    def _define_embed_graph(self):
        options = self._options

        loss, seq = self.build_sub_graph(length=options.max_length, reuse=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 2.0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = self._optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.train.get_or_create_global_step()
            )

        self._seq, self._loss, self._train_op = seq, loss, train_op

    # training procedure
    def seq_train(self, data, choices=None, epoch=None):
        opts = self._options

        choices = np.random.choice(len(data), size=len(data), replace=True)
        batch_size = opts.batch_size

        num_batch = len(data) // batch_size

        fetches = {
            'loss': self._loss,
            'train_op': self._train_op
        }

        losses = 0
        for i in range(num_batch):
            one_batch_choices = choices[i * batch_size: (i + 1) * batch_size]
            one_batch_data = data.iloc[one_batch_choices]

            feed_dict = {}
            seq = one_batch_data.values[:, :opts.max_length]
            feed_dict[self._seq] = seq

            vals = self.session.run(fetches, feed_dict)

            del one_batch_data

            loss = vals['loss']
            losses += loss
        self._last_mean_loss = losses / num_batch

        return self._last_mean_loss

    def run(self):
        t = time.time()
        train_data = self._train_data
        for i in range(1, self.args.max_epoch + 1):
            time_i = time.time()
            last_mean_loss = self.seq_train(train_data)
            print('epoch %i, avg. batch_loss: %f,  cost time: %.4f s' % (i, last_mean_loss, time.time() - time_i))
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i >= self.args.max_epoch:
                    break
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
