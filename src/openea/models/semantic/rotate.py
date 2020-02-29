import tensorflow as tf
from sklearn import preprocessing

import openea.modules.load.read as rd
from openea.models.basic_model import BasicModel
from openea.modules.base.initializers import init_embeddings
from openea.modules.base.optimizers import get_optimizer, generate_optimizer
from openea.modules.utils.util import generate_out_folder
from openea.modules.utils.util import load_session

'''
This implementation of RotatE is prone to running into a NaN loss under some gamma values.
Although we have tried many solutions, it is still likely to happen.
So, when running into NaN loss, we suggest to run it one more time or try other gamma values.

In the early debug phase, we also found that a small learning rate is more likely to lead to NaN loss.
'''


class RotatE(BasicModel):

    def __init__(self):
        super().__init__()
        self.pi = 3.14159265358979323846
        self.epsilon = 2.0
        self.embedding_range = None

    def set_kgs(self, kgs):
        self.kgs = kgs

    def set_args(self, args):
        self.args = args
        self.out_folder = generate_out_folder(self.args.output, self.args.training_data, self.args.dataset_division,
                                              self.__class__.__name__)

    def init(self):
        self.embedding_range = (self.args.gamma + self.epsilon) / self.args.dim
        self._define_variables()
        self._define_embed_graph()
        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)

        assert self.args.init == 'uniform'
        assert self.args.alignment_module == 'sharing'
        assert self.args.neg_sampling == 'uniform'
        assert self.args.optimizer == 'Adam'
        assert self.args.eval_metric == 'inner'
        # assert self.args.ent_l2_norm is True
        # assert self.args.rel_l2_norm is True
        assert self.args.gamma > 0.0

    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            self.re_ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 're_ent_embeds',
                                                 self.args.init, self.args.ent_l2_norm, tf.float64)
            self.im_ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'im_ent_embeds',
                                                 self.args.init, self.args.ent_l2_norm, tf.float64)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds',
                                              self.args.init, self.args.rel_l2_norm, tf.float64)

    def _generate_scores(self, rh, rr, rt, ih, ir, it, pos=True):
        re_score = rh * rr - ih * ir - rt
        im_score = rh * ir + ih * rr - it
        # print("im_score", im_score.shape)
        scores = tf.stack([re_score, im_score], axis=0)
        # print("scores 1", scores.shape)
        scores = tf.norm(scores, axis=0)
        # print("scores 2", scores.shape)
        scores = tf.reduce_sum(scores, axis=-1)
        # print("scores 3", scores.shape)
        scores = self.args.gamma - scores
        return scores if pos else -scores

    def _generate_loss(self, pos_scores, neg_scores):
        pos_scores = tf.sigmoid(pos_scores)
        neg_scores = tf.sigmoid(neg_scores)
        pos_scores = tf.log(pos_scores)
        neg_scores = tf.log(neg_scores)
        pos_loss = tf.reduce_sum(pos_scores)
        neg_loss = tf.reduce_sum(neg_scores)
        loss = - pos_loss - neg_loss / self.args.neg_triple_num
        return loss

    def lookup_all(self, h, r, t):
        re_head = tf.nn.embedding_lookup(self.re_ent_embeds, h)
        re_tail = tf.nn.embedding_lookup(self.re_ent_embeds, t)
        im_head = tf.nn.embedding_lookup(self.im_ent_embeds, h)
        im_tail = tf.nn.embedding_lookup(self.im_ent_embeds, t)
        relation = tf.nn.embedding_lookup(self.rel_embeds, r)
        phase_relation = relation / (self.embedding_range / self.pi)
        re_relation = tf.cos(phase_relation)
        im_relation = tf.sin(phase_relation)
        return re_head, re_relation, re_tail, im_head, im_relation, im_tail

    def _define_embed_graph(self):
        with tf.name_scope('triple_placeholder'):
            self.pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.pos_ts = tf.placeholder(tf.int32, shape=[None])
            self.neg_hs = tf.placeholder(tf.int32, shape=[None])
            self.neg_rs = tf.placeholder(tf.int32, shape=[None])
            self.neg_ts = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('triple_lookup'):
            prh, prr, prt, pih, pir, pit = self.lookup_all(self.pos_hs, self.pos_rs, self.pos_ts)
            nrh, nrr, nrt, nih, nir, nit = self.lookup_all(self.neg_hs, self.neg_rs, self.neg_ts)

        with tf.name_scope('triple_loss'):
            pos_scores = self._generate_scores(prh, prr, prt, pih, pir, pit, pos=True)
            neg_scores = self._generate_scores(nrh, nrr, nrt, nih, nir, nit, pos=False)
            self.triple_loss = self._generate_loss(pos_scores, neg_scores)
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)
            # optimizer = get_optimizer(self.args.optimizer, self.args.learning_rate)
            # train_vars = tf.trainable_variables()
            # grads, _ = tf.clip_by_global_norm(tf.gradients(self.triple_loss, train_vars), 2.0)  # To avoid Nan loss!!
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            #     self.triple_optimizer = optimizer.apply_gradients(zip(grads, train_vars),
            #                                                       global_step=tf.train.get_or_create_global_step())

    def _eval_valid_embeddings(self):
        embeds1 = tf.nn.embedding_lookup(self.re_ent_embeds, self.kgs.valid_entities1).eval(session=self.session) + \
                  tf.nn.embedding_lookup(self.im_ent_embeds, self.kgs.valid_entities1).eval(session=self.session)
        embeds2 = tf.nn.embedding_lookup(self.re_ent_embeds, self.kgs.valid_entities2 +
                                         self.kgs.test_entities2).eval(session=self.session) + \
                  tf.nn.embedding_lookup(self.im_ent_embeds, self.kgs.valid_entities2 +
                                         self.kgs.test_entities2).eval(session=self.session)
        mapping = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        return embeds1, embeds2, mapping

    def _eval_test_embeddings(self):
        embeds1 = tf.nn.embedding_lookup(self.re_ent_embeds, self.kgs.test_entities1).eval(session=self.session) + \
                  tf.nn.embedding_lookup(self.im_ent_embeds, self.kgs.test_entities1).eval(session=self.session)
        embeds2 = tf.nn.embedding_lookup(self.re_ent_embeds, self.kgs.test_entities2).eval(session=self.session) + \
                  tf.nn.embedding_lookup(self.im_ent_embeds, self.kgs.test_entities2).eval(session=self.session)
        mapping = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        return embeds1, embeds2, mapping

    def save(self):
        ent_embeds = self.re_ent_embeds.eval(session=self.session) + self.im_ent_embeds.eval(session=self.session)
        ent_embeds = preprocessing.normalize(ent_embeds)
        rel_embeds = self.rel_embeds.eval(session=self.session)
        mapping_mat = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        rd.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, mapping_mat=mapping_mat)
