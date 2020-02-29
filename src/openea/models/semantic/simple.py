import tensorflow as tf
from sklearn import preprocessing

import openea.modules.load.read as rd

from openea.models.basic_model import BasicModel
from openea.modules.base.initializers import init_embeddings
from openea.modules.base.optimizers import generate_optimizer
from openea.modules.utils.util import generate_out_folder
from openea.modules.utils.util import load_session


class SimplE(BasicModel):
    def set_kgs(self, kgs):
        self.kgs = kgs

    def set_args(self, args):
        self.args = args
        self.out_folder = generate_out_folder(self.args.output, self.args.training_data, self.args.dataset_division,
                                              self.__class__.__name__)

    def init(self):
        self._define_variables()
        self._define_embed_graph()
        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)

        assert self.args.init == 'xavier'
        assert self.args.alignment_module == 'sharing'
        assert self.args.neg_sampling == 'uniform'
        assert self.args.optimizer == 'Adagrad'
        assert self.args.eval_metric == 'inner'
        assert self.args.ent_l2_norm is True
        assert self.args.rel_l2_norm is True

    def __init__(self):
        super().__init__()

    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            self.head_ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'head_ent_embeds',
                                                   self.args.init, self.args.ent_l2_norm)
            self.tail_ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'tail_ent_embeds',
                                                   self.args.init, self.args.ent_l2_norm)
            self.rel_embeds1 = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds1',
                                               self.args.init, self.args.rel_l2_norm)
            self.rel_embeds2 = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds2',
                                               self.args.init, self.args.rel_l2_norm)

    def _calc(self, hs, rs, ts):
        hrs = tf.multiply(hs, rs)
        hrs = tf.nn.l2_normalize(hrs, 1)
        scores = tf.reduce_sum(tf.multiply(hrs, ts), 1)
        return scores

    def _generate_loss(self, hs1, rs1, ts1, hs2, rs2, ts2, pos=True):
        scores = (self._calc(hs1, rs1, ts1) + self._calc(hs2, rs2, ts2)) / 2
        return tf.reduce_sum(tf.nn.softplus(-scores)) if pos else tf.reduce_sum(tf.nn.softplus(scores))

    def _define_embed_graph(self):
        with tf.name_scope('triple_placeholder'):
            self.pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.pos_ts = tf.placeholder(tf.int32, shape=[None])
            self.neg_hs = tf.placeholder(tf.int32, shape=[None])
            self.neg_rs = tf.placeholder(tf.int32, shape=[None])
            self.neg_ts = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('triple_lookup'):
            phs1 = tf.nn.embedding_lookup(self.head_ent_embeds, self.pos_hs)
            prs1 = tf.nn.embedding_lookup(self.rel_embeds1, self.pos_rs)
            pts1 = tf.nn.embedding_lookup(self.tail_ent_embeds, self.pos_ts)

            phs2 = tf.nn.embedding_lookup(self.head_ent_embeds, self.pos_ts)
            prs2 = tf.nn.embedding_lookup(self.rel_embeds2, self.pos_rs)
            pts2 = tf.nn.embedding_lookup(self.tail_ent_embeds, self.pos_hs)

            nhs1 = tf.nn.embedding_lookup(self.head_ent_embeds, self.neg_hs)
            nrs1 = tf.nn.embedding_lookup(self.rel_embeds1, self.neg_rs)
            nts1 = tf.nn.embedding_lookup(self.tail_ent_embeds, self.neg_ts)

            nhs2 = tf.nn.embedding_lookup(self.head_ent_embeds, self.neg_ts)
            nrs2 = tf.nn.embedding_lookup(self.rel_embeds2, self.neg_rs)
            nts2 = tf.nn.embedding_lookup(self.tail_ent_embeds, self.neg_hs)
        with tf.name_scope('triple_loss'):
            self.triple_loss = self._generate_loss(phs1, prs1, pts1, phs2, prs2, pts2, pos=True) + \
                               self._generate_loss(nhs1, nrs1, nts1, nhs2, nrs2, nts2, pos=False)
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)

    def _eval_valid_embeddings(self):
        print("valid")
        embeds1 = tf.nn.embedding_lookup(self.head_ent_embeds, self.kgs.valid_entities1).eval(session=self.session) + \
                  tf.nn.embedding_lookup(self.tail_ent_embeds, self.kgs.valid_entities1).eval(session=self.session)
        embeds2 = tf.nn.embedding_lookup(self.head_ent_embeds, self.kgs.valid_entities2 +
                                         self.kgs.test_entities2).eval(session=self.session) + \
                  tf.nn.embedding_lookup(self.tail_ent_embeds, self.kgs.valid_entities2 +
                                         self.kgs.test_entities2).eval(session=self.session)
        mapping = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        return embeds1, embeds2, mapping

    def _eval_test_embeddings(self):
        print("test")
        embeds1 = tf.nn.embedding_lookup(self.head_ent_embeds, self.kgs.test_entities1).eval(session=self.session) + \
                  tf.nn.embedding_lookup(self.tail_ent_embeds, self.kgs.test_entities1).eval(session=self.session)
        embeds2 = tf.nn.embedding_lookup(self.head_ent_embeds, self.kgs.test_entities2).eval(session=self.session) + \
                  tf.nn.embedding_lookup(self.tail_ent_embeds, self.kgs.test_entities2).eval(session=self.session)
        mapping = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        return embeds1, embeds2, mapping

    def save(self):
        ent_embeds = self.head_ent_embeds.eval(session=self.session) + self.tail_ent_embeds.eval(session=self.session)
        ent_embeds = preprocessing.normalize(ent_embeds)
        rel_embeds = self.rel_embeds1.eval(session=self.session) + self.rel_embeds2.eval(session=self.session)
        mapping_mat = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        rd.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, mapping_mat=mapping_mat)
