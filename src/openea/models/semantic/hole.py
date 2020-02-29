import tensorflow as tf

from openea.models.basic_model import BasicModel
from openea.modules.base.initializers import init_embeddings
from openea.modules.base.optimizers import generate_optimizer
from openea.modules.utils.util import generate_out_folder
from openea.modules.utils.util import load_session


class HolE(BasicModel):
    """
    A small margin, such as 0.2 or 0.3, works!
    """

    def set_kgs(self, kgs):
        self.kgs = kgs

    def set_args(self, args):
        self.args = args
        self.out_folder = generate_out_folder(self.args.output, self.args.training_data, self.args.dataset_division, self.__class__.__name__)

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
        assert self.args.loss_norm == 'L2'
        assert self.args.ent_l2_norm is True
        assert self.args.rel_l2_norm is True
        assert self.args.margin > 0.0

    def __init__(self):
        super().__init__()

    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'ent_embeds',
                                              self.args.init, self.args.ent_l2_norm)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds',
                                              self.args.init, self.args.rel_l2_norm)

    def _cconv(self, a, b):
        return tf.ifft(tf.fft(a) * tf.fft(b)).real

    def _ccorr(self, a, b):
        a = tf.cast(a, tf.complex64)
        b = tf.cast(b, tf.complex64)
        return tf.real(tf.ifft(tf.conj(tf.fft(a)) * tf.fft(b)))

    def _calc(self, head, tail, rel):
        relation_mention = tf.nn.l2_normalize(rel, 1)
        entity_mention = self._ccorr(head, tail)
        # entity_mention = tf.nn.l2_normalize(entity_mention, 1)
        return -tf.sigmoid(tf.reduce_sum(relation_mention * entity_mention, 1, keep_dims=True))

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
        with tf.name_scope('triple_loss'):
            pos_score = self._calc(phs, pts, prs)
            neg_score = self._calc(nhs, nts, nrs)
            if self.args.neg_triple_num > 1:
                neg_score = tf.reshape(neg_score, [-1, self.args.neg_triple_num])
                neg_score = tf.reduce_mean(neg_score, 1, keep_dims=True)
            self.triple_loss = tf.reduce_sum(tf.nn.relu(tf.constant(self.args.margin) + pos_score - neg_score),
                                             name='margin_loss')
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)
