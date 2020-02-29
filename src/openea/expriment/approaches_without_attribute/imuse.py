import tensorflow as tf
import Levenshtein
import time
import math
import multiprocessing as mp
import multiprocessing

from openea.modules.base.optimizers import generate_optimizer
from openea.modules.utils.util import load_session
from openea.modules.base.initializers import init_embeddings
from openea.modules.base.losses import get_loss_func
from openea.models.basic_model import BasicModel
from openea.modules.utils.util import task_divide
from openea.modules.finding.evaluation import early_stop


def interactive_model(kgs, args):
    start = time.time()
    aligned_ent_pair_set_all = set()
    i = 0
    aligned_attr_pair_set_all = get_aligned_attr_pair_by_name_similarity(kgs, 0.6)
    print('aligned_attr_pair_set:', len(aligned_attr_pair_set_all))
    while True:
        i += 1
        aligned_ent_pair_set_iter = align_entity_by_attributes(kgs, aligned_attr_pair_set_all, args.sim_thresholds_ent)
        aligned_ent_pair_set_all |= aligned_ent_pair_set_iter
        print(i, 'len(aligned_ent_pair_set_all):', len(aligned_ent_pair_set_all), 'len(aligned_ent_pair_set_iter):',
              len(aligned_ent_pair_set_iter))
        if i >= args.interactive_model_iter_num:
            break
        aligned_attr_pair_set_iter = align_attribute_by_entities(kgs, aligned_ent_pair_set_all,
                                                                 args.sim_thresholds_attr)
        if len(aligned_attr_pair_set_all | aligned_attr_pair_set_iter) == len(aligned_attr_pair_set_all):
            break
        aligned_attr_pair_set_all |= aligned_attr_pair_set_iter
        print(i, 'len(aligned_attr_pair_set_all):', len(aligned_attr_pair_set_all), 'len(aligned_attr_pair_set_iter):',
              len(aligned_attr_pair_set_iter))
    print(time.time() - start)
    return aligned_ent_pair_set_all


def run_one_ea(ent_attrs_dict_1, ent_attrs_dict_2, ent_attr_value_dict_1, ent_attr_value_dict_2, sim_thresholds_ent,
               aligned_attr_pair_set):
    aligned_ent_pair_set_i = set()
    cnt = 0
    target_ent_set = set()
    for e1, attrs1 in ent_attrs_dict_1.items():
        cnt += 1
        target_ent = None
        sim_max = sim_thresholds_ent
        for e2, attrs2 in ent_attrs_dict_2.items():
            sim, sim_cnt = 0, 0
            for (a1, a2) in aligned_attr_pair_set:
                if a1 in attrs1 and a2 in attrs2:
                    sim += compute_two_values_similarity(ent_attr_value_dict_1[(e1, a1)],
                                                         ent_attr_value_dict_2[(e2, a2)])
                    sim_cnt += 1
            if sim_cnt > 0:
                sim /= sim_cnt
            if sim > sim_max:
                target_ent = e2
                sim_max = sim
            if target_ent is not None and target_ent not in target_ent_set:
                aligned_ent_pair_set_i.add((e1, target_ent))
                target_ent_set.add(target_ent)
    return aligned_ent_pair_set_i


def align_entity_by_attributes(kgs, aligned_attr_pair_set, sim_thresholds_ent):
    print('align_entity_by_attributes...')
    aligned_ent_pair_set = set()
    if len(aligned_attr_pair_set) == 0:
        return aligned_ent_pair_set
    ent_attrs_dict_1, ent_attr_value_dict_1 = filter_by_aligned_attributes(kgs.kg1.attribute_triples_set,
                                                                           set([a for (a, _) in aligned_attr_pair_set]))
    ent_attrs_dict_2, ent_attr_value_dict_2 = filter_by_aligned_attributes(kgs.kg2.attribute_triples_set,
                                                                           set([a for (_, a) in aligned_attr_pair_set]))
    ent_set_1 = list(ent_attrs_dict_1.keys())
    size = len(ent_set_1) // 8
    pool = multiprocessing.Pool(processes=8)
    res = list()
    for i in range(8):
        if i == 7:
            ent_set_i = ent_set_1[size * i:]
        else:
            ent_set_i = ent_set_1[size * i:size * (i + 1)]
        ent_attrs_dict_1_i = dict([(k, ent_attrs_dict_1[k]) for k in ent_set_i])
        res.append(pool.apply_async(run_one_ea, (ent_attrs_dict_1_i, ent_attrs_dict_2, ent_attr_value_dict_1,
                                                 ent_attr_value_dict_2, sim_thresholds_ent, aligned_attr_pair_set)))
    pool.close()
    pool.join()

    for _res in res:
        aligned_ent_pair_set |= _res.get()
    temp_dict = dict([(x, y) for (x, y) in aligned_ent_pair_set])
    aligned_ent_pair_set = set([(x, y) for x, y in temp_dict.items()])
    return aligned_ent_pair_set


def run_one_ae(attr_ents_dict_1, attr_ents_dict_2, attr_ent_value_dict_1, attr_ent_value_dict_2, sim_thresholds_attr,
               aligned_ent_pair_set):
    aligned_attr_pair_set = set()
    target_attr_set = set()
    for a1, ents1 in attr_ents_dict_1.items():
        target_attr = None
        sim_max = sim_thresholds_attr
        for a2, ents2 in attr_ents_dict_2.items():
            sim, sim_cnt = 0, 0
            for (e1, e2) in aligned_ent_pair_set:
                if e1 in ents1 and e2 in ents2:
                    sim += compute_two_values_similarity(attr_ent_value_dict_1[(a1, e1)],
                                                         attr_ent_value_dict_2[(a2, e2)])
                    sim_cnt += 1
            if sim_cnt > 0:
                sim /= sim_cnt
            if sim > sim_max:
                target_attr = a2
                sim_max = sim
            if target_attr is not None and target_attr not in target_attr_set:
                aligned_attr_pair_set.add((a1, target_attr))
                target_attr_set.add(target_attr)
    return aligned_attr_pair_set


def align_attribute_by_entities(kgs, aligned_ent_pair_set, sim_thresholds_attr):
    print('align_attribute_by_entities...')
    aligned_attr_pair_set = set()
    if aligned_ent_pair_set is None or len(aligned_ent_pair_set) == 0:
        return aligned_attr_pair_set
    attr_ents_dict_1, attr_ent_value_dict_1 = filter_by_aligned_attributes(kgs.kg1.attribute_triples_set,
                                                                           set([e for (e, _) in aligned_ent_pair_set]))
    attr_ents_dict_2, attr_ent_value_dict_2 = filter_by_aligned_attributes(kgs.kg2.attribute_triples_set,
                                                                           set([e for (_, e) in aligned_ent_pair_set]))
    attr_set_1 = list(attr_ents_dict_1.keys())
    size = len(attr_set_1) // 8
    pool = multiprocessing.Pool(processes=8)
    res = list()
    for i in range(8):
        if i == 7:
            attr_set_i = attr_set_1[size * i:]
        else:
            attr_set_i = attr_set_1[size * i:size * (i + 1)]
        attr_ents_dict_1_i = dict([(k, attr_ents_dict_1[k]) for k in attr_set_i])
        res.append(pool.apply_async(run_one_ae, (attr_ents_dict_1_i, attr_ents_dict_2, attr_ent_value_dict_1,
                                                 attr_ent_value_dict_2, sim_thresholds_attr, aligned_ent_pair_set)))
    pool.close()
    pool.join()

    for _res in res:
        aligned_attr_pair_set |= _res.get()
    temp_dict = dict([(x, y) for (x, y) in aligned_attr_pair_set])
    aligned_attr_pair_set = set([(x, y) for x, y in temp_dict.items()])
    return aligned_attr_pair_set


def filter_by_aligned_attributes(attr_triples, attr_set):
    ent_attrs_dict, ent_attr_value_dict = {}, {}
    for (e, a, v) in attr_triples:
        if a in attr_set and (e, a) not in ent_attr_value_dict:
            ent_attr_value_dict[(e, a)] = v
            attrs = set()
            if e in ent_attrs_dict:
                attrs = ent_attrs_dict[e]
            attrs.add(a)
            ent_attrs_dict[e] = attrs
    return ent_attrs_dict, ent_attr_value_dict


def filter_by_aligned_entities(attr_triples, ent_set):
    attr_ents_dict, attr_ent_value_dict = {}, {}
    for (e, a, v) in attr_triples:
        if e in ent_set and (a, e) not in attr_ent_value_dict:
            attr_ent_value_dict[(a, e)] = v
            ents = set()
            if a in attr_ents_dict:
                ents = attr_ents_dict[a]
            attr_ents_dict[e] = ents
    return attr_ents_dict, attr_ent_value_dict


def cal_lcs_sim(first_str, second_str):
    len_1 = len(first_str.strip())
    len_2 = len(second_str.strip())
    len_vv = [[0] * (len_2 + 2)] * (len_1 + 2)
    for i in range(1, len_1 + 1):
        for j in range(1, len_2 + 1):
            if first_str[i - 1] == second_str[j - 1]:
                len_vv[i][j] = 1 + len_vv[i - 1][j - 1]
            else:
                len_vv[i][j] = max(len_vv[i - 1][j], len_vv[i][j - 1])

    return float(float(len_vv[len_1][len_2] * 2) / float(len_1 + len_2))


def compute_two_values_similarity(v1, v2):
    # lcs_sim = cal_lcs_sim(v1, v2)
    # return lcs_sim/(Levenshtein.ratio(v1, v2)+lcs_sim)*2
    return Levenshtein.ratio(v1, v2)


def get_aligned_attr_pair_by_name_similarity(kgs, sim_thresholds_attr, top_k=10):
    def turn_id_attr_dict(attr_id_dict):
        id_attr_dict = {}
        for a, i in attr_id_dict.items():
            id_attr_dict[i] = a
        return id_attr_dict

    id_attr_dict_1 = turn_id_attr_dict(kgs.kg1.attributes_id_dict)
    id_attr_dict_2 = turn_id_attr_dict(kgs.kg2.attributes_id_dict)
    aligned_attr_pair_set = set()
    attr2_set = set()
    for attr1 in kgs.kg1.attributes_set:
        target_attr = None
        sim_max = sim_thresholds_attr
        attr_str_1 = id_attr_dict_1[attr1].split('/')[-1]
        for attr2 in kgs.kg2.attributes_set:
            attr_str_2 = id_attr_dict_2[attr2].split('/')[-1]
            sim = Levenshtein.ratio(attr_str_1, attr_str_2)
            if sim > sim_max:
                target_attr = attr2
                sim_max = sim
        if target_attr is not None and target_attr not in attr2_set:
            aligned_attr_pair_set.add((attr1, target_attr))
            attr2_set.add(target_attr)

    attr_num_dict_1, attr_num_dict_2 = {}, {}
    for (_, a, _) in kgs.kg1.attribute_triples_set:
        num = 1
        if a in attr_num_dict_1:
            num += attr_num_dict_1[a]
        attr_num_dict_1[a] = num
    for (_, a, _) in kgs.kg2.attribute_triples_set:
        num = 1
        if a in attr_num_dict_2:
            num += attr_num_dict_2[a]
        attr_num_dict_2[a] = num
    attr_pair_num_dict = {}
    for (a1, a2) in aligned_attr_pair_set:
        num = 0
        if a1 in attr_num_dict_1:
            num += attr_num_dict_1[a1]
        if a2 in attr_num_dict_2:
            num += attr_num_dict_2[a2]
        attr_pair_num_dict[(a1, a2)] = num
    attr_pair_list = sorted(attr_pair_num_dict.items(), key=lambda d: d[1], reverse=True)
    if top_k > len(attr_pair_list):
        top_k = len(attr_pair_list)
    aligned_attr_pair_set_top = set([a_pair for (a_pair, _) in attr_pair_list[: top_k]])
    return aligned_attr_pair_set_top


class IMUSE(BasicModel):

    def __init__(self):
        super().__init__()

    def init(self):
        # self.aligned_ent_pair_set = interactive_model(self.kgs, self.args)
        self._define_variables()
        self._define_embed_graph()
        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)

        # customize parameters
        assert self.args.init == 'normal'
        assert self.args.loss == 'margin-based'
        assert self.args.neg_sampling == 'uniform'
        assert self.args.optimizer == 'SGD'
        assert self.args.eval_metric == 'inner'
        assert self.args.loss_norm == 'L2'

        assert self.args.ent_l2_norm is True
        assert self.args.rel_l2_norm is True

        assert self.args.neg_triple_num == 1
        assert self.args.learning_rate >= 0.01

    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'ent_embeds',
                                              self.args.init, self.args.ent_l2_norm)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds',
                                              self.args.init, self.args.rel_l2_norm)

    def _define_embed_graph(self):
        with tf.name_scope('triple_placeholder'):
            self.pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.pos_ts = tf.placeholder(tf.int32, shape=[None])
            self.neg_hs = tf.placeholder(tf.int32, shape=[None])
            self.neg_rs = tf.placeholder(tf.int32, shape=[None])
            self.neg_ts = tf.placeholder(tf.int32, shape=[None])

            self.aligned_ents1 = tf.placeholder(tf.int32, shape=[None])
            self.aligned_ents2 = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('triple_lookup'):
            phs = tf.nn.embedding_lookup(self.ent_embeds, self.pos_hs)
            prs = tf.nn.embedding_lookup(self.rel_embeds, self.pos_rs)
            pts = tf.nn.embedding_lookup(self.ent_embeds, self.pos_ts)
            nhs = tf.nn.embedding_lookup(self.ent_embeds, self.neg_hs)
            nrs = tf.nn.embedding_lookup(self.rel_embeds, self.neg_rs)
            nts = tf.nn.embedding_lookup(self.ent_embeds, self.neg_ts)

            ents1 = tf.nn.embedding_lookup(self.ent_embeds, self.aligned_ents1)
            ents2 = tf.nn.embedding_lookup(self.ent_embeds, self.aligned_ents2)
        with tf.name_scope('triple_loss'):
            self.triple_loss = get_loss_func(phs, prs, pts, nhs, nrs, nts, self.args)
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)
        with tf.name_scope('align_loss'):
            self.align_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(ents1 - ents2, 2), 1))
            self.align_optimizer = generate_optimizer(self.align_loss, self.args.learning_rate,
                                                      opt=self.args.optimizer)

    # def launch_align_training_1epo(self, epoch):
    #     start = time.time()
    #     epoch_loss = 0
    #     trained_samples_num = 0
    #     steps = int(math.ceil(len(self.aligned_ent_pair_set) / self.args.batch_size))
    #     for i in range(steps):
    #         batch_ent_pairs = list(self.aligned_ent_pair_set)
    #         batch_loss, _ = self.session.run(fetches=[self.align_loss, self.align_optimizer],
    #                                          feed_dict={self.aligned_ents1: [x[0] for x in batch_ent_pairs],
    #                                                     self.aligned_ents2: [x[1] for x in batch_ent_pairs]})
    #         trained_samples_num += len(batch_ent_pairs)
    #         epoch_loss += batch_loss
    #     epoch_loss /= trained_samples_num
    #     print('epoch {}, align learning loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def run(self):
        t = time.time()
        relation_triples_num = len(self.kgs.kg1.relation_triples_list) + len(self.kgs.kg2.relation_triples_list)
        relation_triple_steps = int(math.ceil(relation_triples_num / self.args.batch_size))
        relation_step_tasks = task_divide(list(range(relation_triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        relation_batch_queue = manager.Queue()
        for i in range(1, self.args.max_epoch + 1):
            self.launch_triple_training_1epo(i, relation_triple_steps, relation_step_tasks, relation_batch_queue, None, None)
            # self.launch_align_training_1epo(i)
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))
