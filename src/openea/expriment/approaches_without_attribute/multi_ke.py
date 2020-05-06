import math
import multiprocessing as mp
import numpy as np
import random
import gc
from sklearn import preprocessing

import openea.modules.train.batch as bat
from openea.modules.base.initializers import xavier_init

from openea.modules.base.optimizers import generate_optimizer
from openea.modules.utils.util import *
from openea.modules.load.read import generate_sup_attribute_triples
from openea.modules.base.losses import logistic_loss, positive_loss
from openea.modules.finding.evaluation import early_stop

from openea.models.basic_model import BasicModel
from openea.approaches.literal_encoder import LiteralEncoder
from openea.approaches.predicate_alignmnet import PredicateAlignModel
import openea.modules.finding.evaluation as eva


def test_WVA(model):
    nv_ent_embeds1 = tf.nn.embedding_lookup(model.name_embeds, model.kgs.test_entities1).eval(session=model.session)
    rv_ent_embeds1 = tf.nn.embedding_lookup(model.rv_ent_embeds, model.kgs.test_entities1).eval(session=model.session)
    av_ent_embeds1 = tf.nn.embedding_lookup(model.av_ent_embeds, model.kgs.test_entities1).eval(session=model.session)
    weight11, weight21, weight31 = wva(nv_ent_embeds1, rv_ent_embeds1, av_ent_embeds1)

    test_list = model.kgs.test_entities2
    nv_ent_embeds2 = tf.nn.embedding_lookup(model.name_embeds, test_list).eval(session=model.session)
    rv_ent_embeds2 = tf.nn.embedding_lookup(model.rv_ent_embeds, test_list).eval(session=model.session)
    av_ent_embeds2 = tf.nn.embedding_lookup(model.av_ent_embeds, test_list).eval(session=model.session)
    weight12, weight22, weight32 = wva(nv_ent_embeds2, rv_ent_embeds2, av_ent_embeds2)

    weight1 = weight11 + weight12
    weight2 = weight21 + weight22
    weight3 = weight31 + weight32
    all_weight = weight1 + weight2 + weight3
    weight1 /= all_weight
    weight2 /= all_weight
    weight3 /= all_weight

    print('weights', weight1, weight2, weight3)

    embeds1 = weight1 * nv_ent_embeds1 + \
              weight2 * rv_ent_embeds1 + \
              weight3 * av_ent_embeds1
    embeds2 = weight1 * nv_ent_embeds2 + \
              weight2 * rv_ent_embeds2 + \
              weight3 * av_ent_embeds2
    print('wvag test results:')
    hits1_12, mrr_12 = eva.valid(embeds1, embeds2, None, model.args.top_k, model.args.test_threads_num,
                                 normalize=True)
    del embeds1, embeds2
    gc.collect()
    return mrr_12


def _compute_weight(embeds1, embeds2, embeds3):
    def min_max_normalization(mat):
        min_ = np.min(mat)
        max_ = np.max(mat)
        return (mat - min_) / (max_ - min_)

    other_embeds = (embeds1 + embeds2 + embeds3) / 3
    other_embeds = preprocessing.normalize(other_embeds)
    embeds1 = preprocessing.normalize(embeds1)
    sim_mat = np.matmul(embeds1, other_embeds.T)
    weights = np.diag(sim_mat)
    return np.mean(weights)


def wva(embeds1, embeds2, embeds3):
    weight1 = _compute_weight(embeds1, embeds2, embeds3)
    weight2 = _compute_weight(embeds2, embeds1, embeds3)
    weight3 = _compute_weight(embeds3, embeds1, embeds2)
    return weight1, weight2, weight3
    all_weight = weight1 + weight2 + weight3
    weight1 /= all_weight
    weight2 /= all_weight
    weight3 /= all_weight
    print('final weights', weight1, weight2, weight3)
    ent_embeds = weight1 * embeds1 + \
                 weight2 * embeds2 + \
                 weight3 * embeds3
    return ent_embeds


def valid_WVA(model):
    nv_ent_embeds1 = tf.nn.embedding_lookup(model.name_embeds, model.kgs.valid_entities1).eval(session=model.session)
    rv_ent_embeds1 = tf.nn.embedding_lookup(model.rv_ent_embeds, model.kgs.valid_entities1).eval(session=model.session)
    av_ent_embeds1 = tf.nn.embedding_lookup(model.av_ent_embeds, model.kgs.valid_entities1).eval(session=model.session)
    weight11, weight21, weight31 = wva(nv_ent_embeds1, rv_ent_embeds1, av_ent_embeds1)

    test_list = model.kgs.valid_entities2 + model.kgs.test_entities2
    nv_ent_embeds2 = tf.nn.embedding_lookup(model.name_embeds, test_list).eval(session=model.session)
    rv_ent_embeds2 = tf.nn.embedding_lookup(model.rv_ent_embeds, test_list).eval(session=model.session)
    av_ent_embeds2 = tf.nn.embedding_lookup(model.av_ent_embeds, test_list).eval(session=model.session)
    weight12, weight22, weight32 = wva(nv_ent_embeds2, rv_ent_embeds2, av_ent_embeds2)

    weight1 = weight11 + weight12
    weight2 = weight21 + weight22
    weight3 = weight31 + weight32
    all_weight = weight1 + weight2 + weight3
    weight1 /= all_weight
    weight2 /= all_weight
    weight3 /= all_weight

    print('weights', weight1, weight2, weight3)

    embeds1 = weight1 * nv_ent_embeds1 + \
              weight2 * rv_ent_embeds1 + \
              weight3 * av_ent_embeds1
    embeds2 = weight1 * nv_ent_embeds2 + \
              weight2 * rv_ent_embeds2 + \
              weight3 * av_ent_embeds2
    print('wvag valid results:')
    hits1_12, mrr_12 = eva.valid(embeds1, embeds2, None, model.args.top_k, model.args.test_threads_num,
                                 normalize=True)

    del nv_ent_embeds1, rv_ent_embeds1, av_ent_embeds1
    del nv_ent_embeds2, rv_ent_embeds2, av_ent_embeds2
    del embeds1, embeds2
    gc.collect()

    return mrr_12


def valid_temp(model, embed_choice='avg', w=(1, 1, 1)):
    if embed_choice == 'nv':
        ent_embeds = model.name_embeds.eval(session=model.session)
    elif embed_choice == 'rv':
        ent_embeds = model.rv_ent_embeds.eval(session=model.session)
    elif embed_choice == 'av':
        ent_embeds = model.av_ent_embeds.eval(session=model.session)
    elif embed_choice == 'final':
        ent_embeds = model.ent_embeds.eval(session=model.session)
    elif embed_choice == 'avg':
        ent_embeds = w[0] * model.name_embeds.eval(session=model.session) + \
                     w[1] * model.rv_ent_embeds.eval(session=model.session) + \
                     w[2] * model.av_ent_embeds.eval(session=model.session)
    else:  # 'final'
        ent_embeds = model.ent_embeds
    print(embed_choice, 'valid results:')
    embeds1 = ent_embeds[model.kgs.valid_entities1,]
    embeds2 = ent_embeds[model.kgs.valid_entities2 + model.kgs.test_entities2,]
    hits1_12, mrr_12 = eva.valid(embeds1, embeds2, None, model.args.top_k, model.args.test_threads_num,
                                 normalize=True)
    del embeds1, embeds2
    gc.collect()
    return mrr_12


def conv(attr_hs, attr_as, attr_vs, dim, feature_map_size=2, kernel_size=[2, 4], activation=tf.nn.tanh, layer_num=2):
    attr_as = tf.reshape(attr_as, [-1, 1, dim])
    attr_vs = tf.reshape(attr_vs, [-1, 1, dim])

    input_avs = tf.concat([attr_as, attr_vs], 1)
    input_shape = input_avs.shape.as_list()
    input_layer = tf.reshape(input_avs, [-1, input_shape[1], input_shape[2], 1])
    _conv = input_layer
    _conv = tf.layers.batch_normalization(_conv, 2)
    for i in range(layer_num):
        _conv = tf.layers.conv2d(inputs=_conv,
                                 filters=feature_map_size,
                                 kernel_size=kernel_size,
                                 strides=[1, 1],
                                 padding="same",
                                 activation=activation)
    _conv = tf.nn.l2_normalize(_conv, 2)
    _shape = _conv.shape.as_list()
    _flat = tf.reshape(_conv, [-1, _shape[1] * _shape[2] * _shape[3]])
    dense = tf.layers.dense(inputs=_flat, units=dim, activation=activation)
    dense = tf.nn.l2_normalize(dense)  # important!!
    score = -tf.reduce_sum(tf.square(attr_hs - dense), 1)
    return score


def read_word2vec(file_path, vector_dimension):
    print('\n', file_path)
    word2vec = dict()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n').split(' ')
            if len(line) != vector_dimension + 1:
                continue
            v = np.array(list(map(float, line[1:])), dtype=np.float32)
            word2vec[line[0]] = v
    file.close()
    return word2vec


def clear_attribute_triples(attribute_triples):
    print('\nbefore clear:', len(attribute_triples))
    # step 1
    attribute_triples_new = set()
    attr_num = {}
    for (e, a, _) in attribute_triples:
        ent_num = 1
        if a in attr_num:
            ent_num += attr_num[a]
        attr_num[a] = ent_num
    attr_set = set(attr_num.keys())
    attr_set_new = set()
    for a in attr_set:
        if attr_num[a] >= 10:
            attr_set_new.add(a)
    for (e, a, v) in attribute_triples:
        if a in attr_set_new:
            attribute_triples_new.add((e, a, v))
    attribute_triples = attribute_triples_new
    print('after step 1:', len(attribute_triples))

    # step 2
    attribute_triples_new = []
    literals_number, literals_string = [], []
    for (e, a, v) in attribute_triples:
        if '"^^' in v:
            v = v[:v.index('"^^')]
        if v.endswith('"@en'):
            v = v[:v.index('"@en')]
        if is_number(v):
            literals_number.append(v)
        else:
            literals_string.append(v)
        v = v.replace('.', '').replace('(', '').replace(')', '').replace(',', '').replace('"', '')
        v = v.replace('_', ' ').replace('-', ' ').replace('/', ' ')
        if 'http' in v:
            continue
        attribute_triples_new.append((e, a, v))
    attribute_triples = attribute_triples_new
    print('after step 2:', len(attribute_triples))
    return attribute_triples, literals_number, literals_string


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def generate_neg_attribute_triples(pos_batch, all_triples_set, entity_list, neg_triples_num, neighbor=None):
    if neighbor is None:
        neighbor = dict()
    neg_batch = list()
    for head, attribute, value, w in pos_batch:
        for i in range(neg_triples_num):
            while True:
                neg_head = random.choice(neighbor.get(head, entity_list))
                if (neg_head, attribute, value, w) not in all_triples_set:
                    break
            neg_batch.append((neg_head, attribute, value, w))
    assert len(neg_batch) == neg_triples_num * len(pos_batch)
    return neg_batch


def generate_attribute_triple_batch_queue(triple_list1, triple_list2, triple_set1, triple_set2, entity_list1,
                                          entity_list2, batch_size, steps, out_queue, neighbor1, neighbor2,
                                          neg_triples_num):
    for step in steps:
        pos_batch, neg_batch = generate_attribute_triple_batch(triple_list1, triple_list2, triple_set1, triple_set2,
                                                               entity_list1, entity_list2, batch_size,
                                                               step, neighbor1, neighbor2, neg_triples_num)
        out_queue.put((pos_batch, neg_batch))
    exit(0)


def generate_attribute_triple_batch(triple_list1, triple_list2, triple_set1, triple_set2,
                                    entity_list1, entity_list2, batch_size,
                                    step, neighbor1, neighbor2, neg_triples_num):
    batch_size1 = int(len(triple_list1) / (len(triple_list1) + len(triple_list2)) * batch_size)
    batch_size2 = batch_size - batch_size1
    pos_batch1 = bat.generate_pos_triples(triple_list1, batch_size1, step)
    pos_batch2 = bat.generate_pos_triples(triple_list2, batch_size2, step)
    neg_batch1 = generate_neg_attribute_triples(pos_batch1, triple_set1, entity_list1,
                                                neg_triples_num, neighbor=neighbor1)
    neg_batch2 = generate_neg_attribute_triples(pos_batch2, triple_set2, entity_list2,
                                                neg_triples_num, neighbor=neighbor2)
    return pos_batch1 + pos_batch2, neg_batch1 + neg_batch2


def positive_loss_with_weight(phs, pas, pvs, pws):
    pos_distance = phs + pas - pvs
    pos_score = -tf.reduce_sum(tf.square(pos_distance), axis=1)
    pos_score = tf.log(1 + tf.exp(-pos_score))
    pos_score = tf.multiply(pos_score, pws)
    pos_loss = tf.reduce_sum(pos_score)
    return pos_loss


def alignment_loss(ents1, ents2):
    distance = ents1 - ents2
    loss = tf.reduce_sum(tf.reduce_sum(tf.square(distance), axis=1))
    return loss


def space_mapping_loss(view_embeds, shared_embeds, mapping, eye, orthogonal_weight, norm_w=0.0001):
    mapped_ents2 = tf.matmul(view_embeds, mapping)
    mapped_ents2 = tf.nn.l2_normalize(mapped_ents2)
    map_loss = tf.reduce_sum(tf.reduce_sum(tf.square(shared_embeds - mapped_ents2), 1))
    norm_loss = tf.reduce_sum(tf.reduce_sum(tf.square(mapping), 1))
    orthogonal_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(tf.matmul(mapping, mapping, transpose_b=True) - eye, 2), 1))
    return map_loss + orthogonal_weight * orthogonal_loss + norm_w * norm_loss


class MultiKE(BasicModel):

    def __init__(self):
        super().__init__()

    def init(self):
        self.entities = self.kgs.kg1.entities_set | self.kgs.kg2.entities_set
        # self.entity_local_name_dict = self._get_local_name_by_name_triple()
        # print('len(self.entity_local_name_dict):', len(self.entity_local_name_dict))
        # self._generate_literal_vectors()
        # self._generate_name_vectors_mat()
        # self._generate_attribute_value_vectors()
        self.predicate_align_model = PredicateAlignModel(self.kgs, self.args)

        self._define_variables()
        self._define_relation_view_graph()
        # self._define_attribute_view_graph()
        self._define_cross_kg_entity_reference_relation_view_graph()
        # self._define_cross_kg_entity_reference_attribute_view_graph()
        self._define_cross_kg_relation_reference_graph()
        # self._define_cross_kg_attribute_reference_graph()
        # self._define_common_space_learning_graph()
        # self._define_space_mapping_graph()
        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)

    def _get_local_name_by_name_triple(self, name_attribute_list=None):
        if name_attribute_list is None:
            if 'D_Y' in self.args.training_data:
                name_attribute_list = {'skos:prefLabel', 'http://dbpedia.org/ontology/birthName'}
            elif 'D_W' in self.args.training_data:
                name_attribute_list = {'http://www.wikidata.org/entity/P373', 'http://www.wikidata.org/entity/P1476'}
            else:
                name_attribute_list = {}

        triples = self.kgs.kg1.local_attribute_triples_set | self.kgs.kg2.local_attribute_triples_set
        id_ent_dict = {}
        for e, e_id in self.kgs.kg1.entities_id_dict.items():
            id_ent_dict[e_id] = e
        for e, e_id in self.kgs.kg2.entities_id_dict.items():
            id_ent_dict[e_id] = e
        print(len(id_ent_dict))

        name_ids = set()
        for a, a_id in self.kgs.kg1.attributes_id_dict.items():
            if a in name_attribute_list:
                name_ids.add(a_id)
        for a, a_id in self.kgs.kg2.attributes_id_dict.items():
            if a in name_attribute_list:
                name_ids.add(a_id)

        for a, a_id in self.kgs.kg1.attributes_id_dict.items():
            if a_id in name_ids:
                print(a)
        for a, a_id in self.kgs.kg2.attributes_id_dict.items():
            if a_id in name_ids:
                print(a)
        print(name_ids)

        local_name_dict = {}
        ents = self.kgs.kg1.entities_set | self.kgs.kg2.entities_set
        print(len(ents))
        for (e, a, v) in triples:
            if a in name_ids:
                local_name_dict[id_ent_dict[e]] = v
        print('after name_ids:', len(local_name_dict))
        for e in ents:
            if id_ent_dict[e] not in local_name_dict:
                    local_name_dict[id_ent_dict[e]] = id_ent_dict[e].split('/')[-1].replace('_', ' ')
        return local_name_dict

    def _generate_literal_vectors(self):
        cleaned_attribute_triples_list1, _, _ = clear_attribute_triples(self.kgs.kg1.local_attribute_triples_list)
        cleaned_attribute_triples_list2, _, _ = clear_attribute_triples(self.kgs.kg2.local_attribute_triples_list)
        value_list = [v for (_, _, v) in cleaned_attribute_triples_list1 + cleaned_attribute_triples_list2]
        local_name_list = list(self.entity_local_name_dict.values())
        self.literal_list = list(set(value_list + local_name_list))
        print('literal num:', len(local_name_list), len(value_list), len(self.literal_list))

        word2vec = read_word2vec(self.args.word2vec_path, self.args.word2vec_dim)
        literal_encoder = LiteralEncoder(self.literal_list, word2vec, self.args, 300)
        self.literal_vectors_mat = literal_encoder.encoded_literal_vector
        assert self.literal_vectors_mat.shape[0] == len(self.literal_list)
        self.literal_id_dic = dict()
        for i in range(len(self.literal_list)):
            self.literal_id_dic[self.literal_list[i]] = i
        assert len(self.literal_list) == len(self.literal_id_dic)

    def _generate_name_vectors_mat(self):
        name_ordered_list = list()
        num = len(self.entities)
        print("total entities:", num)
        entity_id_uris_dic = dict(zip(self.kgs.kg1.entities_id_dict.values(), self.kgs.kg1.entities_id_dict.keys()))
        entity_id_uris_dic2 = dict(zip(self.kgs.kg2.entities_id_dict.values(), self.kgs.kg2.entities_id_dict.keys()))
        entity_id_uris_dic.update(entity_id_uris_dic2)
        print('total entities ids:', len(entity_id_uris_dic))
        assert len(entity_id_uris_dic) == num
        for i in range(num):
            assert i in entity_id_uris_dic
            entity_uri = entity_id_uris_dic.get(i)
            assert entity_uri in self.entity_local_name_dict
            entity_name = self.entity_local_name_dict.get(entity_uri)
            entity_name_index = self.literal_id_dic.get(entity_name)
            name_ordered_list.append(entity_name_index)
        print('name_ordered_list', len(name_ordered_list))
        name_mat = self.literal_vectors_mat[name_ordered_list,]
        print("entity name embeddings mat:", type(name_mat), name_mat.shape)
        if self.args.literal_normalize:
            name_mat = preprocessing.normalize(name_mat)
        self.local_name_vectors = name_mat

    def _generate_attribute_value_vectors(self):
        self.literal_set = set(self.literal_list)
        values_set = set()
        cleaned_attribute_triples_list1, _, _ = clear_attribute_triples(self.kgs.kg1.local_attribute_triples_list)
        cleaned_attribute_triples_list2, _, _ = clear_attribute_triples(self.kgs.kg2.local_attribute_triples_list)
        attribute_triples_list1, attribute_triples_list2 = set(), set()
        for h, a, v in cleaned_attribute_triples_list1:
            if v in self.literal_set:
                values_set.add(v)
                attribute_triples_list1.add((h, a, v))

        for h, a, v in cleaned_attribute_triples_list2:
            if v in self.literal_set:
                values_set.add(v)
                attribute_triples_list2.add((h, a, v))
        print("selected attribute triples", len(attribute_triples_list1), len(attribute_triples_list2))
        values_id_dic = dict()
        values_list = list(values_set)
        num = len(values_list)
        for i in range(num):
            values_id_dic[values_list[i]] = i
        id_attribute_triples1 = set([(h, a, int(values_id_dic[v])) for (h, a, v) in attribute_triples_list1])
        id_attribute_triples2 = set([(h, a, int(values_id_dic[v])) for (h, a, v) in attribute_triples_list2])
        self.kgs.kg1.set_attributes(id_attribute_triples1)
        self.kgs.kg2.set_attributes(id_attribute_triples2)
        sup_triples1, sup_triples2 = generate_sup_attribute_triples(self.kgs.train_links, self.kgs.kg1.av_dict,
                                                                    self.kgs.kg2.av_dict)
        self.kgs.kg1.add_sup_attribute_triples(sup_triples1)
        self.kgs.kg2.add_sup_attribute_triples(sup_triples2)
        num = len(values_id_dic)
        value_ordered_list = list()
        for i in range(num):
            value = values_list[i]
            value_index = self.literal_id_dic.get(value)
            value_ordered_list.append(value_index)
        print('value_ordered_list', len(value_ordered_list))
        value_vectors = self.literal_vectors_mat[value_ordered_list,]
        print("value embeddings mat:", type(value_vectors), value_vectors.shape)
        if self.args.literal_normalize:
            value_vectors = preprocessing.normalize(value_vectors)
        self.value_vectors = value_vectors

    def _define_variables(self):
        # with tf.variable_scope('literal' + 'embeddings'):
        #     self.literal_embeds = tf.constant(self.value_vectors, dtype=tf.float32)
        # with tf.variable_scope('name_view' + 'embeddings'):
        #     self.name_embeds = tf.constant(self.local_name_vectors, dtype=tf.float32)
        with tf.variable_scope('relation_view' + 'embeddings'):
            self.rv_ent_embeds = xavier_init([self.kgs.entities_num, self.args.dim], 'rv_ent_embeds', True)
            self.rel_embeds = xavier_init([self.kgs.relations_num, self.args.dim], 'rel_embeds', True)
        # with tf.variable_scope('attribute_view' + 'embeddings'):
        #     self.av_ent_embeds = xavier_init([self.kgs.entities_num, self.args.dim], 'av_ent_embeds', True)
            # False important!
            # self.attr_embeds = xavier_init([self.kgs.attributes_num, self.args.dim], 'attr_embeds', False)
        with tf.variable_scope('shared' + 'embeddings'):
            self.ent_embeds = self.rv_ent_embeds  # without name and attribute views
        with tf.variable_scope('shared' + 'combination'):
            self.nv_mapping = tf.get_variable('nv_mapping', shape=[self.args.dim, self.args.dim],
                                              initializer=tf.initializers.orthogonal())
            self.rv_mapping = tf.get_variable('rv_mapping', shape=[self.args.dim, self.args.dim],
                                              initializer=tf.initializers.orthogonal())
            self.av_mapping = tf.get_variable('av_mapping', shape=[self.args.dim, self.args.dim],
                                              initializer=tf.initializers.orthogonal())
            self.eye_mat = tf.constant(np.eye(self.args.dim), dtype=tf.float32, name='eye')

    # --- The followings are view-specific embedding models --- #

    def _define_relation_view_graph(self):
        with tf.name_scope('relation_triple_placeholder'):
            self.rel_pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.rel_pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.rel_pos_ts = tf.placeholder(tf.int32, shape=[None])
            self.rel_neg_hs = tf.placeholder(tf.int32, shape=[None])
            self.rel_neg_rs = tf.placeholder(tf.int32, shape=[None])
            self.rel_neg_ts = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('relation_triple_lookup'):
            rel_phs = tf.nn.embedding_lookup(self.rv_ent_embeds, self.rel_pos_hs)
            rel_prs = tf.nn.embedding_lookup(self.rel_embeds, self.rel_pos_rs)
            rel_pts = tf.nn.embedding_lookup(self.rv_ent_embeds, self.rel_pos_ts)
            rel_nhs = tf.nn.embedding_lookup(self.rv_ent_embeds, self.rel_neg_hs)
            rel_nrs = tf.nn.embedding_lookup(self.rel_embeds, self.rel_neg_rs)
            rel_nts = tf.nn.embedding_lookup(self.rv_ent_embeds, self.rel_neg_ts)
        with tf.name_scope('relation_triple_loss'):
            self.relation_loss = logistic_loss(rel_phs, rel_prs, rel_pts, rel_nhs, rel_nrs, rel_nts, 'L2')
            print("relation cv")
            final_phs = tf.nn.embedding_lookup(self.ent_embeds, self.rel_pos_hs)
            final_pts = tf.nn.embedding_lookup(self.ent_embeds, self.rel_pos_ts)
            # name_phs = tf.nn.embedding_lookup(self.name_embeds, self.rel_pos_hs)
            # name_pts = tf.nn.embedding_lookup(self.name_embeds, self.rel_pos_ts)
            align_loss = positive_loss(final_phs, rel_prs, rel_pts, 'L2')
            align_loss += positive_loss(rel_phs, rel_prs, final_pts, 'L2')
            # align_loss += 0.5 * alignment_loss(final_phs, name_phs)
            # align_loss += 0.5 * alignment_loss(final_pts, name_pts)
            self.relation_loss += align_loss
            self.relation_optimizer = generate_optimizer(self.relation_loss, self.args.learning_rate,
                                                         opt=self.args.optimizer)

    def _define_attribute_view_graph(self):
        with tf.name_scope('attribute_triple_placeholder'):
            self.attr_pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.attr_pos_as = tf.placeholder(tf.int32, shape=[None])
            self.attr_pos_vs = tf.placeholder(tf.int32, shape=[None])
            self.attr_pos_ws = tf.placeholder(tf.float32, shape=[None])
        with tf.name_scope('attribute_triple_lookup'):
            attr_phs = tf.nn.embedding_lookup(self.av_ent_embeds, self.attr_pos_hs)
            attr_pas = tf.nn.embedding_lookup(self.attr_embeds, self.attr_pos_as)
            attr_pvs = tf.nn.embedding_lookup(self.literal_embeds, self.attr_pos_vs)
        with tf.variable_scope('cnn'):
            pos_score = conv(attr_phs, attr_pas, attr_pvs, self.args.dim)
            pos_score = tf.log(1 + tf.exp(-pos_score))
            pos_score = tf.multiply(pos_score, self.attr_pos_ws)
            pos_loss = tf.reduce_sum(pos_score)
            print("attribute cv")
            final_phs = tf.nn.embedding_lookup(self.ent_embeds, self.attr_pos_hs)
            name_phs = tf.nn.embedding_lookup(self.name_embeds, self.attr_pos_hs)
            pos_score = conv(final_phs, attr_pas, attr_pvs, self.args.dim)
            pos_loss += tf.reduce_sum(tf.log(1 + tf.exp(-pos_score)))
            pos_loss += 0.5 * alignment_loss(final_phs, name_phs)
            self.attribute_loss = pos_loss
            self.attribute_optimizer = generate_optimizer(self.attribute_loss, self.args.learning_rate,
                                                          opt=self.args.optimizer)

    # --- The followings are cross-kg identity inference --- #

    def _define_cross_kg_entity_reference_relation_view_graph(self):
        with tf.name_scope('cross_kg_relation_triple_placeholder'):
            self.ckge_rel_pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.ckge_rel_pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.ckge_rel_pos_ts = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('cross_kg_relation_triple_lookup'):
            ckge_rel_phs = tf.nn.embedding_lookup(self.rv_ent_embeds, self.ckge_rel_pos_hs)
            ckge_rel_prs = tf.nn.embedding_lookup(self.rel_embeds, self.ckge_rel_pos_rs)
            ckge_rel_pts = tf.nn.embedding_lookup(self.rv_ent_embeds, self.ckge_rel_pos_ts)
        with tf.name_scope('cross_kg_relation_triple_loss'):
            self.ckge_relation_loss = 2 * positive_loss(ckge_rel_phs, ckge_rel_prs, ckge_rel_pts, 'L2')
            self.ckge_relation_optimizer = generate_optimizer(self.ckge_relation_loss, self.args.learning_rate,
                                                              opt=self.args.optimizer)

    def _define_cross_kg_entity_reference_attribute_view_graph(self):
        with tf.name_scope('cross_kg_attribute_triple_placeholder'):
            self.ckge_attr_pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.ckge_attr_pos_as = tf.placeholder(tf.int32, shape=[None])
            self.ckge_attr_pos_vs = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('cross_kg_attribute_triple_lookup'):
            ckge_attr_phs = tf.nn.embedding_lookup(self.av_ent_embeds, self.ckge_attr_pos_hs)
            ckge_attr_pas = tf.nn.embedding_lookup(self.attr_embeds, self.ckge_attr_pos_as)
            ckge_attr_pvs = tf.nn.embedding_lookup(self.literal_embeds, self.ckge_attr_pos_vs)
        with tf.name_scope('cross_kg_attribute_triple_loss'):
            pos_score = conv(ckge_attr_phs, ckge_attr_pas, ckge_attr_pvs, self.args.dim)
            self.ckge_attribute_loss = 2 * tf.reduce_sum(tf.log(1 + tf.exp(-pos_score)))
            self.ckge_attribute_optimizer = generate_optimizer(self.ckge_attribute_loss, self.args.learning_rate,
                                                               opt=self.args.optimizer)

    def _define_cross_kg_relation_reference_graph(self):
        with tf.name_scope('cross_kg_relation_reference_placeholder'):
            self.ckgp_rel_pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.ckgp_rel_pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.ckgp_rel_pos_ts = tf.placeholder(tf.int32, shape=[None])
            self.ckgp_rel_pos_ws = tf.placeholder(tf.float32, shape=[None])
        with tf.name_scope('cross_kg_relation_reference_lookup'):
            ckgp_rel_phs = tf.nn.embedding_lookup(self.rv_ent_embeds, self.ckgp_rel_pos_hs)
            ckgp_rel_prs = tf.nn.embedding_lookup(self.rel_embeds, self.ckgp_rel_pos_rs)
            ckgp_rel_pts = tf.nn.embedding_lookup(self.rv_ent_embeds, self.ckgp_rel_pos_ts)
        with tf.name_scope('cross_kg_relation_reference_loss'):
            self.ckgp_relation_loss = 2 * positive_loss_with_weight(ckgp_rel_phs, ckgp_rel_prs, ckgp_rel_pts,
                                                                    self.ckgp_rel_pos_ws)
            self.ckgp_relation_optimizer = generate_optimizer(self.ckgp_relation_loss, self.args.learning_rate,
                                                              opt=self.args.optimizer)

    def _define_cross_kg_attribute_reference_graph(self):
        with tf.name_scope('cross_kg_attribute_reference_placeholder'):
            self.ckga_attr_pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.ckga_attr_pos_as = tf.placeholder(tf.int32, shape=[None])
            self.ckga_attr_pos_vs = tf.placeholder(tf.int32, shape=[None])
            self.ckga_attr_pos_ws = tf.placeholder(tf.float32, shape=[None])
        with tf.name_scope('cross_kg_attribute_reference_lookup'):
            ckga_attr_phs = tf.nn.embedding_lookup(self.av_ent_embeds, self.ckga_attr_pos_hs)
            ckga_attr_pas = tf.nn.embedding_lookup(self.attr_embeds, self.ckga_attr_pos_as)
            ckga_attr_pvs = tf.nn.embedding_lookup(self.literal_embeds, self.ckga_attr_pos_vs)
        with tf.name_scope('cross_kg_attribute_reference_loss'):
            pos_score = conv(ckga_attr_phs, ckga_attr_pas, ckga_attr_pvs, self.args.dim)
            pos_score = tf.log(1 + tf.exp(-pos_score))
            pos_score = tf.multiply(pos_score, self.ckga_attr_pos_ws)
            pos_loss = tf.reduce_sum(pos_score)
            self.ckga_attribute_loss = pos_loss
            self.ckga_attribute_optimizer = generate_optimizer(self.ckga_attribute_loss, self.args.learning_rate,
                                                               opt=self.args.optimizer)

    # --- The followings are intermediate combination --- #

    def _define_common_space_learning_graph(self):
        with tf.name_scope('cross_name_view_placeholder'):
            self.cn_hs = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('cross_name_view_lookup'):
            final_cn_phs = tf.nn.embedding_lookup(self.ent_embeds, self.cn_hs)
            cn_hs_names = tf.nn.embedding_lookup(self.name_embeds, self.cn_hs)
            cr_hs = tf.nn.embedding_lookup(self.rv_ent_embeds, self.cn_hs)
            ca_hs = tf.nn.embedding_lookup(self.av_ent_embeds, self.cn_hs)
        with tf.name_scope('cross_name_view_loss'):
            self.cross_name_loss = alignment_loss(final_cn_phs, cn_hs_names)
            self.cross_name_loss += alignment_loss(final_cn_phs, cr_hs)
            # self.cross_name_loss += alignment_loss(final_cn_phs, ca_hs)
            self.cross_name_optimizer = generate_optimizer(self.args.cv_weight * self.cross_name_loss,
                                                           self.args.ITC_learning_rate, opt=self.args.optimizer)

    def _define_space_mapping_graph(self):
        with tf.name_scope('final_entities_placeholder'):
            self.entities = tf.placeholder(tf.int32, shape=[self.args.entity_batch_size, ])
        with tf.name_scope('multi_view_entities_lookup'):
            final_ents = tf.nn.embedding_lookup(self.ent_embeds, self.entities)
            nv_ents = tf.nn.embedding_lookup(self.name_embeds, self.entities)
            rv_ents = tf.nn.embedding_lookup(self.rv_ent_embeds, self.entities)
            av_ents = tf.nn.embedding_lookup(self.av_ent_embeds, self.entities)
        with tf.name_scope('mapping_loss'):
            nv_space_mapping_loss = space_mapping_loss(nv_ents, final_ents, self.nv_mapping, self.eye_mat,
                                                       self.args.orthogonal_weight)
            rv_space_mapping_loss = space_mapping_loss(rv_ents, final_ents, self.rv_mapping, self.eye_mat,
                                                       self.args.orthogonal_weight)
            av_space_mapping_loss = space_mapping_loss(av_ents, final_ents, self.av_mapping, self.eye_mat,
                                                       self.args.orthogonal_weight)
            self.shared_comb_loss = nv_space_mapping_loss + rv_space_mapping_loss + av_space_mapping_loss
            opt_vars = [v for v in tf.trainable_variables() if v.name.startswith("shared")]
            self.shared_comb_optimizer = generate_optimizer(self.shared_comb_loss, self.args.learning_rate,
                                                            var_list=opt_vars, opt=self.args.optimizer)
    # --- The followings are training for multi-view embeddings --- #

    def train_relation_view_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1, neighbors2):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        for steps_task in steps_tasks:
            mp.Process(target=bat.generate_relation_triple_batch_queue,
                       args=(self.kgs.kg1.local_relation_triples_list, self.kgs.kg2.local_relation_triples_list,
                             self.kgs.kg1.local_relation_triples_set, self.kgs.kg2.local_relation_triples_set,
                             self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                             self.args.batch_size, steps_task,
                             batch_queue, neighbors1, neighbors2, self.args.neg_triple_num)).start()
        for i in range(triple_steps):
            batch_pos, batch_neg = batch_queue.get()
            batch_loss, _ = self.session.run(fetches=[self.relation_loss, self.relation_optimizer],
                                             feed_dict={self.rel_pos_hs: [x[0] for x in batch_pos],
                                                        self.rel_pos_rs: [x[1] for x in batch_pos],
                                                        self.rel_pos_ts: [x[2] for x in batch_pos],
                                                        self.rel_neg_hs: [x[0] for x in batch_neg],
                                                        self.rel_neg_rs: [x[1] for x in batch_neg],
                                                        self.rel_neg_ts: [x[2] for x in batch_neg]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        random.shuffle(self.kgs.kg1.local_relation_triples_list)
        random.shuffle(self.kgs.kg2.local_relation_triples_list)
        end = time.time()
        print('epoch {} of rel. view, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss, end - start))

    def train_attribute_view_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1, neighbors2):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        for steps_task in steps_tasks:
            mp.Process(target=generate_attribute_triple_batch_queue,
                       args=(self.predicate_align_model.attribute_triples_w_weights1,
                             self.predicate_align_model.attribute_triples_w_weights2,
                             self.predicate_align_model.attribute_triples_w_weights_set1,
                             self.predicate_align_model.attribute_triples_w_weights_set2,
                             self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                             self.args.attribute_batch_size, steps_task,
                             batch_queue, neighbors1, neighbors2, 0)).start()
        for i in range(triple_steps):
            batch_pos, batch_neg = batch_queue.get()
            batch_loss, _ = self.session.run(fetches=[self.attribute_loss, self.attribute_optimizer],
                                             feed_dict={self.attr_pos_hs: [x[0] for x in batch_pos],
                                                        self.attr_pos_as: [x[1] for x in batch_pos],
                                                        self.attr_pos_vs: [x[2] for x in batch_pos],
                                                        self.attr_pos_ws: [x[3] for x in batch_pos]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        random.shuffle(self.predicate_align_model.attribute_triples_w_weights1)
        random.shuffle(self.predicate_align_model.attribute_triples_w_weights2)
        end = time.time()
        print('epoch {} of att. view, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss, end - start))

    # --- The followings are training for cross-kg identity inference --- #

    def train_cross_kg_entity_inference_relation_view_1epo(self, epoch, sup_triples):
        if len(sup_triples) == 0:
            return
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(sup_triples) / self.args.batch_size))
        batch_size = self.args.batch_size if steps > 1 else len(sup_triples)
        for i in range(steps):
            batch_pos = random.sample(sup_triples, batch_size)
            batch_loss, _ = self.session.run(fetches=[self.ckge_relation_loss, self.ckge_relation_optimizer],
                                             feed_dict={self.ckge_rel_pos_hs: [x[0] for x in batch_pos],
                                                        self.ckge_rel_pos_rs: [x[1] for x in batch_pos],
                                                        self.ckge_rel_pos_ts: [x[2] for x in batch_pos]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of cross-kg entity inference in rel. view, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch,
                                                                                                            epoch_loss,
                                                                                                            end - start))

    def train_cross_kg_entity_inference_attribute_view_1epo(self, epoch, sup_triples):
        if len(sup_triples) == 0:
            return
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(sup_triples) / self.args.attribute_batch_size))
        batch_size = self.args.attribute_batch_size if steps > 1 else len(sup_triples)
        for i in range(steps):
            batch_pos = random.sample(sup_triples, batch_size)
            batch_loss, _ = self.session.run(fetches=[self.ckge_attribute_loss, self.ckge_attribute_optimizer],
                                             feed_dict={self.ckge_attr_pos_hs: [x[0] for x in batch_pos],
                                                        self.ckge_attr_pos_as: [x[1] for x in batch_pos],
                                                        self.ckge_attr_pos_vs: [x[2] for x in batch_pos]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of cross-kg entity inference in attr. view, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch,
                                                                                                             epoch_loss,
                                                                                                             end - start))

    def train_cross_kg_relation_inference_1epo(self, epoch, sup_triples):
        if len(sup_triples) == 0:
            return
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(sup_triples) / self.args.batch_size))
        batch_size = self.args.batch_size if steps > 1 else len(sup_triples)
        for i in range(steps):
            batch_pos = random.sample(sup_triples, batch_size)
            batch_loss, _ = self.session.run(fetches=[self.ckgp_relation_loss, self.ckgp_relation_optimizer],
                                             feed_dict={self.ckgp_rel_pos_hs: [x[0] for x in batch_pos],
                                                        self.ckgp_rel_pos_rs: [x[1] for x in batch_pos],
                                                        self.ckgp_rel_pos_ts: [x[2] for x in batch_pos],
                                                        self.ckgp_rel_pos_ws: [x[3] for x in batch_pos]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of cross-kg relation inference in rel. view, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch,
                                                                                                              epoch_loss,
                                                                                                              end - start))

    def train_cross_kg_attribute_inference_1epo(self, epoch, sup_triples):
        if len(sup_triples) == 0:
            return
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(sup_triples) / self.args.attribute_batch_size))
        batch_size = self.args.attribute_batch_size if steps > 1 else len(sup_triples)
        for i in range(steps):
            batch_pos = random.sample(sup_triples, batch_size)
            batch_loss, _ = self.session.run(fetches=[self.ckga_attribute_loss, self.ckga_attribute_optimizer],
                                             feed_dict={self.ckga_attr_pos_hs: [x[0] for x in batch_pos],
                                                        self.ckga_attr_pos_as: [x[1] for x in batch_pos],
                                                        self.ckga_attr_pos_vs: [x[2] for x in batch_pos],
                                                        self.ckga_attr_pos_ws: [x[3] for x in batch_pos]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of cross-kg attribute inference in attr. view, avg. loss: {:.4f}, time: {:.4f}s'
              .format(epoch, epoch_loss, end - start))

    def train_shared_space_mapping_1epo(self, epoch, entities):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(entities) / self.args.entity_batch_size))
        batch_size = self.args.entity_batch_size if steps > 1 else len(entities)
        for i in range(steps):
            batch_pos = random.sample(entities, batch_size)
            batch_loss, _ = self.session.run(fetches=[self.shared_comb_loss, self.shared_comb_optimizer],
                                             feed_dict={self.entities: batch_pos})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of shared space learning, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss,
                                                                                           end - start))

    # --- The followings are training for cross-view inference --- #

    def train_common_space_learning_1epo(self, epoch, entities):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        steps = int(math.ceil(len(entities) / self.args.entity_batch_size))
        batch_size = self.args.entity_batch_size if steps > 1 else len(entities)
        for i in range(steps):
            batch_pos = random.sample(entities, batch_size)
            batch_loss, _ = self.session.run(fetches=[self.cross_name_loss, self.cross_name_optimizer],
                                             feed_dict={self.cn_hs: batch_pos})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        end = time.time()
        print('epoch {} of common space learning, avg. loss: {:.4f}, time: {:.4f}s'.format(epoch, epoch_loss,
                                                                                           end - start))

    def run(self):
        t = time.time()
        relation_triples_num = self.kgs.kg1.local_relation_triples_num + self.kgs.kg2.local_relation_triples_num
        attribute_triples_num = self.kgs.kg1.local_attribute_triples_num + self.kgs.kg2.local_attribute_triples_num
        relation_triple_steps = int(math.ceil(relation_triples_num / self.args.batch_size))
        attribute_triple_steps = int(math.ceil(attribute_triples_num / self.args.batch_size))
        relation_step_tasks = task_divide(list(range(relation_triple_steps)), self.args.batch_threads_num)
        # attribute_step_tasks = task_divide(list(range(attribute_triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        relation_batch_queue = manager.Queue()
        # attribute_batch_queue = manager.Queue()
        cross_kg_relation_triples = self.kgs.kg1.sup_relation_triples_list + self.kgs.kg2.sup_relation_triples_list
        # cross_kg_entity_inference_in_attribute_triples = self.kgs.kg1.sup_attribute_triples_list + \
        #                                                  self.kgs.kg2.sup_attribute_triples_list
        cross_kg_relation_inference = self.predicate_align_model.sup_relation_alignment_triples1 + \
                                      self.predicate_align_model.sup_relation_alignment_triples2
        # cross_kg_attribute_inference = self.predicate_align_model.sup_attribute_alignment_triples1 + \
        #                                self.predicate_align_model.sup_attribute_alignment_triples2
        neighbors1, neighbors2 = None, None
        entity_list = self.kgs.kg1.entities_list + self.kgs.kg2.entities_list

        # valid_temp(self, embed_choice='nv')
        for i in range(1, self.args.max_epoch + 1):
            print('epoch {}:'.format(i))
            self.train_relation_view_1epo(i, relation_triple_steps, relation_step_tasks, relation_batch_queue,
                                          neighbors1, neighbors2)
            # self.train_common_space_learning_1epo(i, entity_list)
            self.train_cross_kg_entity_inference_relation_view_1epo(i, cross_kg_relation_triples)
            if i > self.args.start_predicate_soft_alignment:
                self.train_cross_kg_relation_inference_1epo(i, cross_kg_relation_inference)

            # self.train_attribute_view_1epo(i, attribute_triple_steps, attribute_step_tasks, attribute_batch_queue,
            #                                neighbors1, neighbors2)
            # self.train_common_space_learning_1epo(i, entity_list)
            # self.train_cross_kg_entity_inference_attribute_view_1epo(i, cross_kg_entity_inference_in_attribute_triples)
            # if i > self.args.start_predicate_soft_alignment:
            #     self.train_cross_kg_attribute_inference_1epo(i, cross_kg_attribute_inference)

            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                valid_temp(self, embed_choice='rv')
                # valid_temp(self, embed_choice='av')
                # valid_temp(self, embed_choice='final')
                # valid_temp(self, embed_choice='avg')
                # valid_WVA(self)
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break

            if i >= self.args.start_predicate_soft_alignment and i % 10 == 0:
                self.predicate_align_model.update_predicate_alignment(self.rel_embeds.eval(session=self.session))
                # self.predicate_align_model.update_predicate_alignment(self.attr_embeds.eval(session=self.session),
                #                                                       predicate_type='attribute')
                cross_kg_relation_inference = self.predicate_align_model.sup_relation_alignment_triples1 + \
                                              self.predicate_align_model.sup_relation_alignment_triples2
                # cross_kg_attribute_inference = self.predicate_align_model.sup_attribute_alignment_triples1 + \
                #                                self.predicate_align_model.sup_attribute_alignment_triples2

            # if self.early_stop or i == self.args.max_epoch:
            #     break

            if self.args.neg_sampling == 'truncated' and i % self.args.truncated_freq == 0:
                t1 = time.time()
                assert 0.0 < self.args.truncated_epsilon < 1.0
                neighbors_num1 = int((1 - self.args.truncated_epsilon) * self.kgs.kg1.entities_num)
                neighbors_num2 = int((1 - self.args.truncated_epsilon) * self.kgs.kg2.entities_num)
                if neighbors1 is not None:
                    del neighbors1, neighbors2
                gc.collect()
                neighbors1 = bat.generate_neighbours(self.eval_kg1_useful_ent_embeddings(),
                                                     self.kgs.useful_entities_list1,
                                                     neighbors_num1, self.args.batch_threads_num)
                neighbors2 = bat.generate_neighbours(self.eval_kg2_useful_ent_embeddings(),
                                                     self.kgs.useful_entities_list2,
                                                     neighbors_num2, self.args.batch_threads_num)
                ent_num = len(self.kgs.kg1.entities_list) + len(self.kgs.kg2.entities_list)
                print("\ngenerating neighbors of {} entities costs {:.3f} s.".format(ent_num, time.time() - t1))
                gc.collect()

        # for i in range(1, self.args.shared_learning_max_epoch + 1):
        #     self.train_shared_space_mapping_1epo(i, entity_list)
        #     if i >= self.args.start_valid and i % self.args.eval_freq == 0:
        #         self.valid(self)
        # test_WVA(self)

        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))

