import numpy as np
import Levenshtein
from sklearn import preprocessing


def link2dic(links):
    dic1, dic2 = dict(), dict()
    for i, j, w in links:
        dic1[i] = (j, w)
        dic2[j] = (i, w)
    assert len(dic1) == len(dic2)
    return dic1, dic2


def generate_sup_predicate_triples(predicate_links, triples1, triples2):
    link_dic1, link_dic2 = link2dic(predicate_links)
    sup_triples1, sup_triples2 = set(), set()
    for s, p, o in triples1:
        if p in link_dic1:
            sup_triples1.add((s, link_dic1.get(p)[0], o, link_dic1.get(p)[1]))
    for s, p, o in triples2:
        if p in link_dic2:
            sup_triples2.add((s, link_dic2.get(p)[0], o, link_dic2.get(p)[1]))
    return list(sup_triples1), list(sup_triples2)


def add_weights(predicate_links, triples1, triples2, min_w_before):
    link_dic1, link_dic2 = link2dic(predicate_links)
    weighted_triples1, weighted_triples2 = set(), set()
    w = 0.2
    for s, p, o in triples1:
        if p in link_dic1:
            weighted_triples1.add((s, p, o, zoom_weight(link_dic1.get(p)[1], min_w_before)))
        else:
            weighted_triples1.add((s, p, o, w))
    for s, p, o in triples2:
        if p in link_dic2:
            weighted_triples2.add((s, p, o, zoom_weight(link_dic2.get(p)[1], min_w_before)))
        else:
            weighted_triples2.add((s, p, o, w))
    assert len(triples1) == len(weighted_triples1)
    assert len(triples2) == len(weighted_triples2)
    return list(weighted_triples1), list(weighted_triples2), weighted_triples1, weighted_triples2


def init_predicate_alignment(predicate_local_name_dict_1, predicate_local_name_dict_2, predicate_init_sim):
    def get_predicate_match_dict(p_ln_dict_1, p_ln_dict_2):
        predicate_match_dict, sim_dict = {}, {}
        for p1, ln1 in p_ln_dict_1.items():
            match_p2 = ''
            max_sim = 0
            for p2, ln2 in p_ln_dict_2.items():
                sim_p2 = Levenshtein.ratio(ln1, ln2)
                if sim_p2 > max_sim:
                    match_p2 = p2
                    max_sim = sim_p2
            predicate_match_dict[p1] = match_p2
            sim_dict[p1] = max_sim
        return predicate_match_dict, sim_dict

    match_dict_1_2, sim_dict_1 = get_predicate_match_dict(predicate_local_name_dict_1, predicate_local_name_dict_2)
    match_dict_2_1, sim_dict_2 = get_predicate_match_dict(predicate_local_name_dict_2, predicate_local_name_dict_1)

    predicate_match_pairs_set = set()
    predicate_latent_match_pairs_similarity_dict = {}
    for p1, p2 in match_dict_1_2.items():
        if p2 in match_dict_2_1 and match_dict_2_1[p2] == p1:
            predicate_latent_match_pairs_similarity_dict[(p1, p2)] = sim_dict_1[p1]
            if sim_dict_1[p1] > predicate_init_sim:
                predicate_match_pairs_set.add((p1, p2, sim_dict_1[p1]))
                # print(p1, p2, sim_dict_1[p1], sim_dict_2[p2])
    return predicate_match_pairs_set, predicate_latent_match_pairs_similarity_dict


def read_predicate_local_name_file(file_path, relation_set):
    relation_local_name_dict, attribute_local_name_dict = {}, {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n').split('\t')
            assert len(line) == 2
            if line[0] in relation_set:
                relation_local_name_dict[line[0]] = line[1]
            else:
                attribute_local_name_dict[line[0]] = line[1]
    file.close()
    return relation_local_name_dict, attribute_local_name_dict


def predicate2id_matched_pairs(predicate_match_pairs_set, predicate_id_dict_1, predicate_id_dict_2):
    id_match_pairs_set = set()
    for (p1, p2, w) in predicate_match_pairs_set:
        if p1 in predicate_id_dict_1 and p2 in predicate_id_dict_2:
            id_match_pairs_set.add((predicate_id_dict_1[p1], predicate_id_dict_2[p2], w))
    return id_match_pairs_set


def find_predicate_alignment_by_embedding(embed, predicate_list1, predicate_list2, predicate_id_dict1,
                                          predicate_id_dict2):
    embed = preprocessing.normalize(embed)
    sim_mat = np.matmul(embed, embed.T)

    matched_1, matched_2 = {}, {}
    for i in predicate_list1:
        sorted_sim = (-sim_mat[i, :]).argsort()
        for j in sorted_sim:
            if j in predicate_list2:
                matched_1[i] = j
                break
    for j in predicate_list2:
        sorted_sim = (-sim_mat[j, :]).argsort()
        for i in sorted_sim:
            if i in predicate_list1:
                matched_2[j] = i
                break

    id_attr_dict1, id_attr_dict2 = {}, {}
    for a, i in predicate_id_dict1.items():
        id_attr_dict1[i] = a
    for a, i in predicate_id_dict2.items():
        id_attr_dict2[i] = a

    predicate_latent_match_pairs_similarity_dict = {}
    for i, j in matched_1.items():
        if matched_2[j] == i:
            predicate_latent_match_pairs_similarity_dict[(i, j)] = sim_mat[i, j]
    return predicate_latent_match_pairs_similarity_dict


def zoom_weight(weight, min_w_before, min_w_after=0.5):
    weight_new = 1.0 - (1.0 - weight) * (1.0 - min_w_after) / (1.0 - min_w_before)
    return weight_new


def get_local_name(item_set):
    item_local_name_dict = {}
    for item in item_set:
        item_local_name_dict[item] = item.split('/')[-1].replace('_', ' ')
    return item_local_name_dict


class PredicateAlignModel:
    def __init__(self, kgs, args):
        self.kgs = kgs
        self.args = args
        self.relation_name_dict1 = get_local_name(set(self.kgs.kg1.relations_id_dict.keys()))
        self.attribute_name_dict1 = get_local_name(set(self.kgs.kg1.attributes_id_dict.keys()))
        self.relation_name_dict2 = get_local_name(set(self.kgs.kg2.relations_id_dict.keys()))
        self.attribute_name_dict2 = get_local_name(set(self.kgs.kg2.attributes_id_dict.keys()))

        self.relation_id_alignment_set = None
        self.train_relations1, self.train_relations2 = None, None
        self.sup_relation_alignment_triples1, self.sup_relation_alignment_triples2 = None, None
        self.relation_triples_w_weights1, self.relation_triples_w_weights2 = None, None
        self.relation_triples_w_weights_set1, self.relation_triples_w_weights_set2 = None, None

        self.attribute_id_alignment_set = None
        self.train_attributes1, self.train_attributes2 = None, None
        self.sup_attribute_alignment_triples1, self.sup_attribute_alignment_triples2 = None, None
        self.attribute_triples_w_weights1, self.attribute_triples_w_weights2 = None, None
        self.attribute_triples_w_weights_set1, self.attribute_triples_w_weights_set2 = None, None

        self.relation_alignment_set, self.relation_latent_match_pairs_similarity_dict_init = \
            init_predicate_alignment(self.relation_name_dict1, self.relation_name_dict2, self.args.predicate_init_sim)
        self.attribute_alignment_set, self.attribute_latent_match_pairs_similarity_dict_init = \
            init_predicate_alignment(self.attribute_name_dict1, self.attribute_name_dict2, self.args.predicate_init_sim)
        self.relation_alignment_set_init = self.relation_alignment_set
        self.attribute_alignment_set_init = self.attribute_alignment_set
        self.update_relation_triples(self.relation_alignment_set)
        self.update_attribute_triples(self.attribute_alignment_set)

    def update_attribute_triples(self, attribute_alignment_set):
        self.attribute_id_alignment_set = predicate2id_matched_pairs(attribute_alignment_set,
                                                                     self.kgs.kg1.attributes_id_dict,
                                                                     self.kgs.kg2.attributes_id_dict)
        self.train_attributes1 = [a for (a, _, _) in self.attribute_id_alignment_set]
        self.train_attributes2 = [a for (_, a, _) in self.attribute_id_alignment_set]
        self.sup_attribute_alignment_triples1, self.sup_attribute_alignment_triples2 = \
            generate_sup_predicate_triples(self.attribute_id_alignment_set, self.kgs.kg1.local_attribute_triples_list,
                                           self.kgs.kg2.local_attribute_triples_list)
        self.attribute_triples_w_weights1, self.attribute_triples_w_weights2, self.attribute_triples_w_weights_set1, \
        self.attribute_triples_w_weights_set2 = add_weights(self.attribute_id_alignment_set,
                                                            self.kgs.kg1.local_attribute_triples_list,
                                                            self.kgs.kg2.local_attribute_triples_list,
                                                            self.args.predicate_soft_sim)

    def update_relation_triples(self, relation_alignment_set):
        self.relation_id_alignment_set = predicate2id_matched_pairs(relation_alignment_set,
                                                                    self.kgs.kg1.relations_id_dict,
                                                                    self.kgs.kg2.relations_id_dict)
        self.train_relations1 = [a for (a, _, _) in self.relation_id_alignment_set]
        self.train_relations2 = [a for (_, a, _) in self.relation_id_alignment_set]
        self.sup_relation_alignment_triples1, self.sup_relation_alignment_triples2 = \
            generate_sup_predicate_triples(self.relation_id_alignment_set, self.kgs.kg1.local_relation_triples_list,
                                           self.kgs.kg2.local_relation_triples_list)
        self.relation_triples_w_weights1, self.relation_triples_w_weights2, self.relation_triples_w_weights_set1, \
        self.relation_triples_w_weights_set2 = add_weights(self.relation_id_alignment_set,
                                                           self.kgs.kg1.local_relation_triples_list,
                                                           self.kgs.kg2.local_relation_triples_list,
                                                           self.args.predicate_soft_sim)

    def update_predicate_alignment(self, embed, predicate_type='relation', w=0.7):
        if predicate_type == 'relation':
            predicate_list1, predicate_list2 = self.kgs.kg1.relations_list, self.kgs.kg2.relations_list
            predicate_id_dict1, predicate_id_dict2 = self.kgs.kg1.relations_id_dict, self.kgs.kg2.relations_id_dict
            predicate_alignment_set_init = self.relation_alignment_set_init
        else:
            predicate_list1, predicate_list2 = self.kgs.kg1.attributes_list, self.kgs.kg2.attributes_list
            predicate_id_dict1, predicate_id_dict2 = self.kgs.kg1.attributes_id_dict, self.kgs.kg2.attributes_id_dict
            predicate_alignment_set_init = self.attribute_alignment_set_init

        predicate_latent_match_pairs_similarity_dict = \
            find_predicate_alignment_by_embedding(embed, predicate_list1, predicate_list2, predicate_id_dict1,
                                                  predicate_id_dict2)

        predicate_alignment_set = set()
        for (p1, p2, sim_init) in predicate_alignment_set_init:
            p_id_1 = predicate_id_dict1[p1]
            p_id_2 = predicate_id_dict2[p2]
            sim = sim_init
            if (p_id_1, p_id_2) in predicate_latent_match_pairs_similarity_dict:
                sim = w * sim + (1 - w) * predicate_latent_match_pairs_similarity_dict[(p_id_1, p_id_2)]
            if sim > self.args.predicate_soft_sim:
                predicate_alignment_set.add((p1, p2, sim))
        print('update ' + predicate_type + ' alignment:', len(predicate_alignment_set))

        if predicate_type == 'relation':
            self.relation_alignment_set = predicate_alignment_set
            self.update_relation_triples(predicate_alignment_set)
        else:
            self.attribute_alignment_set = predicate_alignment_set
            self.update_attribute_triples(predicate_alignment_set)
