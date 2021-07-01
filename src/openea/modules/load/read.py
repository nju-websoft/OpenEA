import os

import numpy as np


def load_embeddings(file_name):
    if os.path.exists(file_name):
        return np.load(file_name)
    return None


def sort_elements(triples, elements_set):
    dic = dict()
    for s, p, o in triples:
        if s in elements_set:
            dic[s] = dic.get(s, 0) + 1
        if p in elements_set:
            dic[p] = dic.get(p, 0) + 1
        if o in elements_set:
            dic[o] = dic.get(o, 0) + 1
    # set the frequency of other entities that have no relation triples to zero
    for e in elements_set:
        if e not in dic:
            dic[e] = 0
    # firstly sort by values (i.e., frequencies), if equal, by keys (i.e, URIs)
    sorted_list = sorted(dic.items(), key=lambda x: (x[1], x[0]), reverse=True)
    ordered_elements = [x[0] for x in sorted_list]
    assert len(dic) == len(elements_set)
    return ordered_elements, dic


def generate_sharing_id(train_links, kg1_triples, kg1_elements, kg2_triples, kg2_elements, ordered=True):
    ids1, ids2 = dict(), dict()
    if ordered:
        linked_dic = dict()
        for x, y in train_links:
            linked_dic[y] = x
        kg2_linked_elements = [x[1] for x in train_links]
        kg2_unlinked_elements = set(kg2_elements) - set(kg2_linked_elements)
        ids1, ids2 = generate_mapping_id(kg1_triples, kg1_elements, kg2_triples, kg2_unlinked_elements, ordered=ordered)
        for ele in kg2_linked_elements:
            ids2[ele] = ids1[linked_dic[ele]]
    else:
        index = 0
        for e1, e2 in train_links:
            assert e1 in kg1_elements
            assert e2 in kg2_elements
            ids1[e1] = index
            ids2[e2] = index
            index += 1
        for ele in kg1_elements:
            if ele not in ids1:
                ids1[ele] = index
                index += 1
        for ele in kg2_elements:
            if ele not in ids2:
                ids2[ele] = index
                index += 1
    assert len(ids1) == len(set(kg1_elements))
    assert len(ids2) == len(set(kg2_elements))
    return ids1, ids2


def generate_mapping_id(kg1_triples, kg1_elements, kg2_triples, kg2_elements, ordered=True):
    ids1, ids2 = dict(), dict()
    if ordered:
        kg1_ordered_elements, _ = sort_elements(kg1_triples, kg1_elements)
        kg2_ordered_elements, _ = sort_elements(kg2_triples, kg2_elements)
        n1 = len(kg1_ordered_elements)
        n2 = len(kg2_ordered_elements)
        n = max(n1, n2)
        for i in range(n):
            if i < n1 and i < n2:
                ids1[kg1_ordered_elements[i]] = i * 2
                ids2[kg2_ordered_elements[i]] = i * 2 + 1
            elif i >= n1:
                ids2[kg2_ordered_elements[i]] = n1 * 2 + (i - n1)
            else:
                ids1[kg1_ordered_elements[i]] = n2 * 2 + (i - n2)
    else:
        index = 0
        for ele in kg1_elements:
            if ele not in ids1:
                ids1[ele] = index
                index += 1
        for ele in kg2_elements:
            if ele not in ids2:
                ids2[ele] = index
                index += 1
    assert len(ids1) == len(set(kg1_elements))
    assert len(ids2) == len(set(kg2_elements))
    return ids1, ids2


def uris_list_2ids(uris, ids):
    id_uris = list()
    for u in uris:
        assert u in ids
        id_uris.append(ids[u])
    assert len(id_uris) == len(set(uris))
    return id_uris


def uris_pair_2ids(uris, ids1, ids2):
    id_uris = list()
    for u1, u2 in uris:
        # assert u1 in ids1
        # assert u2 in ids2
        if u1 in ids1 and u2 in ids2:
            id_uris.append((ids1[u1], ids2[u2]))
    # assert len(id_uris) == len(set(uris))
    return id_uris


def uris_relation_triple_2ids(uris, ent_ids, rel_ids):
    id_uris = list()
    for u1, u2, u3 in uris:
        assert u1 in ent_ids
        assert u2 in rel_ids
        assert u3 in ent_ids
        id_uris.append((ent_ids[u1], rel_ids[u2], ent_ids[u3]))
    assert len(id_uris) == len(set(uris))
    return id_uris


def uris_attribute_triple_2ids(uris, ent_ids, attr_ids):
    id_uris = list()
    for u1, u2, u3 in uris:
        assert u1 in ent_ids
        assert u2 in attr_ids
        id_uris.append((ent_ids[u1], attr_ids[u2], u3))
    assert len(id_uris) == len(set(uris))
    return id_uris
    
    
def generate_sup_relation_triples_one_link(e1, e2, rt_dict, hr_dict):
    new_triples = set()
    for r, t in rt_dict.get(e1, set()):
        new_triples.add((e2, r, t))
    for h, r in hr_dict.get(e1, set()):
        new_triples.add((h, r, e2))
    return new_triples


def generate_sup_relation_triples(sup_links, rt_dict1, hr_dict1, rt_dict2, hr_dict2):
    new_triples1, new_triples2 = set(), set()
    for ent1, ent2 in sup_links:
        new_triples1 |= (generate_sup_relation_triples_one_link(ent1, ent2, rt_dict1, hr_dict1))
        new_triples2 |= (generate_sup_relation_triples_one_link(ent2, ent1, rt_dict2, hr_dict2))
    print("supervised relation triples: {}, {}".format(len(new_triples1), len(new_triples2)))
    return new_triples1, new_triples2


def generate_sup_attribute_triples_one_link(e1, e2, av_dict):
    new_triples = set()
    for a, v in av_dict.get(e1, set()):
        new_triples.add((e2, a, v))
    return new_triples


def generate_sup_attribute_triples(sup_links, av_dict1, av_dict2):
    new_triples1, new_triples2 = set(), set()
    for ent1, ent2 in sup_links:
        new_triples1 |= (generate_sup_attribute_triples_one_link(ent1, ent2, av_dict1))
        new_triples2 |= (generate_sup_attribute_triples_one_link(ent2, ent1, av_dict2))
    print("supervised attribute triples: {}, {}".format(len(new_triples1), len(new_triples2)))
    return new_triples1, new_triples2


#
# def generate_input(triples_1_file, triples_2_file,
#                    total_links_file, train_links_file, valid_links_file, test_links_file,
#                    attr_triples_1_file=None, attr_triples_2_file=None,
#                    alignment="sharing"):
#     assert alignment in ["sharing", "mapping", "swapping"]
#     print("training data folder:", triples_1_file, triples_2_file)
#     triples1, ents1, rels1 = read_relation_triples(triples_1_file)
#     triples2, ents2, rels2 = read_relation_triples(triples_2_file)
#     triples_num = len(triples1) + len(triples2)
#     print('total triples: %d + %d = %d' % (len(triples1), len(triples2), triples_num))
#     ent_num = len(ents1) + len(ents2)
#     print("ent num", ent_num)
#     rel_num = len(rels1) + len(rels2)
#     print("rel num", rel_num)
#     total_links = read_links(total_links_file)
#     train_links = read_links(train_links_file)
#     valid_links = read_links(valid_links_file)
#     test_links = read_links(test_links_file)
#     print("train links:", len(train_links))
#     print("valid links:", len(valid_links))
#     print("test links:", len(test_links))
#     if alignment == "sharing":
#         ent_ids1, ent_ids2 = generate_sharing_id(total_links, train_links, valid_links, test_links, ents1, ents2)
#         rel_ids1, rel_ids2 = generate_sharing_id([], [], [], [], rels1, rels2)
#     else:
#         ent_ids1, ent_ids2 = generate_mapping_id(total_links, train_links, valid_links, test_links, ents1, ents2)
#         rel_ids1, rel_ids2 = generate_mapping_id([], [], [], [], rels1, rels2)
#     id_triple1, id_triple2, id_train_links, id_valid_links, id_test_links = \
#         uris2ids(ent_ids1, rel_ids1, ent_ids2, rel_ids2, triples1, triples2, train_links, valid_links, test_links)
#
#     if attr_triples_1_file is not None:
#         attr_triples1, attr_ents1, attrs1 = read_attribute_triples(attr_triples_1_file)
#         attr_triples2, attr_ents2, attrs2 = read_attribute_triples(attr_triples_2_file)
#         attr_triples_num = len(attr_triples1) + len(attr_triples2)
#         print('total attribute triples: %d + %d = %d' % (len(attr_triples1), len(attr_triples2), attr_triples_num))
#         attr_ids1, attr_ids2 = generate_mapping_id([], [], [], [], attrs1, attrs2)
#
#         attr_id_triples1, attr_id_triples2 = attr_uris2ids(ent_ids1, attr_ids1, ent_ids2, attr_ids2, attr_triples1, attr_triples2)
#     else:
#         attr_id_triples1, attr_id_triples2 = set(), set()
#         attr_ids1, attr_ids2 = None, None
#     kg1 = KG(id_triple1, ent_ids1, rel_ids1, attr_triples=attr_id_triples1, attrs_id_dict=attr_ids1)
#     kg2 = KG(id_triple2, ent_ids2, rel_ids2, attr_triples=attr_id_triples2, attrs_id_dict=attr_ids2)
#     if alignment == "swapping":
#         id_new_triples1, id_new_triples2 = generate_sup_triples(kg1, kg2, id_train_links)
#         kg1.add_sup_triples(id_new_triples1)
#         kg2.add_sup_triples(id_new_triples2)
#     kgs = KGs(kg1, kg2, id_train_links, id_valid_links, id_test_links)
#     return kgs


def read_relation_triples(file_path):
    print("read relation triples:", file_path)
    if file_path is None:
        return set(), set(), set()
    triples = set()
    entities, relations = set(), set()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 3
        h = params[0].strip()
        r = params[1].strip()
        t = params[2].strip()
        triples.add((h, r, t))
        entities.add(h)
        entities.add(t)
        relations.add(r)
    return triples, entities, relations


def read_links(file_path):
    print("read links:", file_path)
    links = list()
    refs = list()
    reft = list()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        e1 = params[0].strip()
        e2 = params[1].strip()
        refs.append(e1)
        reft.append(e2)
        links.append((e1, e2))
    assert len(refs) == len(reft)
    return links


def read_dict(file_path):
    file = open(file_path, 'r', encoding='utf8')
    ids = dict()
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        ids[params[0]] = int(params[1])
    file.close()
    return ids


def read_pair_ids(file_path):
    file = open(file_path, 'r', encoding='utf8')
    pairs = list()
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        pairs.append((int(params[0]), int(params[1])))
    file.close()
    return pairs


def pair2file(file, pairs):
    if pairs is None:
        return
    with open(file, 'w', encoding='utf8') as f:
        for i, j in pairs:
            f.write(str(i) + '\t' + str(j) + '\n')
        f.close()


def dict2file(file, dic):
    if dic is None:
        return
    with open(file, 'w', encoding='utf8') as f:
        for i, j in dic.items():
            f.write(str(i) + '\t' + str(j) + '\n')
        f.close()
    print(file, "saved.")


def line2file(file, lines):
    if lines is None:
        return
    with open(file, 'w', encoding='utf8') as f:
        for line in lines:
            f.write(line + '\n')
        f.close()
    print(file, "saved.")


def radio_2file(radio, folder):
    path = folder + str(radio).replace('.', '_')
    if not os.path.exists(path):
        os.makedirs(path)
    return path + '/'


def save_results(folder, rest_12):
    if not os.path.exists(folder):
        os.makedirs(folder)
    pair2file(folder + 'alignment_results_12', rest_12)
    print("Results saved!")


def save_embeddings(folder, kgs, ent_embeds, rel_embeds, attr_embeds, mapping_mat=None, rev_mapping_mat=None):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if ent_embeds is not None:
        np.save(folder + 'ent_embeds.npy', ent_embeds)
    if rel_embeds is not None:
        np.save(folder + 'rel_embeds.npy', rel_embeds)
    if attr_embeds is not None:
        np.save(folder + 'attr_embeds.npy', attr_embeds)
    if mapping_mat is not None:
        np.save(folder + 'mapping_mat.npy', mapping_mat)
    if rev_mapping_mat is not None:
        np.save(folder + 'rev_mapping_mat.npy', rev_mapping_mat)
    dict2file(folder + 'kg1_ent_ids', kgs.kg1.entities_id_dict)
    dict2file(folder + 'kg2_ent_ids', kgs.kg2.entities_id_dict)
    dict2file(folder + 'kg1_rel_ids', kgs.kg1.relations_id_dict)
    dict2file(folder + 'kg2_rel_ids', kgs.kg2.relations_id_dict)
    dict2file(folder + 'kg1_attr_ids', kgs.kg1.attributes_id_dict)
    dict2file(folder + 'kg2_attr_ids', kgs.kg2.attributes_id_dict)
    
    embed2file(folder, 'ent_embeds_txt', ent_embeds, kgs.kg1.entities_id_dict, kgs.kg2.entities_id_dict)
    embed2file(folder, 'rel_embeds_txt', rel_embeds, kgs.kg1.relations_id_dict, kgs.kg2.relations_id_dict)
    embed2file(folder, 'attr_embeds_txt', attr_embeds, kgs.kg1.attributes_id_dict, kgs.kg2.attributes_id_dict)
    
    print("Embeddings saved!")

def embed2file(results_folder, file_name, embedding, kg1_id_dict, kg2_id_dict, seperate=True):
    if embedding is None or kg1_id_dict is None or kg2_id_dict is None:
        return
    if seperate:
        with open(results_folder + 'kg1_' + file_name, 'w', encoding='utf8') as f:
            for entity_uri, entity_index in kg1_id_dict.items():
                f.write(str(entity_uri) + ' ' + ' '.join(map(str, embedding[entity_index])) + '\n')
        with open(results_folder + 'kg2_' + file_name, 'w', encoding='utf8') as f:
            for entity_uri, entity_index in kg2_id_dict.items():
                f.write(str(entity_uri) + ' ' + ' '.join(map(str, embedding[entity_index])) + '\n')
    else:
        with open(results_folder + 'combined_' + file_name, 'w', encoding='utf8') as f:
            for entity_uri, entity_index in kg1_id_dict.items():
                f.write(str(entity_uri) + ' ' + ' '.join(map(str, embedding[entity_index])) + '\n')
            for entity_uri, entity_index in kg2_id_dict.items():
                f.write(str(entity_uri) + ' ' + ' '.join(map(str, embedding[entity_index])) + '\n')

def read_attribute_triples(file_path):
    print("read attribute triples:", file_path)
    if file_path is None:
        return set(), set(), set()
    if file_path is None:
        return set(), set(), set()
    triples = set()
    entities, attributes = set(), set()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip().strip('\n').split('\t')
        if len(params) < 3:
            continue
        head = params[0].strip()
        attr = params[1].strip()
        value = params[2].strip()
        if len(params) > 3:
            for p in params[3:]:
                value = value + ' ' + p.strip()
        value = value.strip().rstrip('.').strip()
        entities.add(head)
        attributes.add(attr)
        triples.add((head, attr, value))
    return triples, entities, attributes


if __name__ == '__main__':
    mydict = {'b': 10, 'c': 10, 'a': 10, 'd': 20}
    sorted_dic = sorted(mydict.items(), key=lambda x: (x[1], x[0]), reverse=True)
    print(sorted_dic, type(sorted_dic))
