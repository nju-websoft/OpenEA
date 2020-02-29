import numpy as np
import time

from triples import Triples


def read_input(folder):
    triples_set1 = read_triples(folder + 'triples_1')
    triples_set2 = read_triples(folder + 'triples_2')
    triples1 = Triples(triples_set1)
    triples2 = Triples(triples_set2)
    total_ent_num = len(triples1.ents | triples2.ents)
    total_rel_num = len(triples1.props | triples2.props)
    total_triples_num = len(triples1.triple_list) + len(triples2.triple_list)
    print('total ents:', total_ent_num)
    print('total rels:', len(triples1.props), len(triples2.props), total_rel_num)
    print('total triples: %d + %d = %d' % (len(triples1.triples), len(triples2.triples), total_triples_num))
    ref_ent1, ref_ent2 = read_references(folder + 'ref_ent_ids')
    assert len(ref_ent1) == len(ref_ent2)
    print("To aligned entities:", len(ref_ent1))
    sup_ent1, sup_ent2 = read_references(folder + 'sup_ent_ids')
    return triples1, triples2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, total_triples_num, total_ent_num, total_rel_num


def read_dbp15k_input(folder):
    triples_set1 = read_triples(folder + 'triples_1')
    triples_set2 = read_triples(folder + 'triples_2')
    triples1 = Triples(triples_set1)
    triples2 = Triples(triples_set2)
    total_ent_num = len(triples1.ents | triples2.ents)
    total_rel_num = len(triples1.props | triples2.props)
    total_triples_num = len(triples1.triple_list) + len(triples2.triple_list)
    print('total ents:', total_ent_num)
    print('total rels:', len(triples1.props), len(triples2.props), total_rel_num)
    print('total triples: %d + %d = %d' % (len(triples1.triples), len(triples2.triples), total_triples_num))
    ref_ent1, ref_ent2 = read_references(folder + 'ref_pairs')
    assert len(ref_ent1) == len(ref_ent2)
    print("To aligned entities:", len(ref_ent1))
    sup_ent1, sup_ent2 = read_references(folder + 'sup_pairs')
    return triples1, triples2, sup_ent1, sup_ent2, ref_ent1, ref_ent2, total_triples_num, total_ent_num, total_rel_num


def generate_sup_triples(triples1, triples2, ents1, ents2):
    def generate_newly_triples(ent1, ent2, rt_dict1, hr_dict1):
        newly_triples = set()
        for r, t in rt_dict1.get(ent1, set()):
            newly_triples.add((ent2, r, t))
        for h, r in hr_dict1.get(ent1, set()):
            newly_triples.add((h, r, ent2))
        return newly_triples

    assert len(ents1) == len(ents2)
    newly_triples1, newly_triples2 = set(), set()
    for i in range(len(ents1)):
        newly_triples1 |= (generate_newly_triples(ents1[i], ents2[i], triples1.rt_dict, triples1.hr_dict))
        newly_triples2 |= (generate_newly_triples(ents2[i], ents1[i], triples2.rt_dict, triples2.hr_dict))
    print("supervised triples: {}, {}".format(len(newly_triples1), len(newly_triples2)))
    return newly_triples1, newly_triples2


def add_sup_triples(triples1, triples2, sup_ent1, sup_ent2):
    newly_triples1, newly_triples2 = generate_sup_triples(triples1, triples2, sup_ent1, sup_ent2)
    triples1 = Triples(triples1.triples | newly_triples1, ori_triples=triples1.triples)
    triples2 = Triples(triples2.triples | newly_triples2, ori_triples=triples2.triples)
    print("now triples: {}, {}".format(len(triples1.triples), len(triples2.triples)))
    return triples1, triples2


def pair2file(file, pairs):
    with open(file, 'w', encoding='utf8') as f:
        for i, j in pairs:
            f.write(str(i) + '\t' + str(j) + '\n')
        f.close()


def read_triples(file):
    triples = set()
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            print(params)
            assert len(params) == 3
            h = int(params[0])
            r = int(params[1])
            t = int(params[2])
            triples.add((h, r, t))
        f.close()
    return triples


def read_references(file):
    ref1, ref2 = list(), list()
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            params = line.strip('\n').split('\t')
            assert len(params) == 2
            e1 = int(params[0])
            e2 = int(params[1])
            ref1.append(e1)
            ref2.append(e2)
        f.close()
        assert len(ref1) == len(ref2)
    return ref1, ref2


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


def triples2ht_set(triples):
    ht_set = set()
    for h, r, t in triples:
        ht_set.add((h, t))
    print("the number of ht: {}".format(len(ht_set)))
    return ht_set


def merge_dic(dic1, dic2):
    return {**dic1, **dic2}


def generate_adjacency_mat(triples1, triples2, ent_num, sup_ents):
    adj_mat = np.mat(np.zeros((ent_num, len(sup_ents)), dtype=np.int32))
    ht_set = triples2ht_set(triples1) | triples2ht_set(triples2)
    for i in range(ent_num):
        for j in sup_ents:
            if (i, j) in ht_set:
                adj_mat[i, sup_ents.index(j)] = 1
    print("shape of adj_mat: {}".format(adj_mat.shape))
    print("the number of 1 in adjacency matrix: {}".format(np.count_nonzero(adj_mat)))
    return adj_mat


def generate_adj_input_mat(adj_mat, d):
    W = np.random.randn(adj_mat.shape[1], d)
    M = np.matmul(adj_mat, W)
    print("shape of input adj_mat: {}".format(M.shape))
    return M


def generate_ent_attrs_sum(ent_num, ent_attrs1, ent_attrs2, attr_embeddings):
    t1 = time.time()
    ent_attrs_embeddings = None
    for i in range(ent_num):
        attrs_index = list(ent_attrs1.get(i, set()) | ent_attrs2.get(i, set()))
        assert len(attrs_index) > 0
        attrs_embeds = np.sum(attr_embeddings[attrs_index,], axis=0)
        if ent_attrs_embeddings is None:
            ent_attrs_embeddings = attrs_embeds
        else:
            ent_attrs_embeddings = np.row_stack((ent_attrs_embeddings, attrs_embeds))
    print("shape of ent_attr_embeds: {}".format(ent_attrs_embeddings.shape))
    print("generating ent features costs: {:.3f} s".format(time.time() - t1))
    return ent_attrs_embeddings


if __name__ == '__main__':
    l = np.mat([9, 8, 7, 6, 5, 4, 3, 2, 1])
    print(l)
    print(np.argpartition(l, np.array([0, 1, 2])))
    print(np.argpartition(l, 1))
    print(np.argpartition(l, 2))
    print(np.argpartition(l, 3))
    print(np.argpartition(l, 4))

