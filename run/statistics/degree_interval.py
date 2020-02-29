from utils import *
import os


def count_ent_degree(triples, is_sorted=False):
    ent_degree = {}
    for (h, _, t) in triples:
        degree = 1
        if h in ent_degree:
            degree += ent_degree[h]
        ent_degree[h] = degree

        degree = 1
        if t in ent_degree:
            degree += ent_degree[t]
        ent_degree[t] = degree
    if is_sorted:
        ent_degree = sorted(ent_degree.items(), key=lambda d: d[1], reverse=True)
        return [e for (e, _) in ent_degree]
    return ent_degree


def filter_pairs_by_degree_interval(pair_degree, degree_interval):
    pair_set = set()
    for pair, degree in pair_degree.items():
        if degree_interval[0] <= degree < degree_interval[1]:
            pair_set.add(pair)
    return pair_set


def gold_standard_compare(gold_set, exp_set):

    right_set = gold_set & exp_set
    print(len(right_set), len(exp_set), len(gold_set))
    if len(right_set) == 0:
        return 0, 0, 0
    p = len(right_set) / len(exp_set)
    r = len(right_set) / len(gold_set)
    f1 = 2*p*r / (p+r)
    return p, r, f1


def count_pair_degree(ent_degree_1, ent_degree_2, links):
    pair_degree = {}
    for (e1, e2) in links:
        pair_degree[(e1, e2)] = (ent_degree_1[e1] + ent_degree_2[e2]) / 2
    return pair_degree


def run(dataset, data_split, method, degree_interval=None):
    if degree_interval is None:
        degree_interval = [1, 6, 11, 16, 21, 1000000]
    data_folder = '../../../datasets/'+dataset+'/'
    result_folder = '../../../output/results/'+method+'/'+dataset+'/'+data_split+'/'
    result_folder += list(os.walk(result_folder))[0][1][0] + '/'
    assert os.path.exists(result_folder)
    assert os.path.exists(data_folder)

    rel_triples_1, _, _ = read_relation_triples(data_folder + 'rel_triples_1')
    rel_triples_2, _, _ = read_relation_triples(data_folder + 'rel_triples_2')
    ent_degree_1 = count_ent_degree(rel_triples_1)
    ent_degree_2 = count_ent_degree(rel_triples_2)
    ent_links = read_links(data_folder+'/'+data_split+'/'+'test_links')
    pair_degree_gold = count_pair_degree(ent_degree_1, ent_degree_2, ent_links)

    id_ent_dict_1, id_ent_dict_2 = id2ent_by_ent_links_index(ent_links)
    aligned_ent_id_pair_set = read_alignment_results(result_folder+'alignment_results_12')
    aligned_ent_pair_set = set([(id_ent_dict_1[e1], id_ent_dict_2[e2]) for (e1, e2) in aligned_ent_id_pair_set])
    pair_degree_exp = count_pair_degree(ent_degree_1, ent_degree_2, aligned_ent_pair_set)

    pairs_gold = filter_pairs_by_degree_interval(pair_degree_gold, [1, 1000000])
    pairs_exp = filter_pairs_by_degree_interval(pair_degree_exp, [1, 1000000])
    p, r, f1 = gold_standard_compare(pairs_gold, pairs_exp)
    print('[%d, %d): [P, R, F1] = [%.4f, %.4f, %.4f]' % (1, 1000000, p, r, f1))

    f1s = []
    ps = []
    rs = []
    for i in range(len(degree_interval)-1):
        pairs_gold = filter_pairs_by_degree_interval(pair_degree_gold, [degree_interval[i], degree_interval[i+1]])
        pairs_exp = filter_pairs_by_degree_interval(pair_degree_exp, [degree_interval[i], degree_interval[i+1]])
        p, r, f1 = gold_standard_compare(pairs_gold, pairs_exp)
        print('[%d, %d): [P, R, F1] = [%.4f, %.4f, %.4f]' % (degree_interval[i], degree_interval[i+1], p, r, f1))
        f1s.append(f1)
        ps.append(p)
        rs.append(r)
    return ps, rs, f1s


if __name__ == '__main__':
    dataset = 'DBP_en_DBP_fr_15K_V1'
    data_split = '721_5fold/1'
    p_r_f1 = 'r'
    methods = ['MTransE', 'IPTransE', 'JAPE', 'KDCoE', 'BootEA', 'GCN_Align', 'AttrE', 'IMUSE', 'SEA', 'RSN4EA',
               'MultiKE', 'RDGCN']
    res = [[0 for i in range(len(methods))] for j in range(4)]
    cnt = 0
    for method in methods:
        ps, rs, f1s = run(dataset, data_split, method, degree_interval=[1, 6, 11, 16, 1000000])
        results = ps
        if p_r_f1 == 'r':
            results = rs
        elif p_r_f1 == 'f1':
            results = f1s
        res[0][cnt] = results[0]
        res[1][cnt] = results[1]
        res[2][cnt] = results[2]
        res[3][cnt] = results[3]
        cnt += 1
    for i in range(4):
        output = ''
        for j in range(len(methods)):
            output += str(res[i][j])
            if j != len(methods) - 1:
                output += '\t'
        print(output)

