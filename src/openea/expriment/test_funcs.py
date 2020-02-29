import multiprocessing

import gc
import os

import numpy as np
import time

from param import P
from utils import div_list, merge_dic
from scipy.spatial.distance import cdist

from itertools import product

g = 1000000000


def test_normal(ent_embeds, rel_embeds, embed1, embed2, ppre_hits1, pre_hits1, out_path, pairs,
                is_final_save=False, is_euclidean=False):
    prec_set1, hits12 = eval_alignment_by_div_embed(embed1, embed2, P.ent_top_k,
                                                    selected_pairs=pairs, is_euclidean=is_euclidean)
    ppre_hits1, pre_hits1, is_early = early_stop(ppre_hits1, pre_hits1, hits12)
    if is_early:
        prec_set1, hits12 = eval_alignment_by_div_embed(embed1, embed2, P.ent_top_k, selected_pairs=pairs, accurate=True)
        if P.is_save:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            pair2file(out_path + "hits11", prec_set1)
            np.save(out_path + "ent_embeds", ent_embeds)
            np.save(out_path + "rel_embeds", rel_embeds)
    elif is_final_save:
        prec_set1, hits12 = eval_alignment_by_div_embed(embed1, embed2, P.ent_top_k, selected_pairs=pairs, accurate=True)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        pair2file(out_path + "hits11", prec_set1)
        np.save(out_path + "ent_embeds", ent_embeds)
        np.save(out_path + "rel_embeds", rel_embeds)
    gc.collect()
    return ppre_hits1, pre_hits1, is_early


def test_csls(ent_embeds, rel_embeds, embed1, embed2, ppre_hits1, pre_hits1, out_path, is_final_save=False):
    if embed1.shape[0] < 20000:
        prec_set1, hits12 = eval_alignment_by_sim_mat(embed1, embed2, P.ent_top_k, csls=P.csls)
    prec_set1, hits12 = eval_alignment_by_div_embed(embed1, embed2, P.ent_top_k)
    ppre_hits1, pre_hits1, is_early = early_stop(ppre_hits1, pre_hits1, hits12)
    if is_early:
        prec_set1, hits12 = eval_alignment_by_sim_mat(embed1, embed2, P.ent_top_k, csls=P.csls, accurate=True)
        if P.is_save:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            pair2file(out_path + "hits11", prec_set1)
            np.save(out_path + "ent_embeds", ent_embeds)
            np.save(out_path + "rel_embeds", rel_embeds)
    elif is_final_save:
        prec_set1, hits12 = eval_alignment_by_sim_mat(embed1, embed2, P.ent_top_k, csls=P.csls, accurate=True)
        if P.is_save:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            pair2file(out_path + "hits11", prec_set1)
            np.save(out_path + "ent_embeds", ent_embeds)
            np.save(out_path + "rel_embeds", rel_embeds)
    gc.collect()
    return ppre_hits1, pre_hits1, is_early


def cal_rank_by_sim_mat(task, sim, top_k, accurate):
    mean = 0
    mrr = 0
    num = [0 for k in top_k]
    prec_set = set()
    for i in range(len(task)):
        ref = task[i]
        if accurate:
            rank = (-sim[i, :]).argsort()
        else:
            rank = np.argpartition(-sim[i, :], np.array(top_k) - 1)
        prec_set.add((ref, rank[0]))
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
    return mean, mrr, num, prec_set

def eval_alignment_by_mcd_sim_mat(embed1,embed2,top_k,csls=0,accurate=False):
    t = time.time()
    sim_mat = sim_handler(embed1, embed2, csls)
    n, m = sim_mat.shape[0], sim_mat.shape[1]
    row_sum = np.sum(sim_mat, axis=1)
    col_sum = np.sum(sim_mat, axis=0)
    # print(type(row_sum), row_sum.shape)
    # print(type(col_sum), col_sum.shape)
    mcd = np.zeros((n, m))
    for i, j in product(range(n), range(m)):
        mu = (row_sum[i,] + col_sum[j,] - sim_mat[i, j]) / (n + m - 1)
        delte = np.square(sim_mat[i, j] - mu)
        mcd[i, j] = delte
    print("********************mcd*********************")
    print(mcd)

    ref_num = mcd.shape[0]
    t_num = [0 for k in top_k]
    t_mean = 0
    t_mrr = 0
    t_prec_set = set()
    tasks = div_list(np.array(range(ref_num)), P.nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_rank_by_sim_mat, (task, mcd[task, :], top_k, accurate)))
    pool.close()
    pool.join()

    for res in reses:
        mean, mrr, num, prec_set = res.get()
        t_mean += mean
        t_mrr += mrr
        t_num += np.array(num)
        t_prec_set |= prec_set
    assert len(t_prec_set) == ref_num
    acc = np.array(t_num) / ref_num * 100
    for i in range(len(acc)):
        acc[i] = round(acc[i], 2)
    t_mean /= ref_num
    t_mrr /= ref_num
    if csls > 0:
        print("csls = {}".format(csls))
    if accurate:
        print("accurate results: hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(top_k, acc, t_mean,
                                                                                                   t_mrr,
                                                                                                   time.time() - t))
    else:
        print("hits@{} = {}, time = {:.3f} s ".format(top_k, acc, time.time() - t))
    hits1 = acc[0]
    del sim_mat
    del mcd
    gc.collect()
    return top_k,acc,t_mean,t_mrr,time.time()-t
    # return t_prec_set, hits1

def eval_alignment_by_sim_mat(embed1, embed2, top_k, csls=0, accurate=False):
    t = time.time()
    sim_mat = sim_handler(embed1, embed2, csls)
    # *****************************************
    print("*******************sim_mat*****************")
    # ***********************************************

    ref_num = sim_mat.shape[0]
    t_num = [0 for k in top_k]
    t_mean = 0
    t_mrr = 0
    t_prec_set = set()
    tasks = div_list(np.array(range(ref_num)), P.nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_rank_by_sim_mat, (task, sim_mat[task, :], top_k, accurate)))
    pool.close()
    pool.join()

    for res in reses:
        mean, mrr, num, prec_set = res.get()
        t_mean += mean
        t_mrr += mrr
        t_num += np.array(num)
        t_prec_set |= prec_set
    assert len(t_prec_set) == ref_num
    acc = np.array(t_num) / ref_num * 100
    for i in range(len(acc)):
        acc[i] = round(acc[i], 2)
    t_mean /= ref_num
    t_mrr /= ref_num
    if csls > 0:
        print("csls = {}".format(csls))
    if accurate:
        print("accurate results: hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(top_k, acc, t_mean,
                                                                                                   t_mrr,
                                                                                                   time.time() - t))
    else:
        print("hits@{} = {}, time = {:.3f} s ".format(top_k, acc, time.time() - t))
    hits1 = acc[0]
    del sim_mat
    gc.collect()
    # return t_prec_set, hits1
    return top_k, acc, t_mean, t_mrr, time.time() - t


def cal_rank_by_div_embed(frags, dic, sub_embed, embed, top_k, accurate, is_euclidean):
    mean = 0
    mrr = 0
    num = np.array([0 for k in top_k])
    mean1 = 0
    mrr1 = 0
    num1 = np.array([0 for k in top_k])
    if is_euclidean:
        sim_mat = sim_hander_ou(sub_embed, embed)
    else:
        sim_mat = np.matmul(sub_embed, embed.T)
    prec_set = set()
    aligned_e = None
    for i in range(len(frags)):
        ref = frags[i]
        if accurate:
            rank = (-sim_mat[i, :]).argsort()
        else:
            rank = np.argpartition(-sim_mat[i, :], np.array(top_k) - 1)
        aligned_e = rank[0]
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
        # del rank

        if dic is not None and dic.get(ref, -1) > -1:
            e2 = dic.get(ref)
            sim_mat[i, e2] += 1.0
            rank = (-sim_mat[i, :]).argsort()
            aligned_e = rank[0]
            assert ref in rank
            rank_index = np.where(rank == ref)[0][0]
            mean1 += (rank_index + 1)
            mrr1 += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    num1[j] += 1
            # del rank
        else:
            mean1 += (rank_index + 1)
            mrr1 += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    num1[j] += 1

        prec_set.add((ref, aligned_e))
    del sim_mat
    return mean, mrr, num, mean1, mrr1, num1, prec_set


def eval_alignment_by_div_embed(embed1, embed2, top_k, selected_pairs=None, accurate=False, is_euclidean=False):
    def pair2dic(pairs):
        if pairs is None or len(pairs) == 0:
            return None
        dic = dict()
        for i, j in pairs:
            if i not in dic.keys():
                dic[i] = j
        assert len(dic) == len(pairs)
        return dic
    t = time.time()
    dic = pair2dic(selected_pairs)
    ref_num = embed1.shape[0]
    t_num = np.array([0 for k in top_k])
    t_mean = 0
    t_mrr = 0
    t_num1 = np.array([0 for k in top_k])
    t_mean1 = 0
    t_mrr1 = 0
    t_prec_set = set()
    frags = div_list(np.array(range(ref_num)), P.nums_threads)
    pool = multiprocessing.Pool(processes=len(frags))
    reses = list()
    for frag in frags:
        reses.append(pool.apply_async(cal_rank_by_div_embed, (frag, dic, embed1[frag, :],
                                                              embed2, top_k, accurate, is_euclidean)))
    pool.close()
    pool.join()
    for res in reses:
        mean, mrr, num, mean1, mrr1, num1, prec_set = res.get()
        t_mean += mean
        t_mrr += mrr
        t_num += num
        t_mean1 += mean1
        t_mrr1 += mrr1
        t_num1 += num1
        t_prec_set |= prec_set

    assert len(t_prec_set) == ref_num

    acc = t_num / ref_num * 100
    for i in range(len(acc)):
        acc[i] = round(acc[i], 2)
    t_mean /= ref_num
    t_mrr /= ref_num
    if accurate:
        print("accurate results: hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(top_k, acc,
                                                                                                   t_mean, t_mrr,
                                                                                                   time.time() - t))
    else:
        print("hits@{} = {}, time = {:.3f} s ".format(top_k, acc, time.time() - t))
    hits1 = acc[0]
    if selected_pairs is not None and len(selected_pairs) > 0:
        acc1 = t_num1 / ref_num * 100
        for i in range(len(acc1)):
            acc1[i] = round(acc1[i], 2)
        t_mean1 /= ref_num
        t_mrr1 /= ref_num
        hits1 = acc1[0]
        if accurate:
            print("accurate results: hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(top_k, acc,
                                                                                                       t_mean, t_mrr,
                                                                                                       time.time() - t))
        else:
            print("hits@{} = {}, time = {:.3f} s ".format(top_k, acc, time.time() - t))
    gc.collect()
    return t_prec_set, hits1, acc


def early_stop(ppre_hits1, pre_hits1, hits1, small=True):
    if small:
        if hits1 <= pre_hits1 <= ppre_hits1:
            print("\n == should early stop == \n")
            return pre_hits1, hits1, True
        else:
            return pre_hits1, hits1, False
    else:
        if hits1 <= pre_hits1:
            print("\n == should early stop == \n")
            return pre_hits1, hits1, True
        else:
            return pre_hits1, hits1, False


def pair2file(file, pairs):
    with open(file, 'w', encoding='utf8') as f:
        for i, j in pairs:
            f.write(str(i) + '\t' + str(j) + '\n')
        f.close()


def cal_csls_sim(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    sim_values = np.mean(nearest_k, axis=1)
    return sim_values


def CSLS_sim(sim_mat1, k):
    # sorted_mat = -np.partition(-sim_mat1, k, axis=1) # -np.sort(-sim_mat1)
    # nearest_k = sorted_mat[:, 0:k]
    # sim_values = np.mean(nearest_k, axis=1)

    tasks = div_list(np.array(range(sim_mat1.shape[0])), P.nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_csls_sim, (sim_mat1[task, :], k)))
    pool.close()
    pool.join()
    sim_values = None
    for res in reses:
        val = res.get()
        if sim_values is None:
            sim_values = val
        else:
            sim_values = np.append(sim_values, val)
    assert sim_values.shape[0] == sim_mat1.shape[0]
    return sim_values


def sim_handler(embed1, embed2, k):
    sim_mat = np.matmul(embed1, embed2.T)
    if k <= 0:
        print("k = 0")
        return sim_mat
    csls1 = CSLS_sim(sim_mat, k)
    csls2 = CSLS_sim(sim_mat.T, k)
    # for i in range(sim_mat.shape[0]):
    #     for j in range(sim_mat.shape[1]):
    #         sim_mat[i][j] = 2 * sim_mat[i][j] - csls1[i] - csls2[j]
    # return sim_mat
    csls_sim_mat = 2 * sim_mat.T - csls1
    csls_sim_mat = csls_sim_mat.T - csls2
    del sim_mat
    gc.collect()
    return csls_sim_mat


def sim_hander_ou(embed1, embed2):
    return -cdist(embed1, embed2, metric='euclidean')


if __name__ == '__main__':
    m1 = np.array([[1, 1], [1, 2], [1, 2]])
    print(m1[:, 0:2])
    print(m1, type(m1), m1.shape)
    m2 = np.array([1, 2])
    print(m2, type(m2), m2.shape)
    print(np.mat(m2).shape)
    print(m1 - m2)
    print(type(m1 - np.mat(m2).T))

    print(m1.argsort())
    print(np.mat(m1[0, :]).argsort())

