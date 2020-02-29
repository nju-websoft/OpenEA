import numpy as np
import utils as ut
import os
import gc

from param import P
from test_funcs import sim_handler
from test_funcs import eval_alignment_by_sim_mat
from test_funcs import eval_alignment_by_div_embed
import multiprocessing
import time

PROCESS_NUMBER = 15


def pool_collect_dis(dis, a):
    dis_result = list()
    for i in range(dis.shape[0]):
        dis_arg = np.argsort(dis[i])
        dis_result.append(dis_arg[-6:-1])
    return dis_result


def hub_count(ent1, ent2, all_ent, method):
    if method == "GCN_Align":
        ent1_mod = np.sqrt(np.sum(np.square(ent1), axis=1))
        ent2_mod = np.sqrt(np.sum(np.square(ent2), axis=1))
        all_ent_mod = np.sqrt(np.sum(np.square(all_ent), axis=1))
        ent1_mod = ent1_mod.reshape(ent1.shape[0], -1)
        ent2_mod = ent2_mod.reshape(ent2.shape[0], -1)
        all_ent_mod = all_ent_mod.reshape(all_ent.shape[0], -1)
        for i in range(ent1_mod.shape[0]):
            if ent1_mod[i] == 0:
                ent1_mod[i] = 1
            if ent2_mod[i] == 0:
                ent2_mod[i] = 1
        for i in range(all_ent_mod.shape[0]):
            if all_ent_mod[i] == 0:
                all_ent_mod[i] = 1

        ent1_mod = np.tile(ent1_mod, (1, ent1.shape[1]))
        ent2_mod = np.tile(ent2_mod, (1, ent2.shape[1]))
        all_ent_mod = np.tile(all_ent_mod, (1, all_ent.shape[1]))
        ent1 = ent1 / ent1_mod
        ent2 = ent2 / ent2_mod
        all_ent = all_ent / all_ent_mod

    #     ***************************计算实体分布**********************************
    ent1_dis = ent1.dot(all_ent.T)
    part_dis_list = div_list(ent1_dis, 5)
    del ent1_dis
    gc.collect()
    dis_result = list()
    for part_dis in part_dis_list:
        # *************************add multiprocess************
        print("multiprocess")
        pool_result = list()
        tasks = div_list(part_dis, PROCESS_NUMBER)
        del part_dis
        print("divide result ok")
        pool = multiprocessing.Pool(processes=PROCESS_NUMBER)
        for task in tasks:
            pool_result.append(pool.apply_async(pool_collect_dis, (task, 1)))
            del task
            gc.collect()
        pool.close()
        pool.join()
        for item in pool_result:
            temp_pool_result = item.get()
            dis_result.append(temp_pool_result)
    # *******************************************************
    del part_dis_list
    gc.collect()
    appear_array = np.zeros(all_ent.shape[0])
    for part in dis_result:
        for item in part:
            for i in range(item.shape[0]):
                appear_array[item[i]] += 1

    appear_dict = dict()
    appear_dict[0] = 0
    appear_dict[1] = 0
    appear_dict[2] = 0
    for i in range(all_ent.shape[0]):
        if appear_array[i] >= 5:
            appear_dict[0] += 1
        elif 0 < appear_array[i] < 5:
            appear_dict[1] += 1
        else:
            appear_dict[2] += 1
    for key in appear_dict.keys():
        appear_dict[key] = appear_dict[key] / all_ent.shape[0]
    return appear_dict


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


def pool_near_ent(all_ent, ent_dis, ent):
    gc.collect()
    ent_belong_to_one = list()
    ent_near_aver_sim = list()
    ent_same_near_aver_sim = list()
    part_len = ent_dis.shape[0]
    for i in range(0, part_len):
        ent_dis_argsort = np.argsort(ent_dis[i])
        ent1_count = 0
        ent2_count = 0
        ent1_sum = 0
        ent2_sum = 0
        for j in range(1, 11):
            most_sim_ent = all_ent[ent_dis_argsort[-1 - j]].tolist()
            if most_sim_ent in ent:
                ent1_count += 1
                ent1_sum += ent_dis[i][ent_dis_argsort[-1 - j]]
            else:
                ent2_count += 1
                # ent2_sum += ent1_dis[i][ent1_dis_argsort[i][-1 - j]]
                ent2_sum += ent_dis[i][ent_dis_argsort[-1 - j]]
        ent_belong_to_one.append([ent1_count, ent2_count])
        ent_near_aver_sim.append((ent1_sum + ent2_sum) / 10)
        if ent1_count == 0:
            ent_same_near_aver_sim.append(0)
        else:
            ent_same_near_aver_sim.append(ent1_sum / ent1_count)
    return ent_belong_to_one, ent_near_aver_sim, ent_same_near_aver_sim


def near_entity_sim_csls(ent1, ent2, k, hitk, csls=True):
    diff_kg_sim_mat = sim_handler(ent1, ent2, k)
    same_kg_sim_mat = sim_handler(ent1, ent1, k)
    ent2_num = ent2.shape[0]
    sim_mat = np.hstack((diff_kg_sim_mat, same_kg_sim_mat))
    sort_mat = np.argsort(-sim_mat, axis=1)
    diff_number = []
    acc_number = 0
    for i in range(0, ent1.shape[0]):
        diff_number.append(0)
        for j in range(0, 11):
            if sort_mat[i][j] < ent2_num:
                diff_number[i] += 1
        for j in range(0, hitk + 1):
            if sort_mat[i][j] == i:
                acc_number += 1
    acc = acc_number / ent2_num
    eval_alignment_by_sim_mat(ent1, ent2, [1, 2, 5, 10], k)
    print(acc)
    # **************************************不加csls*****************************
    sim_mat1 = ent1.dot(ent2.T)
    sim_mat2 = ent1.dot(ent1.T)
    h_sim_mat = np.hstack((sim_mat1, sim_mat2))
    h_sort_mat = np.argsort(-h_sim_mat, axis=1)
    accu = 0
    for i in range(0, ent1.shape[0]):
        if h_sort_mat[i][1] == i:
            accu += 1
    print("no csls:{}".format(accu / ent2_num))
    diff_number = np.array(diff_number)
    print(np.sum(diff_number) / diff_number.shape[0])


def all_ents_sim(ent1, ent2, method):
    if method == "GCN_Align":
        ent1_mod = np.sqrt(np.sum(np.square(ent1), axis=1))
        ent2_mod = np.sqrt(np.sum(np.square(ent2), axis=1))
        ent1_mod = ent1_mod.reshape(ent1.shape[0], -1)
        ent2_mod = ent2_mod.reshape(ent2.shape[0], -1)
        for i in range(ent1_mod.shape[0]):
            if ent1_mod[i] == 0:
                ent1_mod[i] = 1
            if ent2_mod[i] == 0:
                ent2_mod[i] = 1
        ent1_mod = np.tile(ent1_mod, (1, ent1.shape[1]))
        ent2_mod = np.tile(ent2_mod, (1, ent2.shape[1]))
        ent1 = ent1 / ent1_mod
        ent2 = ent2 / ent2_mod

    result = ent1.dot(ent2.T)
    median_result = np.median(result, axis=1)
    row = result.shape[0]
    col = result.shape[1]

    all_sum = np.sum(result, axis=1)
    aver_sum = all_sum / col
    median_sum = np.median(aver_sum)
    aver_all_sum = np.sum(aver_sum) / row

    most_sim_mat = result.max(axis=1)
    median_most_sim = np.median(most_sim_mat)
    aver_most_sim = np.sum(most_sim_mat) / row
    # *************求所有embedding值与最相似的之间的差值*********
    diff_sum_mat = most_sim_mat * col - all_sum
    diff_sum_mat = diff_sum_mat / (col - 1)
    median_diff_sum = np.median(diff_sum_mat)
    aver_diff_sum = np.sum(diff_sum_mat) / row
    # *******************此处先计算包含最相似的embedding的方差*****
    diff_var_mat = result.var(axis=1)
    median_diff_var = np.median(diff_var_mat)
    aver_diff_var = np.sum(diff_var_mat) / row

    inte_info = [median_result, aver_sum, aver_all_sum, median_sum]
    most_sim_info = [most_sim_mat, aver_most_sim, median_most_sim]
    diff_info = [diff_sum_mat, aver_diff_sum, median_diff_sum]
    var_info = [diff_var_mat, aver_diff_var, median_diff_var]
    return inte_info, most_sim_info, diff_info, var_info


def pool_quartile_deviation(all_ent, ent_dis, ent):
    ent_belong_to_one = list()
    part_len = ent_dis.shape[0]
    for i in range(0, part_len):
        ent1_dis_argsort = np.argsort(ent_dis[i])
        ent1_count = 0
        for j in range(0, 11):
            most_sim_ent = all_ent[ent1_dis_argsort[-1 - j]].tolist()
            if most_sim_ent in ent:
                ent1_count += 1
        ent_belong_to_one.append(ent1_count)
    return ent_belong_to_one


def comp_quartile_deviation(ent1, all_ent):
    ent1_dis = ent1.dot(all_ent.T)
    ent1_num = ent1.shape[0]
    ent1_belong_to_one = []
    ent1 = ent1.tolist()
    # **************************add multiprocess************
    part_dis_list = div_list(ent1_dis, 5)
    del ent1_dis
    gc.collect()
    for part_dis in part_dis_list:
        pool_result = []
        pool = multiprocessing.Pool(processes=PROCESS_NUMBER)
        tasks = div_list(part_dis, PROCESS_NUMBER)
        del part_dis
        for task in tasks:
            pool_result.append(pool.apply_async(pool_quartile_deviation, (all_ent, task, ent1)))
            del task
            gc.collect()
        pool.close()
        pool.join()
        for item in pool_result:
            ent1_belong_to_one.extend(item.get())
    # ******************************************************
    gc.collect()
    ent1_belong_to_one = np.array(ent1_belong_to_one)
    result_argsort = np.argsort(ent1_belong_to_one)
    final_result = np.array(ent1_belong_to_one[result_argsort[int(ent1_num / 4):int(-ent1_num / 4)]])
    quar_result = ent1_belong_to_one[result_argsort[int(-ent1_num / 4)]] - ent1_belong_to_one[
        result_argsort[int(ent1_num / 4)]]
    mean_result = np.mean(final_result)
    return [quar_result, mean_result]


def near_entity_sim(ent1, ent2, align_ent1_id, align_ent2_id, all_ent, method):
    start_time = time.time()
    print("start")
    if method == "GCN_Align":
        ent1_mod = np.sqrt(np.sum(np.square(ent1), axis=1))
        ent2_mod = np.sqrt(np.sum(np.square(ent2), axis=1))
        all_ent_mod = np.sqrt(np.sum(np.square(all_ent), axis=1))
        ent1_mod = ent1_mod.reshape(ent1.shape[0], -1)
        ent2_mod = ent2_mod.reshape(ent2.shape[0], -1)
        all_ent_mod = all_ent_mod.reshape(all_ent.shape[0], -1)
        for i in range(ent1_mod.shape[0]):
            if ent1_mod[i] == 0:
                ent1_mod[i] = 1
            if ent2_mod[i] == 0:
                ent2_mod[i] = 1
        for i in range(all_ent_mod.shape[0]):
            if all_ent_mod[i] == 0:
                all_ent_mod[i] = 1

        ent1_mod = np.tile(ent1_mod, (1, ent1.shape[1]))
        ent2_mod = np.tile(ent2_mod, (1, ent2.shape[1]))
        all_ent_mod = np.tile(all_ent_mod, (1, all_ent.shape[1]))
        ent1 = ent1 / ent1_mod
        ent2 = ent2 / ent2_mod
        all_ent = all_ent / all_ent_mod

    sim_ent_mat = ent1.dot(ent2.T)
    argsort_sim_ent_mat = np.argsort(sim_ent_mat)
    most_sim_ent_eleven = list()
    sim_result = list()
    # ******************************废弃**********************************************
    # t=0
    # for ent1_id,ent2_id in zip(align_ent1_id,align_ent2_id):
    #     if t>=len(most_sim_ent_eleven):
    #         most_sim_ent_eleven.append([])
    #         sim_result.append([])
    #     for i in range(11):
    #         most_sim_ent_eleven[t].append(ent2[argsort_sim_ent_mat[ent1_id][-1-i]])
    #         sim_result[t].append(sim_ent_mat[ent1_id][argsort_sim_ent_mat[ent1_id][-1-i]])
    #     t+=1
    # ******************************************************************************************
    for i in range(ent1.shape[0]):
        temp_sim_result = list()
        temp_most_sim_ent_eleven = list()
        for j in range(10):
            temp_most_sim_ent_eleven.append(ent2[argsort_sim_ent_mat[i][-1 - j]])
            temp_sim_result.append(sim_ent_mat[i][argsort_sim_ent_mat[i][-1 - j]])
        sim_result.append(temp_sim_result)
    sim_result = np.array(sim_result)
    sim_result = np.mean(sim_result, axis=0)
    sim_result = sim_result.tolist()
    # **************************释放numpy内存**************
    del sim_ent_mat
    del argsort_sim_ent_mat
    gc.collect()
    # 计算一个实体最近10个实体在两个KG中的分布
    ent1_dis = ent1.dot(all_ent.T)
    # ent1_dis_argsort = np.argsort(ent1_dis)
    ent1_num = ent1.shape[0]

    ent1_belong_to_one = []
    ent2_belong_to_one = []
    ent1_near_aver_sim = []
    ent2_near_aver_sim = []
    ent1_same_near_aver_sim = []
    ent2_same_near_aver_sim = []
    ent1 = ent1.tolist()

    part_dis_list = div_list(ent1_dis, 5)
    del ent1_dis
    gc.collect()
    for part_dis in part_dis_list:
        # *************************add multiprocess************
        print("multiprocess")
        pool_result = []
        tasks = div_list(part_dis, PROCESS_NUMBER)
        del part_dis
        print("divide result ok")
        pool = multiprocessing.Pool(processes=PROCESS_NUMBER)
        for task in tasks:
            pool_result.append(pool.apply_async(pool_near_ent, (all_ent, task, ent1)))
            del task
            gc.collect()
        pool.close()
        pool.join()
        for item in pool_result:
            temp_pool_result = item.get()
            # ent1_info.append(temp_pool_result)
            ent1_belong_to_one.extend(temp_pool_result[0])
            ent1_near_aver_sim.extend(temp_pool_result[1])
            ent1_same_near_aver_sim.extend(temp_pool_result[2])
    # *******************************************************
    del part_dis_list
    gc.collect()
    # *********************删除ent1相关无用内存*************************
    ent2_dis = ent2.dot(all_ent.T)

    ent2_num = ent2.shape[0]
    ent2 = ent2.tolist()
    part_dis_list = div_list(ent2_dis, 5)
    del ent2_dis
    gc.collect()
    for part_dis in part_dis_list:
        # *************************add multiprocess************
        pool_result = []
        tasks = div_list(part_dis, PROCESS_NUMBER)
        del part_dis
        pool = multiprocessing.Pool(processes=PROCESS_NUMBER)
        for task in tasks:
            pool_result.append(pool.apply_async(pool_near_ent, (all_ent, task, ent2)))
            del task
            gc.collect()
        pool.close()
        pool.join()
        for item in pool_result:
            temp_pool_result = item.get()
            ent2_belong_to_one.extend(temp_pool_result[0])
            ent2_near_aver_sim.extend(temp_pool_result[1])
            ent2_same_near_aver_sim.extend(temp_pool_result[2])
    end_time = time.time()
    print(end_time - start_time)
    return most_sim_ent_eleven, sim_result, [ent1_belong_to_one, ent1_near_aver_sim, ent1_same_near_aver_sim] \
        , [ent2_belong_to_one, ent2_near_aver_sim, ent2_same_near_aver_sim]
    # return most_sim_ent_eleven, sim_result, [ent1_info, ent1_num], [ent2_info, ent2_num]


def pool_near_entity_raw_analyse(ent_belong_to_one, ent_near_aver_sim, ent_same_near_aver_sim):
    # 此处简单的计算平均数、方差、中位数
    # ***********************************************************
    ent_belong_to_one = np.array(ent_belong_to_one)
    ent_aver = np.sum(ent_belong_to_one, axis=0)
    ent_median = np.median(ent_belong_to_one[:][0])
    del ent_belong_to_one

    ent_near_aver_sim = np.array(ent_near_aver_sim)
    ent_aver_near_aver_sim = np.sum(ent_near_aver_sim)
    ent_median_near_aver_sim = np.median(ent_near_aver_sim)
    ent_var_near_aver_sim = ent_near_aver_sim.var()
    del ent_near_aver_sim

    ent_same_near_aver_sim = np.array(ent_same_near_aver_sim)
    ent_aver_same_near_aver_sim = np.sum(ent_same_near_aver_sim)
    ent_median_same_near_aver_sim = np.median(ent_same_near_aver_sim)
    ent_var_same_near_aver_sim = ent_same_near_aver_sim.var()
    del ent_same_near_aver_sim

    ent_inte_info = [ent_aver, ent_median]
    ent_near_info = [ent_aver_near_aver_sim, ent_median_near_aver_sim, ent_var_near_aver_sim]
    ent_same_near_info = [ent_aver_same_near_aver_sim, ent_median_same_near_aver_sim, ent_var_same_near_aver_sim]
    ent_info = [ent_inte_info, ent_near_info, ent_same_near_info]

    return ent_info


def inter_near_entity_analyse(ent1_near_dis, ent2_near_dis):
    ent1_num = ent1_near_dis[1]
    ent1_info_list = ent1_near_dis[0]
    ent1_part_num = len(ent1_info_list)

    ent2_num = ent2_near_dis[1]
    ent2_info_list = ent2_near_dis[0]
    ent2_part_num = len(ent2_info_list)

    inte_list = list()
    near_list = list()
    same_near_list = list()

    for item in ent2_info_list:
        inte_list.append(item[0])
        near_list.append(item[1])
        same_near_list.append(item[2])
    inte_array = np.array(inte_list)
    inte_array = inte_array.sum(axis=0)

    near_array = np.array(near_list)
    near_array = near_array.sum(axis=0)

    same_near_array = np.array(same_near_list)
    same_near_array = same_near_array.sum(axis=0)

    ent2_inte_info = [inte_array[0] / ent2_num, inte_array[1] / ent2_part_num]
    ent2_near_info = [near_array[0] / ent2_num, near_array[1] / ent2_part_num, near_array[2] / ent2_part_num]
    ent2_same_near_info = [same_near_array[0] / ent2_num, same_near_array[1] / ent2_part_num,
                           same_near_array[2] / ent2_part_num]
    ent2_info = [ent2_inte_info, ent2_near_info, ent2_same_near_info]

    inte_list = list()
    near_list = list()
    same_near_list = list()

    for item in ent1_info_list:
        inte_list.append(item[0])
        near_list.append(item[1])
        same_near_list.append(item[2])
    inte_array = np.array(inte_list)
    inte_array = inte_array.sum(axis=0)

    near_array = np.array(near_list)
    near_array = near_array.sum(axis=0)

    same_near_array = np.array(same_near_list)
    same_near_array = same_near_array.sum(axis=0)
    ent1_inte_info = [inte_array[0] / ent1_num, inte_array[1] / ent1_part_num]
    ent1_near_info = [near_array[0] / ent1_num, near_array[1] / ent1_part_num, near_array[2] / ent1_part_num]
    ent1_same_near_info = [same_near_array[0] / ent1_num, same_near_array[1] / ent1_part_num,
                           same_near_array[2] / ent1_part_num]
    ent1_info = [ent1_inte_info, ent1_near_info, ent1_same_near_info]

    return ent1_info, ent2_info


def near_entity_raw_analyse(ent1_near_dis, ent2_near_dis):
    ent1_belong_to_one = ent1_near_dis[0]
    ent1_near_aver_sim = ent1_near_dis[1]
    ent1_same_near_aver_sim = ent1_near_dis[2]
    ent2_belong_to_one = ent2_near_dis[0]
    ent2_near_aver_sim = ent2_near_dis[1]
    ent2_same_near_aver_sim = ent2_near_dis[2]
    # 此处简单的计算平均数、方差、中位数
    # ***********************************************************
    ent1_length = len(ent1_belong_to_one)
    ent1_belong_to_one = np.array(ent1_belong_to_one)
    ent1_aver = np.sum(ent1_belong_to_one, axis=0) / ent1_length
    ent1_median = np.median(ent1_belong_to_one[:][0])

    ent1_near_aver_sim = np.array(ent1_near_aver_sim)
    ent1_aver_near_aver_sim = np.sum(ent1_near_aver_sim) / ent1_length
    ent1_median_near_aver_sim = np.median(ent1_near_aver_sim)
    ent1_var_near_aver_sim = ent1_near_aver_sim.var()

    ent1_same_near_aver_sim = np.array(ent1_same_near_aver_sim)
    ent1_aver_same_near_aver_sim = np.sum(ent1_same_near_aver_sim) / ent1_length
    ent1_median_same_near_aver_sim = np.median(ent1_same_near_aver_sim)
    ent1_var_same_near_aver_sim = ent1_same_near_aver_sim.var()

    ent2_length = len(ent2_belong_to_one)
    ent2_belong_to_one = np.array(ent2_belong_to_one)
    ent2_aver = np.sum(ent2_belong_to_one, axis=0) / ent2_length
    ent2_median = np.median(ent2_belong_to_one[:][0])

    ent2_near_aver_sim = np.array(ent2_near_aver_sim)
    ent2_aver_near_aver_sim = np.sum(ent2_near_aver_sim) / ent2_length
    ent2_median_near_aver_sim = np.median(ent2_near_aver_sim)
    ent2_var_near_aver_sim = ent2_near_aver_sim.var()

    ent2_same_near_aver_sim = np.array(ent2_same_near_aver_sim)
    ent2_aver_same_near_aver_sim = np.sum(ent2_same_near_aver_sim) / ent2_length
    ent2_median_same_near_aver_sim = np.median(ent2_same_near_aver_sim)
    ent2_var_same_near_aver_sim = ent2_same_near_aver_sim.var()

    ent1_inte_info = [ent1_aver, ent1_median]
    ent1_near_info = [ent1_aver_near_aver_sim, ent1_median_near_aver_sim, ent1_var_near_aver_sim]
    ent1_same_near_info = [ent1_aver_same_near_aver_sim, ent1_median_same_near_aver_sim, ent1_var_same_near_aver_sim]
    ent1_info = [ent1_inte_info, ent1_near_info, ent1_same_near_info]

    ent2_inte_info = [ent2_aver, ent2_median]
    ent2_near_info = [ent2_aver_near_aver_sim, ent2_median_near_aver_sim, ent2_var_near_aver_sim]
    ent2_same_near_info = [ent2_aver_same_near_aver_sim, ent2_median_same_near_aver_sim, ent2_var_same_near_aver_sim]
    ent2_info = [ent2_inte_info, ent2_near_info, ent2_same_near_info]
    return ent1_info, ent2_info
    # **************************************************************************************


def ent_conicity(ent1, ent2):
    ent1_mean = np.mean(ent1, axis=0)
    ent2_mean = np.mean(ent2, axis=0)
    all_ents = np.vstack((ent1, ent2))
    all_ents_mean = np.mean(all_ents, axis=0)

    #   ******************计算atm*************************
    ent1_atm = np.dot(ent1, ent1_mean) / np.sqrt(np.sum(np.square(ent1_mean)))
    ent2_atm = np.dot(ent2, ent2_mean) / np.sqrt(np.sum(np.square(ent2_mean)))
    all_ents_atm = np.dot(all_ents, all_ents_mean) / np.sqrt(np.sum(np.square(all_ents_mean)))
    #   ******************计算conicity********************
    ent1_conicity = np.mean(ent1_atm)
    ent2_conicity = np.mean(ent2_atm)
    all_ents_conicity = np.mean(all_ents_atm)
    #   *************************计算VS****************
    ent1_temp_vs = np.square(ent1_atm - ent1_conicity)
    ent1_vs = np.mean(ent1_temp_vs)
    ent2_temp_vs = np.square(ent2_atm - ent2_conicity)
    ent2_vs = np.mean(ent2_temp_vs)
    all_ents_temp_vs = np.square(all_ents_atm - all_ents_conicity)
    all_ents_vs = np.mean(all_ents_temp_vs)

    conicity = [ent1_conicity, ent2_conicity, all_ents_conicity]
    vs = [ent1_vs, ent2_vs, all_ents_vs]
    return conicity, vs


def save_quartile_deviation_result(save_output_file, result):
    save_output_file = save_output_file + "quartile"
    with open(save_output_file, "w", encoding="utf-8") as f:
        f.write("quar_devi:\t{}\tmean_value:\t{}\t".format(result[0], result[1]))


def save_near_ents_sim_result(save_output_file, sim_result, ent1_near_dis, ent2_near_dis):
    # **************************************************************************************
    ent1_info, ent2_info = near_entity_raw_analyse(ent1_near_dis, ent2_near_dis)
    # ent1_info, ent2_info = inter_near_entity_analyse(ent1_near_dis, ent2_near_dis)
    print("write near")
    save_output_file = save_output_file + "nearents"
    with open(save_output_file, "w", encoding="utf-8") as f:
        f.write("sim_result:\t")
        for i in range(9):
            f.write(str(sim_result[i]))
            f.write("\t")
        f.write(str(sim_result[9]))
        # **************接下来是极其枯燥的文件写入工作***********************
        f.write("\n")
        f.write("ent1_aver:\t{}\t ent1_median:\t{}".format(ent1_info[0][0][0], ent1_info[0][1]))
        f.write("\n")
        f.write("ent1_aver_near_aver:\t{}\t ent1_median_near_aver:\t{}\t"
                "ent1_var_near_aver:\t{}".format(ent1_info[1][0],
                                                 ent1_info[1][1],
                                                 ent1_info[1][2]))
        f.write("\n")
        f.write("ent1_aver_same_aver:\t{}\t ent1_median_same_aver:\t{}\t"
                "ent1_var_same_aver:\t{}".format(ent1_info[2][0],
                                                 ent1_info[2][1],
                                                 ent1_info[2][2]))

        f.write("\n")
        f.write("ent2_aver:\t{}\t ent2_median:\t{}".format(ent2_info[0][0][1], ent2_info[0][1]))
        f.write("\n")
        f.write("ent2_aver_near_aver:\t{}\t ent2_median_near_aver:\t{}\t"
                "ent2_var_near_aver:\t{}".format(ent2_info[1][0],
                                                 ent2_info[1][1],
                                                 ent2_info[1][2]))
        f.write("\n")
        f.write("ent2_aver_same_aver:\t{}\t ent2_median_same_aver:\t{}\t"
                "ent2_var_same_aver:\t{}".format(ent2_info[2][0],
                                                 ent2_info[2][1],
                                                 ent2_info[2][2]))


def save_all_ents_sim_result(save_output_file, inte_info, most_sim_info, diff_info, var_info):
    save_output_file = save_output_file + "allents"
    with open(save_output_file, "w", encoding="utf-8") as f:
        f.write("all_aver_sum:\t{}\t median_sum:\t{}".format(inte_info[2], inte_info[3]))
        f.write("\n")
        f.write("aver_most_sim:\t{}\t median_most_sim:\t{}".format(most_sim_info[1], most_sim_info[2]))
        f.write("\n")
        f.write("aver_diff_sum_mat:\t{}\tmedian_diff_sum:\t{}".format(diff_info[1], diff_info[2]))
        f.write("\n")
        f.write("aver_diff_var_mat:\t{}\tmedian_diff_var:\t{}".format(var_info[1], var_info[2]))


def save_conicity(save_output_file, conicity, vs):
    save_output_file = save_output_file + "conicity"
    with open(save_output_file, "w", encoding="utf-8") as f:
        f.write("ent1_conicity:\t{}\tent2_conicity:\t{}\tall_ents_conicity:\t{}\t".format(
            conicity[0], conicity[1], conicity[2]))
        f.write("\n")
        f.write(
            "ent1_vs:\t{}\tent2_vs:\t{}\tall_ents_vs:\t{}\t".format(vs[0], vs[1], vs[2]))


def save_hub_result(save_output_folder, hub_dis):
    save_output_file = save_output_folder+"hub"
    with open(save_output_file, "w", encoding="utf-8") as f:
        f.write("gt5:\t{}\t1to5:\t{}\teq0:\t{}\t".format(
            hub_dis[0], hub_dis[1], hub_dis[2]))


def get_dir(method, param=["en", "de", "15K", "DBP"], root_name="VLDB2020", give_one=False):
    abs_route = os.path.abspath(".")
    abs_route = os.path.normpath(abs_route)
    route_list = abs_route.split(root_name)
    root_route = route_list[0] + root_name

    # ********************************************************
    dataset_route = os.path.normpath(root_route + "/datasets/")
    output_route = os.path.normpath(root_route + "/output/results/" + method)
    dataset_lanua_V1 = "DBP_" + param[0] + "_" + param[3] + "_" + param[1] + "_" + param[2] + "_V1/"
    dataset_lanua_V2 = "DBP_" + param[0] + "_" + param[3] + "_" + param[1] + "_" + param[2] + "_V2/"
    output_route_v1 = os.path.normpath(output_route + "/" + dataset_lanua_V1 + "721_5fold/")
    output_route_v2 = os.path.normpath(output_route + "/" + dataset_lanua_V2 + "721_5fold/")
    # *********************************************************************************
    V1 = []
    V2 = []
    for path in os.listdir(output_route_v1):
        temp_path = output_route_v1 + "/" + path + "/"
        temp_list = []
        for result_file in os.listdir(temp_path):
            temp_list.append(temp_path + result_file + "/")
        V1.append(temp_list)
    for path in os.listdir(output_route_v2):
        temp_path = output_route_v2 + "/" + path + "/"
        temp_list = []
        for result_file in os.listdir(temp_path):
            temp_list.append(temp_path + result_file + "/")
        V2.append(temp_list)
    if give_one:
        return [V1[-1]], [V2[-1]]
    return V1, V2


def get_ent_embedding(dir, testfile, test_method):
    temp_testfile = testfile.split("/")
    temp_save = ""
    for i in range(1, len(temp_testfile) - 2):
        temp_save += "/" + temp_testfile[i]
    testfile = temp_save + "/test_links"

    ent1_name = []
    ent1_id = []
    ent1 = []
    ref_ent1_id = []

    ent2_name = []
    ent2_id = []
    ent2 = []
    ref_ent2_id = []
    matrix = np.load(dir + "ent_embeds.npy")

    with open(dir + "kg1_ent_ids", "r", encoding="utf8") as f:
        for line in f.readlines():
            temp = line.split("\t")
            ent1_name.append(temp[0].strip())
            ent1_id.append(int(temp[1]))
            ent1.append(matrix[int(temp[1])])
    with open(dir + "kg2_ent_ids", "r", encoding="utf8") as f:
        for line in f.readlines():
            temp = line.split("\t")
            ent2_name.append(temp[0].strip())
            ent2_id.append(int(temp[1]))
            ent2.append(matrix[int(temp[1])])
    with open(dir + "alignment_results_12", "r", encoding="utf8") as f:
        for line in f.readlines():
            temp = line.split("\t")
            ref_ent1_id.append(int(temp[0]))
            ref_ent2_id.append(int(temp[1]))
    kg1_name = []
    kg2_name = []
    ref_ent1 = []
    ref_ent2 = []
    with open(testfile, "r", encoding="utf8") as f:
        for line in f.readlines():
            temp = line.split("\t")
            kg1_name.append(temp[0].strip())
            kg2_name.append(temp[1].strip())
    for id in ref_ent1_id:
        name = kg1_name[id]
        pos = ent1_name.index(name)
        mat_id = ent1_id[pos]
        ref_ent1.append(mat_id)
    for id in ref_ent2_id:
        name = kg2_name[id]
        pos = ent2_name.index(name)
        mat_id = ent2_id[pos]
        ref_ent2.append(mat_id)

    ent1 = np.array(ent1)
    ent2 = np.array(ent2)
    if test_method == "MTransE":
        mapping_matric = np.load(dir + "mapping_mat.npy")
        ent1 = ent1.dot(mapping_matric)
        ent1_mod = np.sqrt(np.sum(np.square(ent1), axis=1))
        ent1_mod = ent1_mod.reshape(ent1.shape[0], -1)
        for i in range(ent1_mod.shape[0]):
            if ent1_mod[i] == 0:
                ent1_mod[i] = 1
        ent1_mod = np.tile(ent1_mod, (1, ent1.shape[1]))
        ent1 = ent1 / ent1_mod
        i = 0
        for item in ent1_id:
            matrix[item] = ent1[i]
            i += 1
    return ent1, ent2, ent1_name, ent2_name, ref_ent1, ref_ent2, matrix


def get_mapping_matric(dir):
    matrix = np.load(dir + "ent_embeds.npy")
    return matrix


if __name__ == "__main__":
    root_dir = "VLDB2020"
    output_folder = "analyse_results"

    test_dataset_list = [["en", "fr", "100K", "DBP"], ["en", "de", "100K", "DBP"], ["en", "en", "100K", "WD"],
                         ["en", "en", "100K", "YG"]]
    test_method_list = ["AttrE", "BootEA", "IMUSE", "MultiKE", "SEA", "SimplE", "GCN_Align", "IPTransE", "TransD", "TransH", "RotatE", "ConvE", "ProjE", "HolE", "MTransE"]
    # test_method_list = ["MTransE", "IPTransE"]
    # test_method_list = ["IPTransE", "JAPE", "ProjE", "RotatE", "TransD"]
    # test_method_list = ["RotatE", "TransD"]
    for test_dataset in test_dataset_list:
        for test_method in test_method_list:
            print(test_method)
            # **********************************************
            v1, v2 = get_dir(test_method, test_dataset, root_dir, True)
            all_dataset_kind = [v1, v2]
            for dataset_kind in all_dataset_kind:
                for dataset_fold in dataset_kind:
                    for dataset in dataset_fold:
                        save_output_folder = dataset.replace("results", output_folder)
                        test_file = dataset.replace("output/results/" + test_method, "datasets")
                        if not os.path.exists(save_output_folder):
                            os.makedirs(save_output_folder)
                        # ************************************************************
                        ent1, ent2, ent1_name, ent2_name, ref_ent1, \
                        ref_ent2, ent_embeds = get_ent_embedding(dataset, test_file, test_method)
                        align_ent1 = ent_embeds[ref_ent1]
                        align_ent2 = ent_embeds[ref_ent2]
                        # **********************************测试hub情况************************
                        hub_dis = hub_count(ent1, ent2, ent_embeds, test_method)
                        save_hub_result(save_output_folder, hub_dis)

                        # ***********************************************************
                        # most_sim_ent_eleven, sim_result, \
                        # ent1_near_dis, ent2_near_dis = near_entity_sim(align_ent1, align_ent2,
                        #                                                ref_ent1,
                        #                                                ref_ent2,
                        #                                                ent_embeds, test_method)
                        #
                        # save_near_ents_sim_result(save_output_folder, sim_result, ent1_near_dis, ent2_near_dis)
                        # *************************************************************************************
                        # ********计算所有实体相似度的均值以及方差********************************************
                        # inte_info, most_sim_info, diff_info, var_info = all_ents_sim(align_ent1, align_ent2,
                        #                                                              test_method)
                        # align_inte_info,align_most_sim_info,\
                        # align_diff_inf,align_var_info=all_ents_sim(align_ent1, align_ent2, test_method)
                        # save_all_ents_sim_result(save_output_folder, inte_info, most_sim_info, diff_info, var_info)
                        # ***************************计算csls相关内容***********************************
                        # near_entity_sim_csls(align_ent1,align_ent2,10,10)
                        # *******************计算conicity 以及vs*********************
                        # conicity, vs = ent_conicity(ent1, ent2)
                        # save_conicity(save_output_folder, conicity, vs)
                        # ********************************************************
                        # result = comp_quartile_deviation(align_ent1, ent_embeds)
                        # save_quartile_deviation_result(save_output_folder, result)
