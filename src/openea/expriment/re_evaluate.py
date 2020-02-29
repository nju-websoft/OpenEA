import numpy as np
import utils as ut
import matplotlib.pyplot as plt
import os
from param import P
from test_funcs import eval_alignment_by_div_embed, eval_alignment_by_sim_mat,eval_alignment_by_mcd_sim_mat
from itertools import product

def mcd_matrix(sim_matrix):
    n, m = sim_matrix.shape[0], sim_matrix.shape[1]
    row_sum = np.sum(sim_matrix, axis=1)
    col_sum = np.sum(sim_matrix, axis=0)
    # print(type(row_sum), row_sum.shape)
    # print(type(col_sum), col_sum.shape)
    mcd = np.zeros((n, m))
    for i, j in product(range(n), range(m)):
        mu = (row_sum[i,] + col_sum[j,] - sim_matrix[i, j]) / (n + m - 1)
        delte = np.square(sim_matrix[i, j] - mu)
        mcd[i, j] = delte
    return mcd

def read_dir(dir_file,parent_dir):
    c_parent_dir=parent_dir
    for every_file in os.listdir(dir_file):
        path=os.path.join(dir_file,every_file)
        if os.path.isdir(path):
            read_dir(path,c_parent_dir)
        else:
            if dir_file not in c_parent_dir:
                c_parent_dir.append(dir_file)
    return c_parent_dir

if __name__ == '__main__':

    parent_dir=[]
    all_out_folder=read_dir("out",parent_dir)
    result_dict={}

    for output_folder in all_out_folder:
        training_folder = "./DBP15K/zh_en/0_3/"
        # output_folder = "out/BiasE_BP/zh_en/20180817115215/"
        ent_embeds = np.load(output_folder + "\\ent_embeds.npy")
        _, _, _, _, ref_ent1, ref_ent2, _, _, _ = ut.read_input(training_folder)
        # embed1,embed2对齐的实体embedding
        embed1 = ent_embeds[ref_ent1, ]
        embed2 = ent_embeds[ref_ent2, ]
        # sim_mat=embed1.dot(embed2.T)
        # print(sim_mat[0,])

        # eval_alignment_by_div_embed(embed1, embed2, P.ent_top_k, accurate=True, is_euclidean=True)
        # print("accurate results with CSLS:")
        top_k, acc, t_mean, t_mrr, run_time=eval_alignment_by_sim_mat(embed1, embed2, P.ent_top_k, csls=P.csls, accurate=True)
        temp_save="accurate results: hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(top_k, acc, t_mean,
                                                                                             t_mrr,run_time)
        mcd_top_k, mcd_acc, mcd_t_mean, mcd_t_mrr, mcd_run_time=eval_alignment_by_mcd_sim_mat(embed1,embed2,P.ent_top_k,csls=P.csls,accurate=True)
        mcd_temp_save="\naccurate results: hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(mcd_top_k, mcd_acc, mcd_t_mean,
                                                                                             mcd_t_mrr,mcd_run_time)
        result_dict[output_folder]=temp_save+mcd_temp_save
    with open("result.txt","w",encoding="utf-8") as f:
        for key,value in result_dict.items():
            f.write(key)
            f.write("\n")
            f.write(value)
            f.write("\n")
