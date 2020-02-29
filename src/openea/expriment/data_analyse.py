import numpy as np
import matplotlib.pyplot as plt
import utils as ut
import os

from test_funcs import eval_alignment_by_div_embed
from param import P


def read_all_file(dir_path):
    all_data = []
    all_file = os.listdir(dir_path)
    for path in all_file:
        with open(dir_path + "/" + path, "r") as f:
            count = 0
            for line in f.readlines():
                temp_data = line.strip("\n").split("\t")
                if len(all_data) <= count:
                    all_data.append([])
                if 5 >= len(temp_data) > 1:
                    all_data[count].append({path.split("-")[0]: [temp_data[1], temp_data[3]]})
                    count += 1
                elif len(temp_data) > 5:
                    all_data[count].append({path.split("-")[0]: [temp_data[1], temp_data[3], temp_data[5]]})
                    count += 1
    return all_data


def get_hit_result(dirpath):
    hit_result = dict()
    all_file = os.listdir(dirpath)
    if len(all_file) > 0:
        lan_type = all_file[0].split("-")[1]
    training_folder = "../DBP15K/" + lan_type + "/0_3/"
    for files in all_file:
        part_dir = files.replace("-", "/")
        output_folder = "../out/" + part_dir
        ent_embeds = np.load(output_folder + "/ent_embeds.npy")
        triples1, triples2, _, _, ref_ent1, ref_ent2, total_triples_num, total_ent_num, total_rel_num = ut.read_input(
            training_folder)
        embed1 = ent_embeds[ref_ent1,]
        embed2 = ent_embeds[ref_ent2,]
        _, _, result = eval_alignment_by_div_embed(embed1, embed2, P.ent_top_k, accurate=True, is_euclidean=True)
        hit_result[files.split("-")[0]] = result
    return hit_result


def draw_hit_result(dirpath):
    hit_result = get_hit_result(dirpath)
    plt.figure(figsize=(20, 46))
    ylim = (0, 100)
    ylabel = "hit acc"
    y1 = []
    y5 = []
    y10 = []
    x = []
    for key, value in hit_result.items():
        x.append(key)
        y1.append(value[0])
        y5.append(value[1])
        y10.append(value[2])
    plt.ylim(ylim)
    plt.ylabel(ylabel)
    L1, = plt.plot(x, y1, color="#D35400", marker="o")
    L2, = plt.plot(x, y5, color="#2ECC71", marker="o")
    L3, = plt.plot(x, y10, color="#3498DB", marker="o")
    plt.legend(handles=[L1, L2, L3], labels=["hit1", "hit5", "hit10"])


def draw_conicity_result(datapath):
    all_data = read_all_file(datapath)
    #   **************conicity、vs分别位于all_data[15],all_data[16]
    conicity = all_data[15]
    xlabel = []
    ylabel = []
    for i in range(len(conicity)):
        for key, values in conicity[i].items():
            xlabel.append(key)
            ylabel.append(values)
    plt.figure(figsize=(20, 46))
    plt.ylim((0, 0.5))
    plt.ylabel("conicity")
    print([x[0] for x in ylabel])
    L1, = plt.plot(xlabel, [float(x[0]) for x in ylabel], color="#D35400", marker="o", )
    L2, = plt.plot(xlabel, [float(x[1]) for x in ylabel], color="#2ECC71", marker="o", )
    L3, = plt.plot(xlabel, [float(x[2]) for x in ylabel], color="#3498DB", marker="*", )
    plt.legend(handles=[L1, L2, L3], labels=["ent1", "ent2", "all_ents"])


def draw_quartile_deviation_result(datapath):
    all_data = read_all_file(datapath)
    #   **************conicity、vs分别位于all_data[15],all_data[16]
    conicity = all_data[17]
    xlabel = []
    ylabel = []
    for i in range(len(conicity)):
        for key, values in conicity[i].items():
            xlabel.append(key)
            ylabel.append(values)
    plt.figure(figsize=(20, 46))
    plt.ylim((0, 10))
    plt.ylabel("conicity")
    print([x[0] for x in ylabel])
    L1, = plt.plot(xlabel, [float(x[0]) for x in ylabel], color="#D35400", marker="o", )
    L2, = plt.plot(xlabel, [float(x[1]) for x in ylabel], color="#2ECC71", marker="o", )
    plt.legend(handles=[L1, L2], labels=["quar_devi", "mean_value"])


def analyse_all(datapath):
    all_data = read_all_file(datapath)
    plt.figure(figsize=(20, 46))
    ylim_list = [(0, 0.3), (0.3, 1), (0.3, 1), (0, 0.3), (0, 0.3), (0.3, 1), (0.3, 1), (0, 0.3)]
    ylabel_list = ["all_aver_sum", "aver_most_sim", "aver_diff_sum", "aver_diff_var", "align_all_aver_sum"
        , "align_aver_most_sim", "align_aver_diff_sum", "align_aver_diff_var"]
    for i in range(0, 8):
        # for i in range(len(all_data)):
        temp_y = []
        temp_x = []
        aver_temp_y = []
        median_temp_y = []

        plt.subplot(4, 2, i + 1)
        plt.ylim(ylim_list[i])
        plt.ylabel(ylabel_list[i])
        for j in range(len(all_data[i])):
            for key, values in all_data[i][j].items():
                print(key)
                print(values)
                temp_x.append(key)
                aver_temp_y.append(float(values[0]))
                median_temp_y.append(float(values[1]))
        L1, = plt.plot(temp_x, aver_temp_y, color="orange", marker="o", )
        # for xy in zip(temp_x,aver_temp_y):
        #     plt.annotate("(%s,%.6f)"%xy,xy=xy,xytext=(-20,10),textcoords="offset points")
        L2, = plt.plot(temp_x, median_temp_y, color="#2980B9", linestyle="--", marker="o")
        # for xy in zip(temp_x,aver_temp_y):
        #     plt.annotate("(%s,%.6f)"%xy,xy=xy,xytext=(-20,-10),textcoords="offset points")
        # plt.xlabel("method")
        # plt.legend(handles=[l1,l2],labels=["average","median"],loc="best")
        plt.legend(handles=[L1, L2], labels=["average", "median"])
    # plt.show()


def analyse_near_sim(datapath):
    all_data = read_all_file(datapath)
    ylim_list = [(0, 11), (0, 1), (0, 1), (0, 0.3), (0, 0.3)]
    ylabel_list = ["aver_near_number", "aver_near_sim", "aver_near_same_sim", "aver_var_near", "aver_var_near_same"]
    x_label = []
    for i in range(len(all_data[0])):
        for key, _ in all_data[0][i].items():
            x_label.append(key)
    ent1_ann_data = dict()
    ent2_ann_data = dict()
    first_fig = [ent1_ann_data, ent2_ann_data]
    ent1_ans_data = dict()
    ent2_ans_data = dict()
    second_fig = [ent1_ans_data, ent2_ans_data]
    ent1_anss_data = dict()
    ent2_anss_data = dict()
    third_fig = [ent1_anss_data, ent2_anss_data]
    ent1_avn_data = dict()
    ent2_avn_data = dict()
    fourth_fig = [ent1_avn_data, ent2_avn_data]
    ent1_avns_data = dict()
    ent2_avns_data = dict()
    fifth_figure = [ent1_avns_data, ent2_avns_data]
    figure_data = [first_fig, second_fig, third_fig, fourth_fig, fifth_figure]

    for i in range(len(all_data[9])):
        for key, value in all_data[9][i].items():
            ent1_ann_data[key] = value
        for key, value in all_data[12][i].items():
            ent2_ann_data[key] = value
    for i in range(len(all_data[10])):
        for key, value in all_data[10][i].items():
            ent1_ans_data[key] = value[0:2]
            ent1_avn_data[key] = value[-1]
        for key, value in all_data[13][i].items():
            ent2_ans_data[key] = value[0:2]
            ent2_avn_data[key] = value[-1]
    for i in range(len(all_data[11])):
        for key, value in all_data[11][i].items():
            ent1_anss_data[key] = value[0:2]
            ent1_avns_data[key] = value[-1]
        for key, value in all_data[14][i].items():
            ent2_anss_data[key] = value[0:2]
            ent2_avns_data[key] = value[-1]
    plt.figure(figsize=(20, 46))
    for i in range(3):
        plt.subplot(3, 2, i + 1)
        plt.ylim(ylim_list[i])
        plt.ylabel(ylabel_list[i])
        draw_data = figure_data[i]
        x_label = []
        x_data = [[], []]
        median_data = [[], []]
        for key, value in draw_data[0].items():
            x_label.append(key)
            x_data[0].append(float(draw_data[0][key][0]))
            median_data[0].append(float(draw_data[0][key][1]))
            x_data[1].append(float(draw_data[1][key][0]))
            median_data[1].append(float(draw_data[1][key][1]))
        L1, = plt.plot(x_label, x_data[0], color="#D35400", marker="o", )
        L2, = plt.plot(x_label, x_data[1], color="#28B463", marker="o", )
        # L3, = plt.plot(x_label, median_data[0], color="#F39C12", marker="o", )
        # L4, = plt.plot(x_label, median_data[1], color="#2E86C1", marker="o", )
        # plt.legend(handles=[L1, L2, L3, L4], labels=["ent1", "ent2", "ent1_median", "ent2_median"])
        plt.legend(handles=[L1, L2], labels=["ent1", "ent2"])

    # 方差太小几乎分辨不出来  先注释掉
    # for i in range(3,5):
    #     plt.subplot(3, 2, i + 1)
    #     plt.ylim((0,0.02))
    #     # plt.ylim(ylim_list[i])
    #     plt.ylabel(ylabel_list[i])
    #     draw_data = figure_data[i]
    #     x_label = []
    #     x_data = [[], []]
    #     for key, value in draw_data[0].items():
    #         x_label.append(key)
    #         x_data[0].append(float(draw_data[0][key]))
    #         x_data[1].append(float(draw_data[1][key]))
    #     L4, = plt.plot(x_label, x_data[0], color="#D35400", marker="o", )
    #     L5, = plt.plot(x_label, x_data[1], color="#2ECC71", marker="o", )
    #     plt.legend(handles=[L4, L5], labels=["ent1", "ent2",])
    # plt.show()


if __name__ == "__main__":
    dirpath = "../part_analyse/near"
    draw_quartile_deviation_result(dirpath)
    # analyse_near_sim(dirpath)
    draw_hit_result(dirpath)
    plt.show()
