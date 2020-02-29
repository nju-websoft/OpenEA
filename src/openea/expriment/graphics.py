import os
import numpy as np
import matplotlib.pylab as plt
import matplotlib
import xlrd

from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm


def split_data(data):
    result = list()
    temp = data.split("\t")
    for i in range(len(temp)):
        if temp[i] != "":
            result.append(temp[i])
    return result


def get_data(route, method):
    data = dict()
    with open(route + "/" + method, "r", encoding="utf8") as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split(":")
            if len(line) == 2:
                data[line[0]] = split_data(line[1])
            else:
                for i in range(len(line) - 2):
                    data_key = split_data(line[i])[-1]
                    data_content = split_data(line[i + 1])[0:-1]
                    data[data_key] = data_content
                # print(split_data(line[-2]))
                last_data_key = split_data(line[-2])[-1]
                last_data_content = split_data(line[-1])
                data[last_data_key] = last_data_content
    return data


def get_all_data(dataset, method, root_name="VLDB2020"):
    abs_route = os.path.abspath(".")
    abs_route = os.path.normpath(abs_route)
    route_list = abs_route.split(root_name)
    root_route = route_list[0] + root_name

    dataset_root = os.path.normpath(root_route + "/output/analyse_results")

    data_list = dict()
    for item in dataset:
        dataset_dir = os.path.normpath(dataset_root + "/" + item + "/")
        data_list[item] = dict()
        fold_list = list()
        dirs = os.listdir(dataset_dir)
        data_db_list = dict()
        for next_dir in dirs:
            data_db_list[next_dir] = dict()
            temp = os.path.join(dataset_dir, next_dir + "/721_5fold")
            fold_list.append(temp)
        for fold in fold_list:
            data_fold_list = dict()
            for i in range(1, 6):
                temp_path = os.path.normpath(fold + "/" + str(i))
                if not os.path.exists(temp_path):
                    continue
                last_file = os.listdir(temp_path)
                for file in last_file:
                    data_final_dir = os.path.normpath(temp_path + "/" + file)
                    if not os.path.exists(data_final_dir + "/" + method):
                        continue
                    data_fold_list[file] = get_data(data_final_dir, method)
            data_db_list[fold.split("/")[-2]] = data_fold_list
        data_list[item] = data_db_list
    return data_list


def get_graphics_data(data, dataset_list, need):
    compare_method = []
    final_data = []
    for key, value in data.items():
        compare_method.append(key)
        temp_value = []
        for item in dataset_list:
            for need_result in (data[key][item]).values():
                temp_value.append(need_result[need])
            final_data.append(float_data(temp_value))
    return compare_method, final_data


def float_data(data):
    final_data = []
    for item in data:
        if type(item).__name__ == "list":
            temp_data = []
            for part_data in item:
                temp_data.append(float(part_data))
        else:
            temp_data = float(item)
        final_data.append(temp_data)
    np_data = np.array(final_data)
    np_data = np.mean(np_data, axis=0)
    final_data = list(np_data)
    return final_data


def plot_config():
    params = {
        "font.family": "serif",
        "font.serif": "Times New Roman",

    }
    rcParams.update(params)


def config_color_map():
    cdict = {
        "red": (
            (0, 1, 1),
            (0.2, 1, 1),
            (0.5, 0.3, 0.3),
            (0.7, 0, 0),
            (1, 0, 0)
        ),
        "green": (
            (0, 1, 1),
            (0.2, 1, 1),
            (0.5, 0.8, 0.8),
            (0.7, 0.4, 0.4),
            (1, 0.2, 0.2),
        ),
        "blue": (
            (0, 1, 1),
            (0.2, 0.6, 0.6),
            (0.5, 1, 1),
            (0.7, 0.8, 0.8),
            (1, 0.4, 0.4)
        )

    }
    cmap = LinearSegmentedColormap("user_color", cdict, 256)
    return cmap


def grid_figure(x_list, compare_method, analyse_data):
    user_cmap = config_color_map()
    x_position = 0
    plt.figure(figsize=(32, 6))
    for i in range(len(x_list)):
        data = list()
        plt.subplot(1, len(x_list), i + 1)
        plt.xticks([x for x in range(len(compare_method))], compare_method, fontsize=20, rotation=-90, )
        plt.yticks([])
        for j in range(x_position + i, len(analyse_data), len(x_list)):
            data.append(analyse_data[j][0:5])
        pic_data = np.array(data).T
        plt.imshow(pic_data, cmap=user_cmap)
        # sns.heatmap(pic_data)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    # plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.9, bottom=0.3, wspace=0)
    plt.show()


def plot_figure(x_list, compare_method, analyse_data, ran):
    marker = ["o", ".", "p", "s", "d", "*"]
    line_list = []
    x_position = 0
    plt.figure(figsize=(16, 12))
    for i in range(len(x_list)):
        data = list()
        # plt.subplot(len(x_list),2,i+1)
        plt.xticks([x for x in range(len(compare_method))], compare_method, rotation=0)
        plt.yticks(np.arange(ran[0], ran[1], (ran[1] - ran[0]) / 5))
        plt.ylim((ran[0], ran[1]))
        for j in range(x_position + i, len(analyse_data), len(x_list)):
            data.append(analyse_data[j][0])
        line, = plt.plot(compare_method, data, marker=marker[i])
        line_list.append(line)
    plt.legend(line_list, x_list, loc="upper right")
    plt.subplots_adjust(left=0.05, right=0.9, wspace=0)
    plt.show()


# ****************用来测试raw_analyse中分布情况*********************************


def hub_picture(data, dataset, method):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax1.set_ylabel("Proportion", fontsize=16)
    ax2.set_ylabel("Proportion", fontsize=16)


    width = 0.8
    startposition = 0.8
    color_list = ["tomato", 'deepskyblue', "orange"]
    label_list = ["more than 5 times", "1 to 5 times", "never appear"]
    legend_list = list()
    for i in range(len(method)):
        bottom = 0
        for j in range(3):
            bar1, = ax1.bar(startposition+i*width, height=data[j][i][0], width=0.5, bottom=bottom, color=color_list[j], align="edge", edgecolor="black")
            bottom += data[j][i][0]
        ax1.text(startposition+i*width+0.1, y=-0.03, s=method[i], fontsize="14",rotation=-90)
    ax1.text(startposition + (len(method)/2-2) * width, y=1.08, s=dataset[0], fontsize="14")

    for i in range(len(method)):
        bottom = 0
        for j in range(3):
            bar2, = ax2.bar(startposition+i*width, height=data[j][i+len(method)][0], width=0.5, bottom=bottom, color=color_list[j], align="edge", edgecolor="black")
            bottom += data[j][i+len(method)][0]
            if i == 0:
                legend_list.append(bar2)
        ax2.text(startposition + i * width + 0.1, y=-0.03, s=method[i], fontsize="14", rotation=-90)

    ax2.text(startposition + (len(method) / 2 - 2) * width, y=1.08, s=dataset[1], fontsize="14")

    plt.figlegend(legend_list, labels=label_list, loc="best")
    # plt.legend()
    plt.show()



def running_time(dir, method_list, dataset_list):
    user_cmap = config_color_map()
    color_map = cm.get_cmap("tab20c")
    print()
    cnorm = matplotlib.colors.Normalize(vmin=0, vmax=20)
    sclarmap = cm.ScalarMappable(norm=cnorm, cmap=color_map)


    data = xlrd.open_workbook(dir)
    tabel = data.sheet_by_name("run_time")
    # *****************行*******************
    left_row_start = 3
    left_row_end = 14
    right_row_start = 3
    right_row_end = 9
    # ***********************列坐标*************
    left_col_start = 3
    left_col_end = 31
    right_col_start = 34
    right_col_end = 65
    data_kind = ["DBP_en_DBP_de", "DBP_en_DBP_fr", "DBP_en_WD_en", "DBP_en_YG_en"]
    left_data = []
    right_data = []
    plt.figure(figsize=(32, 6))
    sub_i = 1
    for dataset in dataset_list:
        remove = data_kind.index(dataset)
        left_col_start += 11 * remove
        left_col_end += 11 * remove
        right_col_start += 6 * remove
        right_col_end += 6 * remove
        for i in range(left_row_start, left_row_end):
            line_all_row = tabel.row_slice(i, left_col_start, left_col_end)
            left_data.append(line_all_row)
        for i in range(right_row_start, right_row_end):
            line_all_row = tabel.row_slice(i, right_col_start, right_col_end)
            right_data.append(line_all_row)
        print(left_data)
        print(right_data)
    all_data = left_data + right_data
    #     ****************************************开始画图相关工作*********************
    x_label_list = [["15K_V1", "15K_V2"], ["100K_V1", "100K_V2"]]

    color_list = ["darkorange", "forestgreen", "lightsteelblue", "rosybrown", "gold", "indigo", "red", "sienna",
                  "skyblue",
                  "deeppink", "slategray", "peru", "grey", "olive", "cyan", "blue", "lightpink"]
    color_list = [sclarmap.to_rgba(i) for i in range(17)]
    # ****************现在是17个方法******************
    first_x_position = [0.5, 12, ]
    width = 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax1.set_ylabel("Time(s)",fontsize=16)
    ax2.set_ylabel("Time(s)",fontsize=16)
    for method_pos in range(len(first_x_position)):
        for i in range(len(all_data)):
            bar1, = ax1.bar(first_x_position[method_pos] + i * (width),
                            height=all_data[i][5 + method_pos * 7].value, width=0.5,
                            bottom=0, color=color_list[i], edgecolor="black", label=method_list[i])
    ax1.text(x=3.5, y=-120,s=x_label_list[0][0],fontsize="16")
    ax1.text(x=15.5, y=-120, s=x_label_list[0][1],fontsize="16")

    legend_bar=list()
    for method_pos in range(len(first_x_position)):
        for i in range(len(all_data)):
            bar2, = ax2.bar(first_x_position[method_pos] + i * (width),
                            height=all_data[i][14 + 5 + method_pos * 7].value, width=0.5,
                            bottom=0, color=color_list[i], edgecolor="black",label=method_list[i])
            if i < len(first_x_position)/2:
                legend_bar.append(bar2)
    ax2.text(x=3.5, y=-2400, s=x_label_list[1][0], fontsize="16")
    ax2.text(x=15.5, y=-2400, s=x_label_list[1][1], fontsize="16")

    plt.figlegend(legend_bar, labels=method_list,loc="upper center",ncol=10,bbox_to_anchor=(0.5,0.95))
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    # dataset = ["BootEA", "GCN_Align", "IPTransE", "TransD", "TransH", "RotatE", "ConvE", "ProjE"]
    dataset = ["AttrE", "BootEA", "RotatE", "GCN_Align", "ProjE", "IPTransE", "ConvE", "TransH", "MTransE"]
    method_list = ['allents', "conicity", "nearents", "quartile"]
    method = "nearents"
    x_list = ["DBP_en_DBP_de_15K_V1", "DBP_en_DBP_de_15K_V2", ]
    stride = len(x_list)
    # **************************no font ******************************************************
    plot_config()
    # ****************************用来分析hub现象***************************************
    data = get_all_data(dataset, method="hub")
    compare_methond, gt10 = get_graphics_data(data, x_list, "gt5")
    _, to10 = get_graphics_data(data, x_list, "1to5")
    _, eq0 = get_graphics_data(data, x_list, "eq0")
    hub_picture([gt10, to10, eq0], x_list, compare_methond)
    # ******************************draw runningtime**********************************
    # rt_method_list = ["MTransE", 'IPTransE', "JAPE", "BootEA", "KDCoE", "GCN-Align", "AttrE", "IMUSE", "SEA", "RSN4EA"
    #     , "MultiKE", "TransH", "TransD", "ProjE", "ConvE", "SimplE", "RotatE"]
    # # rt_dataset_list = ["DBP_en_DBP_de", "DBP_en_DBP_fr", "DBP_en_WD_en", "DBP_en_YG_en"]
    # rt_dataset_list = ["DBP_en_DBP_de"]
    # rt_dir = "/home/cmwang/桌面/VLDB_exp.xlsx"
    # running_time(rt_dir, rt_method_list, rt_dataset_list)

    # **********************************grid figure***********************************
    # data = get_all_data(dataset, method)
    # compare_method, temp_data = get_graphics_data(data, x_list, "sim_result")
    # grid_figure(x_list, compare_method, temp_data)
    # ********************************************************************************

    # **************************************belong to one***************************
    # data = get_all_data(dataset, method="nearents")
    # compare_method, temp_data = get_graphics_data(data, x_list, "ent1_aver")
    # plot_figure(x_list, compare_method, temp_data,[0,10])
    # *******************************************************************************

    # **********************************aver_near_similarity 距离最近所有实体的平均相似度 ***************************
    # data = get_all_data(dataset, method="nearents")
    # compare_method, temp_data = get_graphics_data(data, x_list, "ent1_aver_near_aver")
    # plot_figure(x_list, compare_method, temp_data, [0,1])
    # *********************************************************************************

    # ***********************************quar_devi 后四分之一与前四分之一的差值*********************************************
    # data = get_all_data(dataset, method="quartile")
    # compare_method, temp_data = get_graphics_data(data, x_list, "quar_devi")
    # plot_figure(x_list, compare_method, temp_data, [0,8])
    # **********************************************************************************

    # ***********************************quar_devi  中间1/2的均值*********************************************
    # data = get_all_data(dataset, method="quartile")
    # compare_method, temp_data = get_graphics_data(data, x_list, "mean_value")
    # plot_figure(x_list, compare_method, temp_data, [3, 10])
    # ****************************************************************************************
    # **************************************不同kg里实体的关系************************************
    # data = get_all_data(dataset, method="allents")
    # compare_method, temp_data = get_graphics_data(data, x_list, "aver_most_sim")
    # plot_figure(x_list, compare_method, temp_data, [0,1])

    # **************************************************************************************
    # data = get_all_data(dataset, method="conicity")
    # compare_method, temp_data = get_graphics_data(data, x_list, "all_ents_conicity")
    # plot_figure(x_list, compare_method, temp_data, [0, 0.6])
    # compare_method, temp_data = get_graphics_data(data, x_list, "all_ents_vs")
    # plot_figure(x_list, compare_method, temp_data, [0, 0.5])
