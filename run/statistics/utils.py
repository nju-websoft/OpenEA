from openea.modules.load.read import *


def read_alignment_results(file_path):
    aligned_ent_id_pair_set = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n').split('\t')
            aligned_ent_id_pair_set.add((int(line[0]), int(line[1])))
    return aligned_ent_id_pair_set


def read_item_id_file(file_path):
    item_id_dict, id_item_dict = {}, {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n').split('\t')
            item = line[0]
            item_id = int(line[1])
            item_id_dict[item] = item_id
            id_item_dict[item_id] = item
    return item_id_dict, id_item_dict


def id2ent_by_ent_links_index(ent_links):
    id_ent_dict_1, id_ent_dict_2 = {}, {}
    cnt_index = 0
    for (e1, e2) in ent_links:
        id_ent_dict_1[cnt_index] = e1
        id_ent_dict_2[cnt_index] = e2
        cnt_index += 1
    return id_ent_dict_1, id_ent_dict_2


