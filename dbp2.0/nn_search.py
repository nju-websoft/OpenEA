import gc
import ray
import numpy as np

from openea.modules.utils.util import task_divide, merge_dic

ray.init()


def generate_neighbours(entity_embeds, entity_list, neighbors_num, frags_num=8):
    ent_frags = task_divide(np.array(entity_list), frags_num)
    ent_frag_indexes = task_divide(np.array(range(len(entity_list))), frags_num)
    dic = dict()
    rest = []
    for i in range(len(ent_frags)):
        res = find_neighbours.remote(ent_frags[i], np.array(entity_list), entity_embeds[ent_frag_indexes[i], :],
                                     entity_embeds, neighbors_num)
        rest.append(res)
    for res in ray.get(rest):
        dic = merge_dic(dic, res)
    gc.collect()
    return dic


@ray.remote(num_cpus=1)
def find_neighbours(frags, entity_list, sub_embed, embed, k):
    dic = dict()
    sim_mat = np.matmul(sub_embed, embed.T)
    for i in range(sim_mat.shape[0]):
        sort_index = np.argpartition(-sim_mat[i, :], k)
        neighbors_index = sort_index[0:k]
        neighbors = entity_list[neighbors_index].tolist()
        dic[frags[i]] = neighbors
    del sim_mat
    return dic
