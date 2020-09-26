import multiprocessing
import random
import gc

import numpy as np

from openea.modules.load.kg import KG
from openea.modules.utils.util import task_divide, merge_dic


def generate_pos_batch_queue(triple_list1, triple_list2, batch_size, steps, out_queue):
    for step in steps:
        out_queue.put(generate_pos_batch(triple_list1, triple_list2, batch_size, step))
    exit(0)


def generate_pos_batch(triple_list1, triple_list2, batch_size, step):
    batch_size1 = int(len(triple_list1) / (len(triple_list1) + len(triple_list2)) * batch_size)
    batch_size2 = batch_size - batch_size1
    pos_batch1 = generate_pos_triples(triple_list1, batch_size1, step)
    pos_batch2 = generate_pos_triples(triple_list2, batch_size2, step)
    return pos_batch1 + pos_batch2


def generate_relation_triple_batch_queue(triple_list1, triple_list2, triple_set1, triple_set2,
                                         entity_list1, entity_list2, batch_size,
                                         steps, out_queue, neighbor1, neighbor2, neg_triples_num):
    for step in steps:
        pos_batch, neg_batch = generate_relation_triple_batch(triple_list1, triple_list2, triple_set1, triple_set2,
                                                              entity_list1, entity_list2, batch_size,
                                                              step, neighbor1, neighbor2, neg_triples_num)
        out_queue.put((pos_batch, neg_batch))
    exit(0)


def generate_relation_triple_batch(triple_list1, triple_list2, triple_set1, triple_set2,
                                   entity_list1, entity_list2, batch_size,
                                   step, neighbor1, neighbor2, neg_triples_num):
    batch_size1 = int(len(triple_list1) / (len(triple_list1) + len(triple_list2)) * batch_size)
    batch_size2 = batch_size - batch_size1
    pos_batch1 = generate_pos_triples(triple_list1, batch_size1, step)
    pos_batch2 = generate_pos_triples(triple_list2, batch_size2, step)
    neg_batch1 = generate_neg_triples_fast(pos_batch1, triple_set1, entity_list1, neg_triples_num, neighbor=neighbor1)
    neg_batch2 = generate_neg_triples_fast(pos_batch2, triple_set2, entity_list2, neg_triples_num, neighbor=neighbor2)
    return pos_batch1 + pos_batch2, neg_batch1 + neg_batch2


def generate_pos_triples(triples, batch_size, step, is_fixed_size=False):
    start = step * batch_size
    end = start + batch_size
    if end > len(triples):
        end = len(triples)
    pos_batch = triples[start: end]
    # pos_batch = random.sample(triples, batch_size)
    if is_fixed_size and len(pos_batch) < batch_size:
        pos_batch += triples[:batch_size - len(pos_batch)]
    return pos_batch


def generate_neg_triples(pos_batch, all_triples_set, entities_list, neg_triples_num, neighbor=None, max_try=10):
    if neighbor is None:
        neighbor = dict()
    neg_batch = list()
    for head, relation, tail in pos_batch:
        head_candidates = neighbor.get(head, entities_list)
        tail_candidates = neighbor.get(tail, entities_list)
        for i in range(neg_triples_num):
            n = 0
            while True:
                corrupt_head_prob = np.random.binomial(1, 0.5)
                neg_head = head
                neg_tail = tail
                if corrupt_head_prob:
                    neg_head = random.choice(head_candidates)
                else:
                    neg_tail = random.choice(tail_candidates)
                if (neg_head, relation, neg_tail) not in all_triples_set:
                    neg_batch.append((neg_head, relation, neg_tail))
                    break
                n += 1
                if n == max_try:
                    neg_tail = random.choice(entities_list)
                    neg_batch.append((head, relation, neg_tail))
                    break
    assert len(neg_batch) == neg_triples_num * len(pos_batch)
    return neg_batch


def generate_neg_triples_fast(pos_batch, all_triples_set, entities_list, neg_triples_num, neighbor=None, max_try=10):
    if neighbor is None:
        neighbor = dict()
    neg_batch = list()
    for head, relation, tail in pos_batch:
        neg_triples = list()
        nums_to_sample = neg_triples_num
        head_candidates = neighbor.get(head, entities_list)
        tail_candidates = neighbor.get(tail, entities_list)
        for i in range(max_try):
            corrupt_head_prob = np.random.binomial(1, 0.5)
            if corrupt_head_prob:
                neg_heads = random.sample(head_candidates, nums_to_sample)
                i_neg_triples = {(h2, relation, tail) for h2 in neg_heads}
            else:
                neg_tails = random.sample(tail_candidates, nums_to_sample)
                i_neg_triples = {(head, relation, t2) for t2 in neg_tails}
            if i == max_try - 1:
                neg_triples += list(i_neg_triples)
                break
            else:
                i_neg_triples = list(i_neg_triples - all_triples_set)
                neg_triples += i_neg_triples
            if len(neg_triples) == neg_triples_num:
                break
            else:
                nums_to_sample = neg_triples_num - len(neg_triples)
        assert len(neg_triples) == neg_triples_num
        neg_batch.extend(neg_triples)
    assert len(neg_batch) == neg_triples_num * len(pos_batch)
    return neg_batch


def generate_neighbours(entity_embeds, entity_list, neighbors_num, threads_num):
    ent_frags = task_divide(np.array(entity_list), threads_num)
    ent_frag_indexes = task_divide(np.array(range(len(entity_list))), threads_num)

    pool = multiprocessing.Pool(processes=len(ent_frags))
    results = list()
    for i in range(len(ent_frags)):
        results.append(pool.apply_async(find_neighbours,
                                        args=(ent_frags[i], np.array(entity_list),
                                              entity_embeds[ent_frag_indexes[i], :],
                                              entity_embeds, neighbors_num)))
    pool.close()
    pool.join()

    dic = dict()
    for res in results:
        dic = merge_dic(dic, res.get())

    del results
    gc.collect()
    return dic


def generate_neighbours_single_thread(entity_embeds, entity_list, neighbors_num, threads_num):
    ent_frags = task_divide(np.array(entity_list), threads_num)
    ent_frag_indexes = task_divide(np.array(range(len(entity_list))), threads_num)
    results = dict()
    for i in range(len(ent_frags)):
        dic = find_neighbours(ent_frags[i], np.array(entity_list),
                              entity_embeds[ent_frag_indexes[i], :],
                              entity_embeds, neighbors_num)
        results = merge_dic(results, dic)
    return results


def find_neighbours(frags, entity_list, sub_embed, embed, k):
    dic = dict()
    sim_mat = np.matmul(sub_embed, embed.T)
    for i in range(sim_mat.shape[0]):
        sort_index = np.argpartition(-sim_mat[i, :], k)
        neighbors_index = sort_index[0:k]
        neighbors = entity_list[neighbors_index].tolist()
        dic[frags[i]] = neighbors
    return dic


def generate_triple_label_batch(triple_list1, triple_list2, triple_set1, triple_set2, entity_list1, entity_list2,
                                batch_size, steps, out_queue, neighbor1, neighbor2, neg_triples_num):
    batch_size1 = int(len(triple_list1) / (len(triple_list1) + len(triple_list2)) * batch_size)
    batch_size2 = batch_size - batch_size1
    for step in steps:
        pos_batch1 = generate_pos_triples(triple_list1, batch_size1, step)
        pos_batch2 = generate_pos_triples(triple_list2, batch_size2, step)
        neg_batch1 = generate_neg_triples(pos_batch1, triple_set1, entity_list1,
                                          neg_triples_num, neighbor=neighbor1)
        neg_batch2 = generate_neg_triples(pos_batch2, triple_set2, entity_list2,
                                          neg_triples_num, neighbor=neighbor2)
        pos_batch = pos_batch1 + pos_batch2
        pos_label = [1] * len(pos_batch)
        neg_batch = neg_batch1 + neg_batch2
        neg_label = [-1] * len(neg_batch)
        out_queue.put((pos_batch + neg_batch, pos_label + neg_label))
    exit(0)


def generate_neg_attribute_triples(pos_batch, all_triples_set, entity_list, neg_triples_num, neighbor=None):
    if neighbor is None:
        neighbor = dict()
    neg_batch = list()
    for head, attribute, value in pos_batch:
        for i in range(neg_triples_num):
            while True:
                neg_head = random.choice(neighbor.get(head, entity_list))
                if (neg_head, attribute, value) not in all_triples_set:
                    break
            neg_batch.append((neg_head, attribute, value))
    assert len(neg_batch) == neg_triples_num * len(pos_batch)
    return neg_batch


def generate_attribute_triple_batch_queue(triple_list1, triple_list2, triple_set1, triple_set2,
                                          entity_list1, entity_list2, batch_size,
                                          steps, out_queue, neighbor1, neighbor2, neg_triples_num, is_fixed_size):
    for step in steps:
        pos_batch, neg_batch = generate_attribute_triple_batch(triple_list1, triple_list2, triple_set1, triple_set2,
                                                               entity_list1, entity_list2, batch_size,
                                                               step, neighbor1, neighbor2, neg_triples_num,
                                                               is_fixed_size)
        out_queue.put((pos_batch, neg_batch))
    exit(0)


def generate_attribute_triple_batch(triple_list1, triple_list2, triple_set1, triple_set2,
                                    entity_list1, entity_list2, batch_size,
                                    step, neighbor1, neighbor2, neg_triples_num, is_fixed_size):
    batch_size1 = int(len(triple_list1) / (len(triple_list1) + len(triple_list2)) * batch_size)
    batch_size2 = batch_size - batch_size1
    pos_batch1 = generate_pos_triples(triple_list1, batch_size1, step, is_fixed_size=is_fixed_size)
    pos_batch2 = generate_pos_triples(triple_list2, batch_size2, step, is_fixed_size=is_fixed_size)
    neg_batch1 = generate_neg_attribute_triples(pos_batch1, triple_set1, entity_list1,
                                                neg_triples_num, neighbor=neighbor1)
    neg_batch2 = generate_neg_attribute_triples(pos_batch2, triple_set2, entity_list2,
                                                neg_triples_num, neighbor=neighbor2)
    return pos_batch1 + pos_batch2, neg_batch1 + neg_batch2
