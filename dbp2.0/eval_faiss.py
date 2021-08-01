import time

import faiss
import numpy as np
from sklearn import preprocessing


def task_divide(idx, n):
    total = len(idx)
    if n <= 0 or 0 == total:
        return [idx]
    if n > total:
        return [idx]
    elif n == total:
        return [[i] for i in idx]
    else:
        j = total // n
        tasks = []
        for i in range(0, (n - 1) * j, j):
            tasks.append(idx[i:i + j])
        tasks.append(idx[(n - 1) * j:])
        return tasks


def test_by_faiss_batch(embeds1, embeds2, top_k, is_norm=True, batch_num=10):
    start = time.time()
    if is_norm:
        embeds1 = preprocessing.normalize(embeds1)
        embeds2 = preprocessing.normalize(embeds2)
    num = embeds2.shape[0]
    dim = embeds2.shape[1]
    hits = [0] * len(top_k)
    mr, mrr = 0, 0
    index = faiss.IndexFlatL2(dim)  # build the index
    index.add(embeds1)  # add vectors to the index
    batches = task_divide(list(range(num)), batch_num)
    t = time.time()
    query_num = 0
    for bj, batch in enumerate(batches):
        query_num += len(batch)
        query = embeds2[batch, :]  # (15743, 300)
        _, index_mat = index.search(query, num)
        for i, ent_i in enumerate(batch):
            golden = ent_i
            vec = index_mat[i,]  # (157438,)
            golden_index = np.where(vec == golden)[0]
            if len(golden_index) > 0:
                rank = golden_index[0]
                mr += (rank + 1)
                mrr += 1 / (rank + 1)
                for j in range(len(top_k)):
                    if rank < top_k[j]:
                        hits[j] += 1
        print("alignment evaluating at batch {}, hits@{} = {} time = {:.3f} s ".
              format(bj, top_k, np.array(hits) / query_num, time.time() - t))
        t = time.time()
    mr /= num
    mrr /= num
    hits = np.array(hits) / num
    mr = round(mr, 8)
    mrr = round(mrr, 8)
    for i in range(len(hits)):
        hits[i] = round(hits[i], 8)
    print("alignment results with faiss: hits@{} = {}, mr = {:.3f}, mrr = {:.6f}, total time = {:.3f} s ".
          format(top_k, hits, mr, mrr, time.time() - start))
    return hits, mrr, mr


def test_by_faiss(embeds1, embeds2, top_k, align1, entities1, align2, entities2):
    start = time.time()
    embeds1 = preprocessing.normalize(embeds1)
    embeds2 = preprocessing.normalize(embeds2)
    num = embeds2.shape[0]
    dim = embeds2.shape[1]
    # dist_mat, index_mat = knn_by_faiss(embeds1, embeds2, embeds1.shape[1], num)
    # print(index_mat)
    hits = [0] * len(top_k)
    mr, mrr = 0, 0
    index = faiss.IndexFlatL2(dim)  # build the index
    index.add(embeds1)  # add vectors to the index
    t = time.time()
    same_num = 0
    for i in range(num):
        # print(entities1[align1[i]], " ", entities2[align2[i]])
        if entities1[align1[i]].split('/')[-1] == entities2[align2[i]].split('/')[-1]:
            # print(embeds1[i, 0], "=?", embeds2[i, 0], embeds1[i, 299], "=?", embeds2[i, 299])
            same_num += 1
        query = embeds2[i, :].reshape(1, dim)
        _, index_vec = index.search(query, num)
        golden = i
        vec = index_vec[0,]
        golden_index = np.where(vec == golden)[0]
        if len(golden_index) > 0:
            rank = golden_index[0]
            mr += (rank + 1)
            mrr += 1 / (rank + 1)
            for j in range(len(top_k)):
                if rank < top_k[j]:
                    hits[j] += 1
        if i % 1000 == 0:
            print("alignment evaluating at {}, hits@{} = {}, same= {}, time = {:.3f} s ".
                  format(i, top_k, np.array(hits) / (i + 1), same_num / (i + 1), time.time() - t))
            t = time.time()
    mr /= num
    mrr /= num
    hits = np.array(hits) / num
    mr = round(mr, 8)
    mrr = round(mrr, 8)
    for i in range(len(hits)):
        hits[i] = round(hits[i], 8)
    print("alignment results with faiss: hits@{} = {}, mr = {:.3f}, mrr = {:.6f}, total time = {:.3f} s ".
          format(top_k, hits, mr, mrr, time.time() - start))
    return hits, mrr, mr
