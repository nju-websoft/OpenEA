import multiprocessing

import numpy as np
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances

from openea.modules.utils.util import task_divide


def sim(embed1, embed2, metric='inner', normalize=False, csls_k=0):
    """
    Compute pairwise similarity between the two collections of embeddings.

    Parameters
    ----------
    embed1 : matrix_like
        An embedding matrix of size n1*d, where n1 is the number of embeddings and d is the dimension.
    embed2 : matrix_like
        An embedding matrix of size n2*d, where n2 is the number of embeddings and d is the dimension.
    metric : str, optional, inner default.
        The distance metric to use. It can be 'cosine', 'euclidean', 'inner'.
    normalize : bool, optional, default false.
        Whether to normalize the input embeddings.
    csls_k : int, optional, 0 by default.
        K value for csls. If k > 0, enhance the similarity by csls.

    Returns
    -------
    sim_mat : An similarity matrix of size n1*n2.
    """
    if normalize:
        embed1 = preprocessing.normalize(embed1)
        embed2 = preprocessing.normalize(embed2)
    if metric == 'inner':
        sim_mat = np.matmul(embed1, embed2.T)  # numpy.ndarray, float32
    elif metric == 'cosine' and normalize:
        sim_mat = np.matmul(embed1, embed2.T)  # numpy.ndarray, float32
    elif metric == 'euclidean':
        sim_mat = 1 - euclidean_distances(embed1, embed2)
        print(type(sim_mat), sim_mat.dtype)
        sim_mat = sim_mat.astype(np.float32)
    elif metric == 'cosine':
        sim_mat = 1 - cdist(embed1, embed2, metric='cosine')   # numpy.ndarray, float64
        sim_mat = sim_mat.astype(np.float32)
    elif metric == 'manhattan':
        sim_mat = 1 - cdist(embed1, embed2, metric='cityblock')
        sim_mat = sim_mat.astype(np.float32)
    else:
        sim_mat = 1 - cdist(embed1, embed2, metric=metric)
        sim_mat = sim_mat.astype(np.float32)
    if csls_k > 0:
        sim_mat = csls_sim(sim_mat, csls_k)
    return sim_mat


def csls_sim(sim_mat, k):
    """
    Compute pairwise csls similarity based on the input similarity matrix.

    Parameters
    ----------
    sim_mat : matrix-like
        A pairwise similarity matrix.
    k : int
        The number of nearest neighbors.

    Returns
    -------
    csls_sim_mat : A csls similarity matrix of n1*n2.
    """

    nearest_values1 = calculate_nearest_k(sim_mat, k)
    nearest_values2 = calculate_nearest_k(sim_mat.T, k)
    csls_sim_mat = 2 * sim_mat.T - nearest_values1
    csls_sim_mat = csls_sim_mat.T - nearest_values2
    return csls_sim_mat


def calculate_nearest_k(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    return np.mean(nearest_k, axis=1)


def csls_sim_multi_threads(sim_mat, k, nums_threads):
    tasks = task_divide(np.array(range(sim_mat.shape[0])), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    rests = list()
    for task in tasks:
        rests.append(pool.apply_async(calculate_nearest_k, (sim_mat[task, :], k)))
    pool.close()
    pool.join()
    sim_values = None
    for res in rests:
        val = res.get()
        if sim_values is None:
            sim_values = val
        else:
            sim_values = np.append(sim_values, val)
    assert sim_values.shape[0] == sim_mat.shape[0]
    return sim_values


def sim_multi_threads(embeds1, embeds2, threads_num=16):
    num = embeds1.shape[0]
    idx_list = task_divide(np.array(range(num)), threads_num)
    pool = multiprocessing.Pool(processes=len(idx_list))
    rests = list()
    for idx in idx_list:
        rests.append(pool.apply_async(np.dot, (embeds1[idx, :], embeds2.T)))
    sim_list = []
    for res in rests:
        sim_list.append(res.get())
    sim_mat = np.concatenate(sim_list, axis=0)
    return sim_mat


def sim_multi_blocks(embeds1, embeds2, blocks_num=16):
    num = embeds1.shape[0]
    idx_list = task_divide(np.array(range(num)), blocks_num)
    sim_list = []
    for idx in idx_list:
        res = np.matmul(embeds1[idx, :], embeds2.T)
        sim_list.append(res)
    sim_mat = np.concatenate(sim_list, axis=0)
    return sim_mat


if __name__ == '__main__':
    dim = 1000
    n = 100000
    a = np.random.randn(n, dim).astype(np.float32)
    b = np.random.randn(n, dim).astype(np.float32)
    sim = sim_multi_blocks(a, b, blocks_num=16)
    print(sim.shape)
