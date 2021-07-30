import numpy as np
import Levenshtein as lev


def edit_distance(str1, str2):
    '''
    Edit distance between two string
    '''
    min_len = min(len(str1), len(str2))
    if min_len == 0:
        return 0
    dist = lev.distance(str1, str2)
    return round(1 - dist / min_len, 4)


def label_sim_matrix(src_labels, target_labels):
    '''
    :param src_labels: labels of source ontology
    :param target_labels: labels of target ontology
    :return: the matrix of similarity
    '''
    s_len = len(src_labels)
    t_len = len(target_labels)
    mat = np.zeros([s_len, t_len])
    for i in range(s_len):
        for j in range(t_len):
            mat[i][j] = edit_distance(src_labels[i].lower(), target_labels[j].lower())
    return mat


def matching_by_similarity_threshold(src_rdf, target_rdf, threshold):
    print('Similarity threshold: {}'.format(threshold))
    # (entity1, entity2, measure, relation): entity1 is from 101
    matching_pairs = []

    # classes
    sim_mat = label_sim_matrix(src_rdf.class_labels, target_rdf.class_labels)
    for i in range(sim_mat.shape[0]):
        for j in range(sim_mat.shape[1]):
            if sim_mat[i][j] >= threshold:
                matching_pairs.append((src_rdf.class_uris[i], target_rdf.class_uris[j], 1.0, '='))

    # attributes
    sim_mat = label_sim_matrix(src_rdf.attribute_labels, target_rdf.attribute_labels)
    for i in range(sim_mat.shape[0]):
        for j in range(sim_mat.shape[1]):
            if sim_mat[i][j] >= threshold:
                matching_pairs.append((src_rdf.attribute_uris[i], target_rdf.attribute_uris[j], 1.0, '='))

    # relationships
    sim_mat = label_sim_matrix(src_rdf.relation_labels, target_rdf.relation_labels)
    for i in range(sim_mat.shape[0]):
        for j in range(sim_mat.shape[1]):
            if sim_mat[i][j] >= threshold:
                matching_pairs.append((src_rdf.relation_uris[i], target_rdf.relation_uris[j], 1.0, '='))

    return matching_pairs
