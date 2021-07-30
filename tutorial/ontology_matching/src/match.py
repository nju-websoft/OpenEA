from demo import  matching_by_similarity_threshold

def ontology_matching(src_rdf, target_rdf):
    '''
    TODO Modify this function to generate all ontology/alignment pairs
    :return quadruple form of ontology matching: (entity1, entity2, measure, relation)
    '''
    return matching_by_similarity_threshold(src_rdf, target_rdf, 0.9)
