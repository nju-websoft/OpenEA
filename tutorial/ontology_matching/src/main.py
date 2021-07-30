from data_input import RdfParser
from match import ontology_matching
import json
import os


def main():
    src_file = '../datasets/101/onto.rdf'
    target_dict = {'301':'http://oaei.ontologymatching.org/2007/benchmarks/301/onto.rdf#',
                   '302':'http://ebiquity.umbc.edu/v2.1/ontology/publication.owl#',
                   '303':'http://www.aifb.uni-karlsruhe.de/ontology#',
                   '304':'http://oaei.ontologymatching.org/2007/benchmarks/304/onto.rdf#'}
    out_dir = '../output'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    src_rdf = RdfParser(src_file, 'http://oaei.ontologymatching.org/2007/benchmarks/101/onto.rdf#')
    for sub_dir, namespace in target_dict.items():
        target_file = os.path.join(os.path.join('../datasets/', sub_dir), 'onto.rdf')
        # parse rdf
        target_rdf = RdfParser(target_file, namespace)

        # ontology matching
        matching_pairs = ontology_matching(src_rdf, target_rdf)

        # output
        outfile = os.path.join(out_dir, '101-{}.json'.format(sub_dir))
        with open(outfile, 'w') as f:
            json.dump(matching_pairs, f)


if __name__ == '__main__':
    main()
