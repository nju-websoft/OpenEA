import rdflib


class RdfParser:
    '''
        Parse onto.rdf
    '''
    def __init__(self, rdf_file, namespace):
        print('----------- {} -----------'.format(rdf_file))
        # uris and labels of ontology
        self.uris = []
        self.labels = []

        # RDF Graph
        self._graph = rdflib.Graph()
        self._graph.parse(rdf_file)
        self.class_uris = [subject for subject in self._graph.subjects(rdflib.RDF.type, rdflib.OWL.Class) if namespace in str(subject)]
        print('The number of classes is {}.'.format(len(self.class_uris)))
        self.attribute_uris = [subject for subject in self._graph.subjects(rdflib.RDF.type, rdflib.OWL.DatatypeProperty) if namespace in str(subject)]
        print('The number of attributes is {}.'.format(len(self.attribute_uris)))
        self.relation_uris = [subject for subject in self._graph.subjects(rdflib.RDF.type, rdflib.OWL.ObjectProperty) if namespace in str(subject)]
        print('The number of relationships is {}.'.format(len(self.relation_uris)))

        # labels
        self.class_labels = [self._graph.label(uri) for uri in self.class_uris]
        self.attribute_labels = [self._graph.label(uri) for uri in self.attribute_uris]
        self.relation_labels = [self._graph.label(uri) for uri in self.relation_uris]

