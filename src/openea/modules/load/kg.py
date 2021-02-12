def parse_triples(triples):
    subjects, predicates, objects = set(), set(), set()
    for s, p, o in triples:
        subjects.add(s)
        predicates.add(p)
        objects.add(o)
    return subjects, predicates, objects


class KG:
    def __init__(self, relation_triples, attribute_triples):

        self.entities_set, self.entities_list = None, None
        self.relations_set, self.relations_list = None, None
        self.attributes_set, self.attributes_list = None, None
        self.entities_num, self.relations_num, self.attributes_num = None, None, None
        self.relation_triples_num, self.attribute_triples_num = None, None
        self.local_relation_triples_num, self.local_attribute_triples_num = None, None

        self.entities_id_dict = None
        self.relations_id_dict = None
        self.attributes_id_dict = None

        self.rt_dict, self.hr_dict = None, None
        self.entity_relations_dict = None
        self.entity_attributes_dict = None
        self.av_dict = None

        self.sup_relation_triples_set, self.sup_relation_triples_list = None, None
        self.sup_attribute_triples_set, self.sup_attribute_triples_list = None, None

        self.relation_triples_set = None
        self.attribute_triples_set = None
        self.relation_triples_list = None
        self.attribute_triples_list = None

        self.local_relation_triples_set = None
        self.local_relation_triples_list = None
        self.local_attribute_triples_set = None
        self.local_attribute_triples_list = None

        self.set_relations(relation_triples)
        self.set_attributes(attribute_triples)

        print()
        print("KG statistics:")
        print("Number of entities:", self.entities_num)
        print("Number of relations:", self.relations_num)
        print("Number of attributes:", self.attributes_num)
        print("Number of relation triples:", self.relation_triples_num)
        print("Number of attribute triples:", self.attribute_triples_num)
        print("Number of local relation triples:", self.local_relation_triples_num)
        print("Number of local attribute triples:", self.local_attribute_triples_num)
        print()

    def set_relations(self, relation_triples):
        self.relation_triples_set = set(relation_triples)
        self.relation_triples_list = list(self.relation_triples_set)
        self.local_relation_triples_set = self.relation_triples_set
        self.local_relation_triples_list = self.relation_triples_list

        heads, relations, tails = parse_triples(self.relation_triples_set)
        self.entities_set = heads | tails
        self.relations_set = relations
        self.entities_list = list(self.entities_set)
        self.relations_list = list(self.relations_set)
        self.entities_num = len(self.entities_set)
        self.relations_num = len(self.relations_set)
        self.relation_triples_num = len(self.relation_triples_set)
        self.local_relation_triples_num = len(self.local_relation_triples_set)
        self.generate_relation_triple_dict()
        self.parse_relations()

    def set_attributes(self, attribute_triples):
        self.attribute_triples_set = set(attribute_triples)
        self.attribute_triples_list = list(self.attribute_triples_set)
        self.local_attribute_triples_set = self.attribute_triples_set
        self.local_attribute_triples_list = self.attribute_triples_list

        entities, attributes, values = parse_triples(self.attribute_triples_set)
        self.attributes_set = attributes
        self.attributes_list = list(self.attributes_set)
        self.attributes_num = len(self.attributes_set)

        # add the new entities from attribute triples
        self.entities_set |= entities
        self.entities_list = list(self.entities_set)
        self.entities_num = len(self.entities_set)

        self.attribute_triples_num = len(self.attribute_triples_set)
        self.local_attribute_triples_num = len(self.local_attribute_triples_set)
        self.generate_attribute_triple_dict()
        self.parse_attributes()

    def generate_relation_triple_dict(self):
        self.rt_dict, self.hr_dict = dict(), dict()
        for h, r, t in self.local_relation_triples_list:
            rt_set = self.rt_dict.get(h, set())
            rt_set.add((r, t))
            self.rt_dict[h] = rt_set
            hr_set = self.hr_dict.get(t, set())
            hr_set.add((h, r))
            self.hr_dict[t] = hr_set
        print("Number of rt_dict:", len(self.rt_dict))
        print("Number of hr_dict:", len(self.hr_dict))

    def generate_attribute_triple_dict(self):
        self.av_dict = dict()
        for h, a, v in self.local_attribute_triples_list:
            av_set = self.av_dict.get(h, set())
            av_set.add((a, v))
            self.av_dict[h] = av_set
        print("Number of av_dict:", len(self.av_dict))

    def parse_relations(self):
        self.entity_relations_dict = dict()
        for ent, attr, _ in self.local_relation_triples_set:
            attrs = self.entity_relations_dict.get(ent, set())
            attrs.add(attr)
            self.entity_relations_dict[ent] = attrs
        print("entity relations dict:", len(self.entity_relations_dict))

    def parse_attributes(self):
        self.entity_attributes_dict = dict()
        for ent, attr, _ in self.local_attribute_triples_set:
            attrs = self.entity_attributes_dict.get(ent, set())
            attrs.add(attr)
            self.entity_attributes_dict[ent] = attrs
        print("entity attributes dict:", len(self.entity_attributes_dict))

    def set_id_dict(self, entities_id_dict, relations_id_dict, attributes_id_dict):
        self.entities_id_dict = entities_id_dict
        self.relations_id_dict = relations_id_dict
        self.attributes_id_dict = attributes_id_dict

    def add_sup_relation_triples(self, sup_triples):
        self.sup_relation_triples_set = set(sup_triples)
        self.sup_relation_triples_list = list(self.sup_relation_triples_set)
        self.relation_triples_set |= sup_triples
        self.relation_triples_list = list(self.relation_triples_set)
        self.relation_triples_num = len(self.relation_triples_list)

    def add_sup_attribute_triples(self, sup_triples):
        self.sup_attribute_triples_set = set(sup_triples)
        self.sup_attribute_triples_list = list(self.sup_attribute_triples_set)
        self.attribute_triples_set |= sup_triples
        self.attribute_triples_list = list(self.attribute_triples_set)
        self.attribute_triples_num = len(self.attribute_triples_list)