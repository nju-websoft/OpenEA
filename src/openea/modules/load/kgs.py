from openea.modules.load.kg import KG
from openea.modules.load.read import *


class KGs:
    def __init__(self, kg1: KG, kg2: KG, train_links, test_links, valid_links=None, mode='mapping', ordered=True):
        if mode == "sharing":
            ent_ids1, ent_ids2 = generate_sharing_id(train_links, kg1.relation_triples_set, kg1.entities_set,
                                                     kg2.relation_triples_set, kg2.entities_set, ordered=ordered)
            rel_ids1, rel_ids2 = generate_sharing_id([], kg1.relation_triples_set, kg1.relations_set,
                                                     kg2.relation_triples_set, kg2.relations_set, ordered=ordered)
            attr_ids1, attr_ids2 = generate_sharing_id([], kg1.attribute_triples_set, kg1.attributes_set,
                                                       kg2.attribute_triples_set, kg2.attributes_set, ordered=ordered)
        else:
            ent_ids1, ent_ids2 = generate_mapping_id(kg1.relation_triples_set, kg1.entities_set,
                                                     kg2.relation_triples_set, kg2.entities_set, ordered=ordered)
            rel_ids1, rel_ids2 = generate_mapping_id(kg1.relation_triples_set, kg1.relations_set,
                                                     kg2.relation_triples_set, kg2.relations_set, ordered=ordered)
            attr_ids1, attr_ids2 = generate_mapping_id(kg1.attribute_triples_set, kg1.attributes_set,
                                                       kg2.attribute_triples_set, kg2.attributes_set, ordered=ordered)

        id_relation_triples1 = uris_relation_triple_2ids(kg1.relation_triples_set, ent_ids1, rel_ids1)
        id_relation_triples2 = uris_relation_triple_2ids(kg2.relation_triples_set, ent_ids2, rel_ids2)

        id_attribute_triples1 = uris_attribute_triple_2ids(kg1.attribute_triples_set, ent_ids1, attr_ids1)
        id_attribute_triples2 = uris_attribute_triple_2ids(kg2.attribute_triples_set, ent_ids2, attr_ids2)

        self.uri_kg1 = kg1
        self.uri_kg2 = kg2

        kg1 = KG(id_relation_triples1, id_attribute_triples1)
        kg2 = KG(id_relation_triples2, id_attribute_triples2)
        kg1.set_id_dict(ent_ids1, rel_ids1, attr_ids1)
        kg2.set_id_dict(ent_ids2, rel_ids2, attr_ids2)

        self.uri_train_links = train_links
        self.uri_test_links = test_links
        self.train_links = uris_pair_2ids(self.uri_train_links, ent_ids1, ent_ids2)
        self.test_links = uris_pair_2ids(self.uri_test_links, ent_ids1, ent_ids2)
        self.train_entities1 = [link[0] for link in self.train_links]
        self.train_entities2 = [link[1] for link in self.train_links]
        self.test_entities1 = [link[0] for link in self.test_links]
        self.test_entities2 = [link[1] for link in self.test_links]

        if mode == 'swapping':
            sup_triples1, sup_triples2 = generate_sup_relation_triples(self.train_links,
                                                                       kg1.rt_dict, kg1.hr_dict,
                                                                       kg2.rt_dict, kg2.hr_dict)
            kg1.add_sup_relation_triples(sup_triples1)
            kg2.add_sup_relation_triples(sup_triples2)

            sup_triples1, sup_triples2 = generate_sup_attribute_triples(self.train_links, kg1.av_dict, kg2.av_dict)
            kg1.add_sup_attribute_triples(sup_triples1)
            kg2.add_sup_attribute_triples(sup_triples2)

        self.kg1 = kg1
        self.kg2 = kg2

        self.valid_links = list()
        self.valid_entities1 = list()
        self.valid_entities2 = list()
        if valid_links is not None:
            self.uri_valid_links = valid_links
            self.valid_links = uris_pair_2ids(self.uri_valid_links, ent_ids1, ent_ids2)
            self.valid_entities1 = [link[0] for link in self.valid_links]
            self.valid_entities2 = [link[1] for link in self.valid_links]

        # self.useful_entities_list1 = self.train_entities1 + self.valid_entities1 + self.test_entities1
        # self.useful_entities_list2 = self.train_entities2 + self.valid_entities2 + self.test_entities2

        self.useful_entities_list1 = self.kg1.entities_list
        self.useful_entities_list2 = self.kg2.entities_list

        self.entities_num = len(self.kg1.entities_set | self.kg2.entities_set)
        self.relations_num = len(self.kg1.relations_set | self.kg2.relations_set)
        self.attributes_num = len(self.kg1.attributes_set | self.kg2.attributes_set)


def read_kgs_from_folder(training_data_folder, division, mode, ordered, remove_unlinked=False):
    if 'dbp15k' in training_data_folder.lower() or 'dwy100k' in training_data_folder.lower():
        return read_kgs_from_dbp_dwy(training_data_folder, division, mode, ordered, remove_unlinked=remove_unlinked)
    kg1_relation_triples, _, _ = read_relation_triples(training_data_folder + 'rel_triples_1')
    kg2_relation_triples, _, _ = read_relation_triples(training_data_folder + 'rel_triples_2')
    kg1_attribute_triples, _, _ = read_attribute_triples(training_data_folder + 'attr_triples_1')
    kg2_attribute_triples, _, _ = read_attribute_triples(training_data_folder + 'attr_triples_2')

    train_links = read_links(training_data_folder + division + 'train_links')
    valid_links = read_links(training_data_folder + division + 'valid_links')
    test_links = read_links(training_data_folder + division + 'test_links')

    if remove_unlinked:
        links = train_links + valid_links + test_links
        kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
        kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)

    kg1 = KG(kg1_relation_triples, kg1_attribute_triples)
    kg2 = KG(kg2_relation_triples, kg2_attribute_triples)
    kgs = KGs(kg1, kg2, train_links, test_links, valid_links=valid_links, mode=mode, ordered=ordered)
    return kgs


def read_reversed_kgs_from_folder(training_data_folder, division, mode, ordered, remove_unlinked=False):
    kg1_relation_triples, _, _ = read_relation_triples(training_data_folder + 'rel_triples_2')
    kg2_relation_triples, _, _ = read_relation_triples(training_data_folder + 'rel_triples_1')
    kg1_attribute_triples, _, _ = read_attribute_triples(training_data_folder + 'attr_triples_2')
    kg2_attribute_triples, _, _ = read_attribute_triples(training_data_folder + 'attr_triples_1')

    temp_train_links = read_links(training_data_folder + division + 'train_links')
    temp_valid_links = read_links(training_data_folder + division + 'valid_links')
    temp_test_links = read_links(training_data_folder + division + 'test_links')
    train_links = [(j, i) for i, j in temp_train_links]
    valid_links = [(j, i) for i, j in temp_valid_links]
    test_links = [(j, i) for i, j in temp_test_links]

    if remove_unlinked:
        links = train_links + valid_links + test_links
        kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
        kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)

    kg1 = KG(kg1_relation_triples, kg1_attribute_triples)
    kg2 = KG(kg2_relation_triples, kg2_attribute_triples)
    kgs = KGs(kg1, kg2, train_links, test_links, valid_links=valid_links, mode=mode, ordered=ordered)
    return kgs


def read_kgs_from_files(kg1_relation_triples, kg2_relation_triples, kg1_attribute_triples, kg2_attribute_triples,
                        train_links, valid_links, test_links, mode):
    kg1 = KG(kg1_relation_triples, kg1_attribute_triples)
    kg2 = KG(kg2_relation_triples, kg2_attribute_triples)
    kgs = KGs(kg1, kg2, train_links, test_links, valid_links=valid_links, mode=mode)
    return kgs


def read_kgs_from_dbp_dwy(folder, division, mode, ordered, remove_unlinked=False):
    folder = folder + division
    kg1_relation_triples, _, _ = read_relation_triples(folder + 'triples_1')
    kg2_relation_triples, _, _ = read_relation_triples(folder + 'triples_2')
    if os.path.exists(folder + 'sup_pairs'):
        train_links = read_links(folder + 'sup_pairs')
    else:
        train_links = read_links(folder + 'sup_ent_ids')
    if os.path.exists(folder + 'ref_pairs'):
        test_links = read_links(folder + 'ref_pairs')
    else:
        test_links = read_links(folder + 'ref_ent_ids')
    print()
    if remove_unlinked:
        for i in range(10000):
            print("removing times:", i)
            links = train_links + test_links
            kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
            kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)
            n1 = len(kg1_relation_triples)
            n2 = len(kg2_relation_triples)
            train_links, test_links = remove_no_triples_link(kg1_relation_triples, kg2_relation_triples,
                                                             train_links, test_links)
            links = train_links + test_links
            kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
            kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)
            n11 = len(kg1_relation_triples)
            n22 = len(kg2_relation_triples)
            if n1 == n11 and n2 == n22:
                break
            print()

    kg1 = KG(kg1_relation_triples, list())
    kg2 = KG(kg2_relation_triples, list())
    kgs = KGs(kg1, kg2, train_links, test_links, mode=mode, ordered=ordered)
    return kgs


def remove_no_triples_link(kg1_relation_triples, kg2_relation_triples, train_links, test_links):
    kg1_entities, kg2_entities = set(), set()
    for h, r, t in kg1_relation_triples:
        kg1_entities.add(h)
        kg1_entities.add(t)
    for h, r, t in kg2_relation_triples:
        kg2_entities.add(h)
        kg2_entities.add(t)
    print("before removing links with no triples:", len(train_links), len(test_links))
    new_train_links, new_test_links = set(), set()
    for i, j in train_links:
        if i in kg1_entities and j in kg2_entities:
            new_train_links.add((i, j))
    for i, j in test_links:
        if i in kg1_entities and j in kg2_entities:
            new_test_links.add((i, j))
    print("after removing links with no triples:", len(new_train_links), len(new_test_links))
    return list(new_train_links), list(new_test_links)


def remove_unlinked_triples(triples, links):
    print("before removing unlinked triples:", len(triples))
    linked_entities = set()
    for i, j in links:
        linked_entities.add(i)
        linked_entities.add(j)
    linked_triples = set()
    for h, r, t in triples:
        if h in linked_entities and t in linked_entities:
            linked_triples.add((h, r, t))
    print("after removing unlinked triples:", len(linked_triples))
    return linked_triples