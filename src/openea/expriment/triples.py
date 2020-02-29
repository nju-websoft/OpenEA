import math


class Triples:
    def __init__(self, triples, ori_triples=None):
        self.triples = set(triples)
        self.triple_list = list(self.triples)
        self.triples_num = len(self.triples)

        self.heads = set([triple[0] for triple in self.triple_list])
        self.props = set([triple[1] for triple in self.triple_list])
        self.tails = set([triple[2] for triple in self.triple_list])
        self.ents = self.heads | self.tails

        print("triples num", self.triples_num)

        print("head ent num", len(self.heads))
        print("total ent num", len(self.ents))

        self.prop_list = list(self.props)
        self.ent_list = list(self.ents)
        self.prop_list.sort()
        self.ent_list.sort()

        # self.remove_useless_tripels()

        if ori_triples is None:
            self.ori_triples = None
        else:
            self.ori_triples = set(ori_triples)

        self._generate_related_ents()
        self._generate_triple_dict()
        self._generate_ht()
        self.__generate_weight()

    def _generate_related_ents(self):
        self.out_related_ents_dict = dict()
        self.in_related_ents_dict = dict()
        for h, r, t in self.triple_list:
            out_related_ents = self.out_related_ents_dict.get(h, set())
            out_related_ents.add(t)
            self.out_related_ents_dict[h] = out_related_ents

            in_related_ents = self.in_related_ents_dict.get(t, set())
            in_related_ents.add(h)
            self.in_related_ents_dict[t] = in_related_ents

    def _generate_triple_dict(self):
        self.rt_dict, self.hr_dict = dict(), dict()
        for h, r, t in self.triple_list:
            rt_set = self.rt_dict.get(h, set())
            rt_set.add((r, t))
            self.rt_dict[h] = rt_set
            hr_set = self.hr_dict.get(t, set())
            hr_set.add((h, r))
            self.hr_dict[t] = hr_set

    def _generate_ht(self):
        self.ht = set()
        for h, r, t in self.triples:
            self.ht.add((h, t))

    def __generate_weight(self):
        triple_num = dict()
        n = 0
        for h, r, t in self.triples:
            if t in self.heads:
                n = n + 1
                triple_num[h] = triple_num.get(h, 0) + 1
                triple_num[t] = triple_num.get(t, 0) + 1
        self.weighted_triples = list()
        self.additional_triples = list()
        ave = math.ceil(n / len(self.heads))
        print("ave outs:", ave)

        for h, r, t in self.triples:
            w = 1
            if t in self.heads and triple_num[h] <= ave:
                w = 2.0
                self.additional_triples.append((h, r, t))
            self.weighted_triples.append((h, r, t, w))
        print("additional triples:", len(self.additional_triples))
        # self.train_triples

    def remove_useless_tripels(self):
        filtered_triples = set()
        # dic = dict()
        # out_ents = self.tails - self.heads
        # for h, r, t in self.triples:
        #     if t in out_ents:
        #         dic[t] = dic.get(t, 0) + 1
        # truncated_ents = set()
        # for t, n in dic.items():
        #     if n < 3:
        #         truncated_ents.add(t)
        # print("truncated ents", len(truncated_ents))
        # for h, r, t in self.triples:
        #     if t not in truncated_ents:
        #         filtered_triples.add((h, r, t))
        # print("filtered triples", len(filtered_triples))

        for h, r, t in self.triples:
            if t in self.heads:
                filtered_triples.add((h, r, t))

        self.triples = set(filtered_triples)
        self.triple_list = list(self.triples)
        self.triples_num = len(self.triples)
