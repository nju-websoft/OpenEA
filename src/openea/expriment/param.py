class Params:
    def __init__(self):
        self.embed_size = 75
        self.epochs = 2000
        self.learning_rate = 0.01
        self.top_k = 20
        self.ent_top_k = [1, 5, 10, 50]
        self.lambda_3 = 0.7
        self.generate_sim = 10
        self.csls = 5
        self.heuristic = True
        self.is_save = True

        self.mu1 = 1.0
        self.mu2 = 1.0

        self.nums_threads = 10
        self.nums_threads_batch = 1

        self.margin_rel = 0.5
        self.margin_neg_triple = 1.2
        self.margin_ent = 0.5
        self.nums_neg = 10
        self.nums_neg_neighbor = 10

        self.epsilon = 0.95
        self.batch_size = 10000
        self.is_disc = True

        self.nn = 5

    def print(self):
        print("Parameters used in this running are as follows:")
        items = sorted(self.__dict__.items(), key=lambda d: d[0])
        for item in items:
            print("%s: %s" % item)
        print()


P = Params()
P.print()

if __name__ == '__main__':
    print("w" + str(1))
