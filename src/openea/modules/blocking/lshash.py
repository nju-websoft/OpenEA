import numpy as np

try:
    from bitarray import bitarray
except ImportError:
    bitarray = None


class HashTable:
    def __init__(self, idx, table=None):
        self.table_id = idx  # id of the hash table
        if table is None:
            self.table = dict()  # key: hash code; value: a tuple (vector id, vector)
        else:
            self.table = table

    def keys(self):
        return self.table.keys()

    def get_bucket_by_code(self, code):
        return self.table.get(code, set())

    def add_vec(self, code, vec, idx):
        self.table.setdefault(code, set()).add((idx, vec))


class LSHash(object):

    def __init__(self, code_dim, vec_dim, num_tables=1):

        self.hash_size = code_dim
        self.input_dim = vec_dim
        self.num_tables = num_tables

        self._init_uniform_planes()
        self._init_tables()

    def _init_uniform_planes(self):
        self.uniform_planes = [self._generate_uniform_planes() for _ in range(self.num_tables)]

    def _init_tables(self):
        self.hash_tables = [HashTable(i) for i in range(self.num_tables)]

    def _generate_uniform_planes(self):
        """ Generate uniformly distributed hyperplanes and return it as a 2D numpy array. """
        return np.random.randn(self.hash_size, self.input_dim)

    def _hash(self, planes, vec):
        """
        Generates the binary hash for `vec` and returns it.
        :param planes:  The planes are random uniform planes with a dimension of `code_dim` * `vec_dim`.
        :param vec: A Python tuple or list object that contains only numbers. The dimension needs to be 1 * `vec_dim`.
        :return:
        """
        vec = np.array(vec)  # for faster dot product
        projections = np.dot(planes, vec)
        return "".join(['1' if i > 0 else '0' for i in projections])

    def _as_np_array(self, input):
        if isinstance(input, (tuple, list)):
            return np.asarray(input)
        return input

    def index(self, vec, idx):
        """
        Index a single input vector by adding it to the hash tables.
        :param vec:
        :param idx:
        :return:
        """
        if isinstance(vec, np.ndarray):
            vec = vec.tolist()
        vec = tuple(vec)
        for i, table in enumerate(self.hash_tables):
            table.add_vec(self._hash(self.uniform_planes[i], vec), vec, idx)

    def query(self, query_vec, query_id=None, dis_mat=None, num_results=1, distance_func="euclidean"):
        """
        Takes `query_vec` which is either a tuple or a list of numbers or numpy array,
        returns `num_results` of results as a list of tuples that are ranked
        based on the supplied metric function `distance_func`.
        :param query_vec:
        :param num_results:
        :param distance_func:
        :return:
        """
        candidates = set()
        if distance_func == "euclidean":
            d_func = LSHash.euclidean_dist_square
        elif distance_func == "true_euclidean":
            d_func = LSHash.euclidean_dist
        elif distance_func == "centred_euclidean":
            d_func = LSHash.euclidean_dist_centred
        elif distance_func == "cosine":
            d_func = LSHash.cosine_dist
        elif distance_func == "l1norm":
            d_func = LSHash.l1norm_dist
        else:
            raise ValueError("The distance function name is invalid.")

        for i, table in enumerate(self.hash_tables):
            binary_hash = self._hash(self.uniform_planes[i], query_vec)
            candidates.update(table.get_bucket_by_code(binary_hash))

        # rank candidates by distance function
        candidates = list(candidates)
        if query_id is not None and dis_mat is not None:
            candidates_dis = [dis_mat[ix[0], query_id] for ix in candidates]
        else:
            candidates_dis = [d_func(query_vec, self._as_np_array(ix[1])) for ix in candidates]
        # print(len(candidates))
        candidates_dis = np.array(candidates_dis)
        if num_results == 1:
            sorted_index = np.argmin(candidates_dis)
            candidates = [candidates[sorted_index]]
        else:
            sorted_index = candidates_dis.argsort()[0:num_results]
            candidates = [candidates[i] for i in sorted_index]

        return candidates

    # distance functions

    @staticmethod
    def hamming_dist(bitarray1, bitarray2):
        xor_result = bitarray(bitarray1) ^ bitarray(bitarray2)
        return xor_result.count()

    @staticmethod
    def euclidean_dist(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.array(x) - y
        return np.sqrt(np.dot(diff, diff))

    @staticmethod
    def euclidean_dist_square(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.array(x) - y
        return np.dot(diff, diff)

    @staticmethod
    def euclidean_dist_centred(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.mean(x) - np.mean(y)
        return np.dot(diff, diff)

    @staticmethod
    def l1norm_dist(x, y):
        return sum(abs(x - y))

    @staticmethod
    def cosine_dist(x, y):
        return 1 - np.dot(x, y) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)
