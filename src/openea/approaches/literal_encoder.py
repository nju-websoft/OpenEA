import numpy as np
import gc
from sklearn import preprocessing
from gensim.models.word2vec import Word2Vec

from openea.modules.base.optimizers import generate_optimizer
from openea.modules.utils.util import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    s1 = np.exp(x) - np.exp(-x)
    s2 = np.exp(x) + np.exp(-x)
    s = s1 / s2
    return s


class AutoEncoderModel:
    def __init__(self, word_vec_list, args, hidden_dimensions=None):
        self.session = load_session()
        self.args = args
        self.weights, self.biases = {}, {}
        self.input_dimension = self.args.literal_len * self.args.word2vec_dim
        if hidden_dimensions is None:
            hidden_dimensions = [1024, 512, self.args.dim]
        self.hidden_dimensions = hidden_dimensions
        self.layer_num = len(self.hidden_dimensions)
        self.encoder_output = None
        self.decoder_output = None
        self.decoder_op = None

        self.word_vec_list = np.reshape(word_vec_list, [len(word_vec_list), self.input_dimension])
        if self.args.encoder_normalize:
            self.word_vec_list = preprocessing.normalize(self.word_vec_list)

        self._init_graph()
        self._loss_optimizer()
        tf.global_variables_initializer().run(session=self.session)

    def _init_graph(self):
        self.hidden_dimensions.insert(0, self.input_dimension)
        hds = self.hidden_dimensions
        for i in range(self.layer_num):
            self.weights['encoder_h' + str(i)] = tf.get_variable('encoder_h' + str(i),
                                                                 initializer=tf.random_normal_initializer,
                                                                 shape=[hds[i], hds[i + 1]], dtype=tf.float32)
            self.biases['encoder_b' + str(i)] = tf.get_variable('encoder_b' + str(i),
                                                                initializer=tf.random_normal_initializer,
                                                                shape=[hds[i + 1]], dtype=tf.float32)
        for i in range(self.layer_num):
            i_decoder = self.layer_num - i
            self.weights['decoder_h' + str(i)] = tf.get_variable('decoder_h' + str(i),
                                                                 initializer=tf.random_normal_initializer,
                                                                 shape=[hds[i_decoder], hds[i_decoder - 1]],
                                                                 dtype=tf.float32)

            self.biases['decoder_b' + str(i)] = tf.get_variable('decoder_b' + str(i),
                                                                initializer=tf.random_normal_initializer,
                                                                shape=[hds[i_decoder - 1]], dtype=tf.float32)
            self.batch = tf.placeholder(tf.float32, shape=[None, self.input_dimension])

    def _loss_optimizer(self):
        encoder_output = self.encoder(self.batch)
        if self.args.encoder_normalize:
            encoder_output = tf.nn.l2_normalize(encoder_output)
        decoder_output = self.decoder(encoder_output)
        self.loss = tf.reduce_mean(tf.pow(decoder_output - self.batch, 2))
        # self.optimizer = generate_optimizer(self.loss, self.args.learning_rate, opt=self.args.optimizer)
        self.optimizer = generate_optimizer(self.loss, 0.01, opt='Adagrad')

    def encoder(self, input_data):
        input_layer = input_data
        for i in range(self.layer_num):
            input_layer = tf.add(tf.matmul(input_layer, self.weights['encoder_h' + str(i)]), self.biases['encoder_b' + str(i)])
            if self.args.encoder_active == 'sigmoid':
                input_layer = tf.nn.sigmoid(input_layer)
            elif self.args.encoder_active == 'tanh':
                input_layer = tf.nn.tanh(input_layer)
        encoder_output = input_layer
        return encoder_output

    def decoder(self, input_data):
        input_layer = input_data
        for i in range(self.layer_num):
            input_layer = tf.add(tf.matmul(input_layer, self.weights['decoder_h' + str(i)]), self.biases['decoder_b' + str(i)])
            if self.args.encoder_active == 'sigmoid':
                input_layer = tf.nn.sigmoid(input_layer)
            elif self.args.encoder_active == 'tanh':
                input_layer = tf.nn.tanh(input_layer)
        decoder_output = input_layer
        return decoder_output

    def train_one_epoch(self, epoch):
        start_time = time.time()

        batches = list()
        batch_size = self.args.batch_size
        num_batch = len(self.word_vec_list) // batch_size + 1
        for i in range(num_batch):
            if i == num_batch - 1:
                batches.append(self.word_vec_list[i * batch_size:])
            else:
                batches.append(self.word_vec_list[i * batch_size:(i + 1) * batch_size])

        loss_sum = 0.0
        for batch_i in range(num_batch):
            loss_train, _ = self.session.run([self.loss, self.optimizer], feed_dict={self.batch: batches[batch_i]})
            loss_sum += loss_train
        loss_sum += self.args.batch_size
        end = time.time()
        print('epoch {} of literal encoder, loss: {:.4f}, time: {:.4f}s'.format(epoch, loss_sum, end - start_time))
        return

    def encoder_multi_batches(self, input_data):
        print('encode literal embeddings...', len(input_data))
        batches = list()
        results = np.zeros((len(input_data), self.args.dim))
        batch_size = self.args.batch_size
        num_batch = len(input_data) // batch_size + 1
        for i in range(num_batch):
            if i == num_batch - 1:
                batches.append(input_data[i * batch_size:])
            else:
                batches.append(input_data[i * batch_size:(i + 1) * batch_size])

        for batch_i in range(num_batch):
            input_layer = np.reshape(batches[batch_i], [len(batches[batch_i]), self.input_dimension])
            for i in range(self.layer_num):
                weight_i = self.weights['encoder_h' + str(i)].eval(session=self.session)
                bias_i = self.biases['encoder_b' + str(i)].eval(session=self.session)
                input_layer = np.matmul(input_layer, weight_i) + bias_i
                if self.args.encoder_active == 'sigmoid':
                    input_layer = sigmoid(input_layer)
                elif self.args.encoder_active == 'tanh':
                    input_layer = tanh(input_layer)
            literal_vectors = input_layer
            if batch_i == num_batch - 1:
                results[batch_i * batch_size:] = np.array(literal_vectors)
            else:
                results[batch_i * batch_size:(batch_i + 1) * batch_size] = np.array(literal_vectors)
            del literal_vectors
            gc.collect()
        print("encoded literal embeddings", results.shape)
        return results


def generate_unlisted_word2vec(word2vec, literal_list, vector_dimension):
    unlisted_words = []
    for literal in literal_list:
        words = literal.split(' ')
        for word in words:
            if word not in word2vec:
                unlisted_words.append(word)

    character_vectors = {}
    alphabet = ''
    ch_num = {}
    for word in unlisted_words:
        for ch in word:
            n = 1
            if ch in ch_num:
                n += ch_num[ch]
            ch_num[ch] = n
    ch_num = sorted(ch_num.items(), key=lambda x: x[1], reverse=True)
    ch_sum = sum([n for (_, n) in ch_num])
    for i in range(len(ch_num)):
        if ch_num[i][1] / ch_sum >= 0.0001:
            alphabet += ch_num[i][0]
    print(alphabet)
    print('len(alphabet):', len(alphabet), '\n')
    char_sequences = [list(word) for word in unlisted_words]
    model = Word2Vec(char_sequences, size=vector_dimension, window=5, min_count=1)
    for ch in alphabet:
        assert ch in model
        character_vectors[ch] = model[ch]

    word2vec_new = {}
    for word in unlisted_words:
        vec = np.zeros(vector_dimension, dtype=np.float32)
        for ch in word:
            if ch in alphabet:
                vec += character_vectors[ch]
        if len(word) != 0:
            word2vec_new[word] = vec / len(word)

    word2vec.update(word2vec_new)
    return word2vec


class LiteralEncoder:

    def __init__(self, literal_list, word2vec, args, word2vec_dimension):
        self.args = args
        self.literal_list = literal_list
        self.word2vec = generate_unlisted_word2vec(word2vec, literal_list, word2vec_dimension)
        self.tokens_max_len = self.args.literal_len
        self.word2vec_dimension = word2vec_dimension

        literal_vector_list = []
        for literal in self.literal_list:
            vectors = np.zeros((self.tokens_max_len, self.word2vec_dimension), dtype=np.float32)
            words = literal.split(' ')
            for i in range(min(self.tokens_max_len, len(words))):
                if words[i] in self.word2vec:
                    vectors[i] = self.word2vec[words[i]]
            literal_vector_list.append(vectors)
        assert len(literal_list) == len(literal_vector_list)
        encoder_model = AutoEncoderModel(literal_vector_list, self.args)
        for i in range(self.args.encoder_epoch):
            encoder_model.train_one_epoch(i + 1)
        self.encoded_literal_vector = encoder_model.encoder_multi_batches(literal_vector_list)



