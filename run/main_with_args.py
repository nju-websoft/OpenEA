import argparse

from openea.modules.args.args_hander import check_args, load_args
from openea.modules.load.kgs import read_kgs_from_folder, read_kgs_from_dbp_dwy
from openea.models.basic_model import BasicModel
from openea.models.trans import TransD
from openea.models.trans import TransE
from openea.models.trans import TransH
from openea.models.trans import TransR
from openea.models.semantic import DistMult
from openea.models.semantic import HolE
from openea.models.semantic import SimplE
from openea.models.semantic import RotatE
from openea.models.neural import ConvE
from openea.models.neural import ProjE
from openea.approaches import AlignE
from openea.approaches import BootEA
from openea.approaches import JAPE
from openea.approaches import Attr2Vec
from openea.approaches import MTransE
from openea.approaches import IPTransE
from openea.approaches import GCN_Align
from openea.approaches import GMNN
from openea.approaches import KDCoE
from openea.approaches import SEA
from openea.approaches import RSN4EA
from openea.approaches import RDGCN
from openea.approaches import AliNet

parser = argparse.ArgumentParser(description='OpenEA')
parser.add_argument('--training_data', type=str, default='../../datasets/EN_DE_15K_V1/')
parser.add_argument('--output', type=str, default='../../output/results/')
parser.add_argument('--dataset_division', type=str, default='721_5fold/1/')

parser.add_argument('--embedding_module', type=str, default='AliNet',
                    choices=['BasicModel',
                             'TransE', 'TransD', 'TransH', 'TransR',
                             'DistMult', 'HolE', 'SimplE', 'RotatE', 'ProjE', 'ConvE', 'SEA', 'RSN4EA',
                             'JAPE', 'Attr2Vec', 'MTransE', 'AlignE', 'BootEA', 'GCN_Align', "GMNN", 'KDCoE', 'RDGCN'])

parser.add_argument('--init', type=str, default='xavier', choices=['normal', 'unit', 'xavier', 'uniform'])
parser.add_argument('--alignment_module', type=str, default='mapping', choices=['sharing', 'mapping', 'swapping'])
parser.add_argument('--search_module', type=str, default='greedy', choices=['greedy', 'global'])
parser.add_argument('--loss', type=str, default='limited', choices=['margin-based', 'logistic', 'limited'])
parser.add_argument('--neg_sampling', type=str, default='truncated', choices=['uniform', 'truncated'])

parser.add_argument('--dim', type=int, default=300)
parser.add_argument('--loss_norm', type=str, default='L2')
parser.add_argument('--ent_l2_norm', type=bool, default=True)
parser.add_argument('--rel_l2_norm', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=3000)

parser.add_argument('--margin', type=float, default=1.5)
parser.add_argument('--pos_margin', type=float, default=0)
parser.add_argument('--neg_margin', type=float, default=1.5)
parser.add_argument('--neg_margin_balance', type=float, default=0.1)

parser.add_argument('--neg_triple_num', type=int, default=10)
parser.add_argument('--truncated_epsilon', type=float, default=0.98)
parser.add_argument('--truncated_freq', type=int, default=10)

parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adagrad', 'Adadelta', 'Adam', 'SGD'])
parser.add_argument('--batch_threads_num', type=int, default=4)
parser.add_argument('--test_threads_num', type=int, default=12)
parser.add_argument('--max_epoch', type=int, default=1000)
parser.add_argument('--eval_freq', type=int, default=10)

parser.add_argument('--ordered', type=bool, default=True)
parser.add_argument('--top_k', type=list, default=[1, 5, 10, 50])
parser.add_argument('--csls', type=int, default=10)

parser.add_argument('--is_save', type=bool, default=True)
parser.add_argument('--eval_norm', type=bool, default=True)
parser.add_argument('--start_valid', type=int, default=0)
parser.add_argument('--stop_metric', type=str, default='mrr', choices=['hits1', 'mrr'])
parser.add_argument('--eval_metric', type=str, default='inner', choices=['inner', 'cosine', 'euclidean', 'manhattan'])

parser.add_argument('--rnn_layer_num', type=int, default=2)
parser.add_argument('--output_keep_prob', type=float, default=0.5)  # dropout
parser.add_argument('--dnn_neg_nums', type=int, default=1024)  # negative sampling
parser.add_argument('--filter_num', type=int, default=32)  # number of filters

parser.add_argument('--sim_th', type=float, default=0.5)  # For bootstrapping
parser.add_argument('--k', type=int, default=10)  # For bootstrapping

parser.add_argument('--likelihood_slice', type=int, default=1000)  # For BootEA likelihood matrix
parser.add_argument('--sub_epoch', type=int, default=20)  # For BootEA

parser.add_argument('--neg_alpha', type=float, default=0.1)  # For JAPE
parser.add_argument('--top_attr_threshold', type=float, default=0.95)  # For JAPE attribute selection
parser.add_argument('--attr_max_epoch', type=int, default=100)  # For JAPE
parser.add_argument('--attr_sim_mat_threshold', type=float, default=0.95)  # For JAPE attribute selection
parser.add_argument('--attr_sim_mat_beta', type=float, default=0.05)  # For JAPE

parser.add_argument('--alpha', type=float, default=0.1)  # For MTransE and KDCoE

parser.add_argument("--support_number", type=int, default=1)  # For GCN
parser.add_argument("--se_dim", type=int, default=100)  # For GCN
parser.add_argument("--ae_dim", type=int, default=100)  # For GCN
parser.add_argument("--weight_decay", type=float, default=0.000)  # For GCN & GMNN
parser.add_argument("--hidden1", type=int, default=100)  # For GCN
parser.add_argument("--gamma", type=float, default=1)  # For GCN
parser.add_argument("--dropout", type=float, default=0.)  # For GCN
parser.add_argument("--test_method", type=str, default="sa")  # For GCN
parser.add_argument("--beta", type=float, default=0.3)  # For GCN

parser.add_argument("--use_pretrained_embedding", type=bool, default=False)  # For GMNN
parser.add_argument("--embedding_path", type=str, default="../../../Crosslingula-KG-Matching-master/DBP15K/DBP15K/sub.glove.300d")  # For GMNN
parser.add_argument("--word_embedding_dim", type=int, default=100)  # For GMNN
parser.add_argument("--train_cand_size", type=int, default=20)  # For GMNN
parser.add_argument("--train_batch_size", type=int, default=20)  # For GMNN
parser.add_argument("--l2_lambda", type=float, default=0.000001)  # For GMNN
parser.add_argument("--encoder_hidden_dim", type=int, default=200)  # For GMNN
parser.add_argument("--word_size_max", type=int, default=1)  # For GMNN
parser.add_argument("--node_vec_method", type=str, default="lstm")  # For GMNN
parser.add_argument("--pretrained_word_embedding_pat", type=str, default="")  # For GMNN

parser.add_argument("--unknown_word", type=str, default="**UNK**")  # For GMNN
parser.add_argument("--deal_unknown_words", type=bool, default=True)  # For GMNN
parser.add_argument("--pretrained_word_embedding_dim", type=int, default=300)  # For GMNN
parser.add_argument("--num_layers", type=int, default=1)  # For GMNN
parser.add_argument("--sample_size_per_layer", type=int, default=1)  # For GMNN
parser.add_argument("--hidden_layer_dim", type=int, default=100)  # For GMNN
parser.add_argument("--feature_max_len", type=int, default=1)  # For GMNN
parser.add_argument("--feature_encode_type", type=str, default="uni")  # For GMNN
parser.add_argument("--graph_encode_direction", type=str, default="bi")  # For GMNN
parser.add_argument("--concat", type=bool, default=True)  # For GMNN
parser.add_argument("--encoder", type=str, default="gated_gcn")  # For GMNN
parser.add_argument("--lstm_in_gcn", type=str, default="none")  # For GMNN
parser.add_argument("--aggregator_dim_first", type=int, default=100)  # For GMNN
parser.add_argument("--aggregator_dim_second", type=int, default=100)  # For GMNN
parser.add_argument("--gcn_window_size_first", type=int, default=1)  # For GMNN
parser.add_argument("--gcn_window_size_second", type=int, default=2)  # For GMNN
parser.add_argument("--gcn_layer_size_first", type=int, default=1)  # For GMNN
parser.add_argument("--gcn_layer_size_second", type=int, default=1)  # For GMNN
parser.add_argument("--with_match_highway", type=bool, default=False)  # For GMNN
parser.add_argument("--with_gcn_highway", type=bool, default=False)  # For GMNN
parser.add_argument("--if_use_multiple_gcn_1_state", type=bool, default=False)  # For GMNN
parser.add_argument("--if_use_multiple_gcn_2_state", type=bool, default=False)  # For GMNN
parser.add_argument("--agg_sim_method", type=str, default="GCN")  # For GMNN
parser.add_argument("--gcn_type_first", type=str, default='mean_pooling')  # For GMNN
parser.add_argument("--gcn_type_second", type=str, default='mean_pooling')  # For GMNN
parser.add_argument("--cosine_MP_dim", type=int, default=10)  # For GMNN
parser.add_argument("--pred_method", type=str, default="node_level")  # For GMNN
parser.add_argument("--cand_size", type=int, default=100)  # For GMNN
parser.add_argument("--build_train_examples", type=bool, default=False)  # For GMNN
parser.add_argument("--dev_batch_size", type=int, default=64)  # For GMNN
parser.add_argument("--word2vec_path", type=str, default="../../datasets/wiki-news-300d-1M.vec")  # For RDGCN
parser.add_argument("--word2vec_dim", type=int, default=300)  # For RDGCN
parser.add_argument("--literal_len", type=int, default=10)
parser.add_argument("--encoder_normalize", type=bool, default=True)
parser.add_argument("--encoder_epoch", type=int, default=100)
parser.add_argument("--retrain_literal_embeds", type=bool, default=True)
parser.add_argument("--literal_normalize", type=bool, default=True)
parser.add_argument("--encoder_active", type=str, default="thah")

parser.add_argument("--desc_batch_size", type=int, default=512)  # For KDCoE
parser.add_argument("--wv_dim", type=int, default=300)  # For KDCoE
parser.add_argument("--default_desc_length", type=int, default=4)  # For KDCoE
parser.add_argument("--word_embed", type=str, default="../../datasets/wiki-news-300d-1M.vec")  # For KDCoE
parser.add_argument("--desc_sim_th", type=int, default=0.99)  # For KDCoE

parser.add_argument('--adj_number', type=int, default=1)  # For AliNet
parser.add_argument('--layer_dims', type=list, default=[500, 400, 300])  # For AliNet
parser.add_argument('--min_rel_win', type=int, default=15)  # For AliNet
parser.add_argument('--start_augment', type=int, default=2)  # For AliNet
parser.add_argument('--rel_param', type=float, default=0.01)  # For AliNet
parser.add_argument('--num_features_nonzero', type=float, default=0.0)  # For AliNet

args = parser.parse_args()
print(args)


# with open("args.json", "w") as f:
#     json.dump(args.__dict__, f)


class ModelFamily(object):
    BasicModel = BasicModel

    TransE = TransE
    TransD = TransD
    TransH = TransH
    TransR = TransR

    DistMult = DistMult
    HolE = HolE
    SimplE = SimplE
    RotatE = RotatE

    ProjE = ProjE
    ConvE = ConvE
    RSN4EA = RSN4EA
    SEA = SEA

    MTransE = MTransE
    IPTransE = IPTransE
    Attr2Vec = Attr2Vec
    JAPE = JAPE
    AlignE = AlignE
    BootEA = BootEA
    GCN_Align = GCN_Align
    GMNN = GMNN
    KDCoE = KDCoE
    RDGCN = RDGCN
    AliNet = AliNet


def get_model(model_name):
    return getattr(ModelFamily, model_name)


if __name__ == '__main__':
    kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, args.ordered)
    model = get_model(args.embedding_module)()
    model.set_args(args)
    model.set_kgs(kgs)
    model.init()
    model.run()
    model.test()
    model.save()
