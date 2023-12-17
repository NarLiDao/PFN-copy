import os
import argparse
import tensorflow as tf

from time import time
from model import OurModel
from LoadData import LoadData as DATA


def parse_arg():
    parser = argparse.ArgumentParser(description="Run OurModel.")
    parser.add_argument('--path', nargs='?', default='../data/',
                        help='Input data path.')
    parser.add_argument('--model', nargs='?', default='pfn')
    parser.add_argument('--dataset', nargs='?', default='movielens',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of epochs.')
    parser.add_argument('--valid_dim', type=int, default=3,
                        help='Valid dimension of the dataset. (e.g. frappe=10, ml-tag=3,avazu=22,criteo=39)')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='Pre-train flag. 0: train from scratch; 1: load from pretrain file)
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.') 
    parser.add_argument('--early_stop', type=int, default=1,
                        help='Whether to perform early stop (0 or 1)')
    parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='auc',
                        help='Specify a loss type (rmse or logloss or auc).')
    parser.add_argument('--hidden_factor', type=int, default=90,
                        help='Number of hidden factors.') 
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='Whether to perform batch normaization (0 or 1))
    parser.add_argument('--hidden_units_mlp', nargs='?', default='[128,2048]',
                        help="Size of each layer.")
    parser.add_argument('--activation', nargs='?', default='relu',
                        help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
    parser.add_argument('--dropout', nargs='?', default='[0.9]')
    parser.add_argument('--n_heads', type=int, default=6)

    parser.add_argument('--train_num', type=int, default=210000)
    parser.add_argument('--valid_num', type=int, default=60000)
    parser.add_argument('--test_num', type=int, default=30000)

    return parser.parse_args()

#Precision
def metric_precision(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    TP = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    precision = TP / (TP + FP)
    return precision


# Recall
def metric_recall(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    # print('y_true, y_pred', y_true, y_pred) 
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    recall = TP / (TP + FN)
    return recall


# F1-score
def metric_F1score(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1score = 2 * precision * recall / (precision + recall)
    return F1score


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    args = parse_arg()
    data = DATA(args.path, args.dataset, args.loss_type, args.train_num, args.valid_num, args.test_num)
    if args.verbose > 0:
        print(
            'OurModel: dataset=%s, loss_type=%s, train_valid=%d,pretrain=%d, batch=%d, lr=%.4f, lambda=%.4f, optimizer=%s, batch_norm=%d, activation=%s'
            % (
                args.dataset, args.loss_type, args.train_num, args.pretrain, args.batch_size, args.lr, args.mlattention,
                args.optimizer, args.batch_norm, args.activation,
            ))
    activation_function = tf.nn.relu
    if args.activation == 'sigmoid':
        activation_function = tf.sigmoid
    elif args.activation == 'tanh':
        activation_function = tf.tanh
    elif args.activation == 'identity':
        activation_function = tf.identity

    t1 = time()
    model = OurModel(data.features_M, eval(args.hidden_units_mlp), activation_function,
                     eval(args.dropout), args.n_heads, args.dataset, args.epoch, args.pretrain,
                     args.batch_size, args.attention, args.mlattention, args.early_stop, args.optimizer, args.lr,
                     args.loss_type, args.hidden_factor, args.verbose, args.batch_norm, args.valid_dim, args.model)

    model.train(data.Train_data, data.Validation_data, data.Test_data)
    best_valid_score = 0
    if args.loss_type == 'rmse':
        best_valid_score = min(model.test_pfn)
    elif args.loss_type == 'logloss':
        best_valid_score = min(model.test_pfn)
    else:
        best_valid_score = max(model.test_pfn)
    best_epoch = model.test_pfn.index(best_valid_score)
    print("Best Iter(test)= %d\t train = %.4f, valid = %.4f, test = %.4f"
          % (
              best_epoch + 1, model.train_pfn[best_epoch], model.valid_pfn[best_epoch], model.test_pfn[best_epoch]))

    print('dataset=%s\t metric=%s\t valid_dim=%s\t hidden_factor=%d' % (
        args.dataset, args.loss_type, args.valid_dim, args.hidden_factor))
    print('model=%s\ttrain=%d,valid=%d,test=%d' % (args.model, args.train_num, args.valid_num, args.test_num))
    print(
        'batch=%d, lr=%f, lambda=%f, optimizer=%s, batch_norm=%d, activation=%s' % (
            args.batch_size, args.lr, args.mlattention, args.optimizer, args.batch_norm, args.activation))
