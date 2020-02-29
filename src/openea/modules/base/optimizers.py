import tensorflow as tf


def generate_optimizer(loss, learning_rate, var_list=None, opt='SGD'):
    optimizer = get_optimizer(opt, learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
    return optimizer.apply_gradients(grads_and_vars)


def get_optimizer(opt, learning_rate):
    if opt == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif opt == 'Adadelta':
        # To match the exact form in the original paper use 1.0.
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif opt == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    else:  # opt == 'SGD'
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return optimizer
