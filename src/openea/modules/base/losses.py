import tensorflow as tf


def get_loss_func(phs, prs, pts, nhs, nrs, nts, args):
    triple_loss = None
    if args.loss == 'margin-based':
        triple_loss = margin_loss(phs, prs, pts, nhs, nrs, nts, args.margin, args.loss_norm)
    elif args.loss == 'logistic':
        triple_loss = logistic_loss(phs, prs, pts, nhs, nrs, nts, args.loss_norm)
    elif args.loss == 'limited':
        triple_loss = limited_loss(phs, prs, pts, nhs, nrs, nts, args.pos_margin, args.neg_margin, args.loss_norm)
    return triple_loss


def margin_loss(phs, prs, pts, nhs, nrs, nts, margin, loss_norm):
    with tf.name_scope('margin_loss_distance'):
        pos_distance = phs + prs - pts
        neg_distance = nhs + nrs - nts
    with tf.name_scope('margin_loss'):
        if loss_norm == 'L1':  # L1 normal
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.abs(neg_distance), axis=1)
        else:  # L2 normal
            pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.square(neg_distance), axis=1)
        loss = tf.reduce_sum(tf.nn.relu(tf.constant(margin) + pos_score - neg_score), name='margin_loss')
    return loss


def positive_loss(phs, prs, pts, loss_norm):
    with tf.name_scope('positive_loss_distance'):
        pos_distance = phs + prs - pts
    with tf.name_scope('positive_loss_score'):
        if loss_norm == 'L1':  # L1 score
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
        else:  # L2 score
            pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
        loss = tf.reduce_sum(pos_score, name='positive_loss')
    return loss


def limited_loss(phs, prs, pts, nhs, nrs, nts, pos_margin, neg_margin, loss_norm, balance=1.0):
    with tf.name_scope('limited_loss_distance'):
        pos_distance = phs + prs - pts
        neg_distance = nhs + nrs - nts
    with tf.name_scope('limited_loss_score'):
        if loss_norm == 'L1':  # L1 score
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.abs(neg_distance), axis=1)
        else:  # L2 score
            pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.square(neg_distance), axis=1)
        pos_loss = tf.reduce_sum(tf.nn.relu(pos_score - tf.constant(pos_margin)))
        neg_loss = tf.reduce_sum(tf.nn.relu(tf.constant(neg_margin) - neg_score))
        loss = tf.add(pos_loss, balance * neg_loss, name='limited_loss')
    return loss


def logistic_loss(phs, prs, pts, nhs, nrs, nts, loss_norm):
    with tf.name_scope('logistic_loss_distance'):
        pos_distance = phs + prs - pts
        neg_distance = nhs + nrs - nts
    with tf.name_scope('logistic_loss_score'):
        if loss_norm == 'L1':  # L1 score
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.abs(neg_distance), axis=1)
        else:  # L2 score
            pos_score = tf.reduce_sum(tf.square(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.square(neg_distance), axis=1)
        pos_loss = tf.reduce_sum(tf.log(1 + tf.exp(pos_score)))
        neg_loss = tf.reduce_sum(tf.log(1 + tf.exp(-neg_score)))
        loss = tf.add(pos_loss, neg_loss, name='logistic_loss')
    return loss


def mapping_loss(tes1, tes2, mapping, eye):
    mapped_tes2 = tf.matmul(tes1, mapping)
    map_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(tes2 - mapped_tes2, 2), 1))
    orthogonal_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(tf.matmul(mapping, mapping, transpose_b=True) - eye, 2), 1))
    return map_loss + orthogonal_loss
