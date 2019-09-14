import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/grouping'))
import tf_util
import pointfly as pf
from tf_grouping import group_point, knn_point

# A shape is (N, P, C)
def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    cls_labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl, cls_labels_pl

def RIConv(pts, fts_prev, qrs, is_training, tag, K, D, P, C, with_local, bn_decay=None):

    indices = pf.knn_indices_general(qrs, pts, int(K), True)
    nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts')
    
    nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')
    nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local') 
    dists_local = tf.norm(nn_pts_local, axis=-1, keepdims=True)  # dist to center

    mean_local = tf.reduce_mean(nn_pts, axis=-2, keepdims=True)
    mean_global = tf.reduce_mean(pts, axis=-2, keepdims=True)
    mean_global = tf.expand_dims(mean_global, axis=-2)

    nn_pts_local_mean = tf.subtract(nn_pts, mean_local, name=tag + 'nn_pts_local_mean') 
    dists_local_mean = tf.norm(nn_pts_local_mean, axis=-1, keepdims=True) # dist to local mean

    vec = mean_local - nn_pts_center
    vec_dist = tf.norm(vec, axis=-1, keepdims =True)
    vec_norm = tf.divide(vec, vec_dist)
    vec_norm = tf.where(tf.is_nan(vec_norm), tf.ones_like(vec_norm) * 0, vec_norm) 

    nn_pts_local_proj = tf.matmul(nn_pts_local, vec_norm, transpose_b=True)
    nn_pts_local_proj_dot = tf.divide(nn_pts_local_proj, dists_local)
    nn_pts_local_proj_dot = tf.where(tf.is_nan(nn_pts_local_proj_dot), tf.ones_like(nn_pts_local_proj_dot) * 0, nn_pts_local_proj_dot)  # check nan

    nn_pts_local_proj_2 = tf.matmul(nn_pts_local_mean, vec_norm, transpose_b=True)
    nn_pts_local_proj_dot_2 = tf.divide(nn_pts_local_proj_2, dists_local_mean)
    nn_pts_local_proj_dot_2 = tf.where(tf.is_nan(nn_pts_local_proj_dot_2), tf.ones_like(nn_pts_local_proj_dot_2) * 0, nn_pts_local_proj_dot_2)  # check nan

    nn_fts = tf.concat([dists_local, dists_local_mean, nn_pts_local_proj_dot, nn_pts_local_proj_dot_2], axis=-1) # d0 d1 a0 a1
    
    # compute indices from nn_pts_local_proj
    vec = mean_global - nn_pts_center
    vec_dist = tf.norm(vec, axis=-1, keepdims =True)
    vec_norm = tf.divide(vec, vec_dist)
    nn_pts_local_proj = tf.matmul(nn_pts_local, vec_norm, transpose_b=True)

    proj_min = tf.reduce_min(nn_pts_local_proj, axis=-2, keepdims=True) 
    proj_max = tf.reduce_max(nn_pts_local_proj, axis=-2, keepdims=True) 
    seg = (proj_max - proj_min) / D

    vec_tmp = tf.range(0, D, 1, dtype=tf.float32)
    vec_tmp = tf.reshape(vec_tmp, (1,1,1,D))

    limit_bottom = vec_tmp * seg + proj_min
    limit_up = limit_bottom + seg

    idx_up = nn_pts_local_proj <= limit_up
    idx_bottom = nn_pts_local_proj >= limit_bottom
    idx = tf.to_float(tf.equal(idx_bottom, idx_up))
    idx_expand = tf.expand_dims(idx, axis=-1)

    
    
    [N,P,K,dim] = nn_fts.shape # (N, P, K, 3)
    nn_fts_local = None
    if with_local:
        C_pts_fts = 64
        nn_fts_local_reshape = tf.reshape(nn_fts, (-1,P*K,dim,1))
        nn_fts_local = tf_util.conv2d(nn_fts_local_reshape, C_pts_fts//2, [1,dim],
                         padding='VALID', stride=[1,1], 
                         bn=True, is_training=is_training,
                         scope=tag+'conv_pts_fts_0', bn_decay=bn_decay)
        nn_fts_local = tf_util.conv2d(nn_fts_local, C_pts_fts, [1,1],
                         padding='VALID', stride=[1,1], 
                         bn=True, is_training=is_training,
                         scope=tag+'conv_pts_fts_1', bn_decay=bn_decay)
        nn_fts_local = tf.reshape(nn_fts_local, (-1,P,K,C_pts_fts))
    else:
        nn_fts_local = nn_fts
    
    if fts_prev is not None:
        fts_prev = tf.gather_nd(fts_prev, indices, name=tag + 'fts_prev')  # (N, P, K, 3)
        pts_X_0 = tf.concat([nn_fts_local,fts_prev], axis=-1)
    else:
        pts_X_0 = nn_fts_local

    pts_X_0_expand = tf.expand_dims(pts_X_0, axis=-2)
    nn_fts_rect = pts_X_0_expand * idx_expand
    idx = tf.to_float(nn_fts_rect == 0.0)
    nn_fts_rect = nn_fts_rect + idx*(-99999999999.0)
    nn_fts_rect = tf.reduce_max(nn_fts_rect, axis=-3)
    
    # nn_fts_rect = tf.matmul(idx_mean, pts_X_0, transpose_a = True)
        
    fts_X = tf_util.conv2d(nn_fts_rect, C, [1,nn_fts_rect.shape[-2].value],
                         padding='VALID', stride=[1,1], 
                         bn=True, is_training=is_training,
                         scope=tag+'conv', bn_decay=bn_decay)
    return tf.squeeze(fts_X, axis=-2) 

    
def get_model(layer_pts, is_training, RIconv_params, RIdconv_params, fc_params, sampling='fps', weight_decay=0.0, bn_decay=None, part_num=50):
    
    if sampling == 'fps':
        sys.path.append(os.path.join(BASE_DIR, 'tf_ops/sampling'))
        from tf_sampling import farthest_point_sample, gather_point

    layer_fts_list = [None]
    layer_pts_list = [layer_pts]
    for layer_idx, layer_param in enumerate(RIconv_params):
        tag = 'xconv_' + str(layer_idx + 1) + '_'
        K = layer_param['K']
        D = layer_param['D']
        P = layer_param['P']
        C = layer_param['C']
        # qrs = layer_pts if P == -1 else layer_pts[:,:P,:]  # (N, P, 3)

        if P == -1:
            qrs = layer_pts
        else:
            if sampling == 'fps':
                qrs = gather_point(layer_pts, farthest_point_sample(P, layer_pts))
            elif sampling == 'random':
                qrs = tf.slice(layer_pts, (0, 0, 0), (-1, P, -1), name=tag + 'qrs')  # (N, P, 3)
            else:
                print('Unknown sampling method!')
                exit()
        
        layer_fts= RIConv(layer_pts_list[-1], layer_fts_list[-1], qrs, is_training, tag, K, D, P, C, True, bn_decay)
        
        layer_pts = qrs
        layer_pts_list.append(qrs)
        layer_fts_list.append(layer_fts)
  
    if RIdconv_params is not None:
        fts = layer_fts_list[-1]
        for layer_idx, layer_param in enumerate(RIdconv_params):
            tag = 'xdconv_' + str(layer_idx + 1) + '_'
            K = layer_param['K'] 
            D = layer_param['D'] 
            pts_layer_idx = layer_param['pts_layer_idx']  # 2 1 0 
            qrs_layer_idx = layer_param['qrs_layer_idx']  # 1 0 -1

            pts = layer_pts_list[pts_layer_idx + 1]
            qrs = layer_pts_list[qrs_layer_idx + 1]
            fts_qrs = layer_fts_list[qrs_layer_idx + 1]

            C = fts_qrs.get_shape()[-1].value if fts_qrs is not None else C//2
            P = qrs.get_shape()[1].value
            
            layer_fts= RIConv(pts, fts, qrs, is_training, tag, K, D, P, C, True, bn_decay)
            if fts_qrs is not None: # this is for last layer
                fts_concat = tf.concat([layer_fts, fts_qrs], axis=-1, name=tag + 'fts_concat')
                fts = pf.dense(fts_concat, C, tag + 'fts_fuse', is_training)
            else:
                fts = layer_fts
        
    for layer_idx, layer_param in enumerate(fc_params):
        C = layer_param['C']
        dropout_rate = layer_param['dropout_rate']
        layer_fts = pf.dense(layer_fts, C, 'fc{:d}'.format(layer_idx), is_training)
        layer_fts = tf.layers.dropout(layer_fts, dropout_rate, training=is_training, name='fc{:d}_drop'.format(layer_idx))
    
    logits_seg = pf.dense(layer_fts, part_num, 'logits', is_training, with_bn=False, activation=None)

    return logits_seg

def get_loss(seg_pred, seg_label):
    """ pred: BxNxC,
        label: BxN, """
    

    # size of seg_pred is batch_size x point_num x part_cat_num
    # size of seg is batch_size x point_num
    per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg_label), axis=1)
    seg_loss = tf.reduce_mean(per_instance_seg_loss)

    per_instance_seg_pred_res = tf.argmax(seg_pred, 2)

    return seg_loss, per_instance_seg_loss, per_instance_seg_pred_res


if __name__ == '__main__':
    print('This is the Rotaion Invairant Convolution Operator')
