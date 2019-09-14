import os
import sys
import h5py
import argparse
import importlib
import numpy as np
from time import time
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='RIConv', help='Model name: RIConv')
parser.add_argument('--log_dir', default='log/classification', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, '../data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, '../data/modelnet40_ply_hdf5_2048/test_files.txt'))

conv_param_name = ('K', 'D', 'P', 'C', 'links')
conv_params = [dict(zip(conv_param_name, conv_param)) for conv_param in
                [
                 (64, 4, 256, 128, []),
                 (32, 2, 128, 256, []),
                 (16, 1, 64, 512, [])]]

x = 4
fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
            [(128 * x, 0.0),
            (64 * x, 0.5)]]
WITH_LOCAL = True
WITH_MULTI = True

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    shape_names = [line.rstrip() for line in \
        open('../data/modelnet40_ply_hdf5_2048/shape_names.txt')] 

    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl = tf.placeholder(tf.float32, [None, NUM_POINT, 3], name='pointclouds_pl')
            labels_pl = tf.placeholder(tf.int64, [None], name='labels_pl')
            is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training_pl')
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred = MODEL.get_model(pointclouds_pl, is_training_pl, conv_params, None, fc_params, sampling='fps', weight_decay=0.0, bn_decay=bn_decay, part_num=NUM_CLASSES)
            
            if WITH_MULTI:
                labels_2d = tf.expand_dims(labels_pl, axis=-1, name='labels_2d')
                labels_2d = tf.tile(labels_2d, (1, pred.shape[1]), name='labels_2d_pl')
                loss = tf.losses.sparse_softmax_cross_entropy(labels=labels_2d, logits=pred)
                tf.summary.scalar('loss', loss)
            else:
                loss = MODEL.get_loss(pred, labels_pl)
                tf.summary.scalar('loss', loss)

            predictions_op = tf.argmax(tf.reduce_mean(pred, axis = -2), axis=-1, name='predictions')
            
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'prediction_op': predictions_op,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
        print('Parameter number: {:d}.'.format(int(parameter_num)))

        start_time = time()
        eval_acc_max = 0
        maxAcc_epoch = 0
        for epoch in range(MAX_EPOCH):
            log_string('\n----------------------------- EPOCH %03d -----------------------------' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            [eval_acc, eval_acc_mean_cls, class_accuracies,_] = eval_one_epoch(sess, ops, test_writer)
            log_string('eval overal acc: %f ---- mean class acc: %f ---- time: %f' % \
                (eval_acc, eval_acc_mean_cls, time() - start_time))

            # Save the variables to disk.
            if eval_acc > eval_acc_max:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                eval_acc_max = eval_acc
                maxAcc_epoch = epoch
                print('best epoch: %f' % (maxAcc_epoch))

                with open(os.path.join(LOG_DIR,"class_accuracies.txt"), 'w') as the_file:
                    the_file.write('best epoch: %f \n' % (maxAcc_epoch))
                    for i, name in enumerate(shape_names):
                        print('%10s:\t%0.3f' % (name, class_accuracies[i]))
                        the_file.write('%10s:\t%0.3f\n' % (name, class_accuracies[i]))
                    the_file.write('%10s:\t%0.3f\n' % ('mean class acc: ', eval_acc_mean_cls))

        train_writer.close()
        test_writer.close()



def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)
    
    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(os.path.join('../data/modelnet40_ply_hdf5_2048/',TRAIN_FILES[train_file_idxs[fn]]))
        current_data = current_data[:,0:NUM_POINT,:]
        current_data, current_label= provider.shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        total_correct = 0
        total_seen = 0
        loss_sum = 0
       
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            # rotation
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])  # z rotation

            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,}
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['prediction_op']], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)

            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val
        
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    eval_start_time = time()  # eval start time
    BATCH_SIZE_val = 4
    for fn in range(len(TEST_FILES)):
        current_data, current_label = provider.loadDataFile(os.path.join('../data/modelnet40_ply_hdf5_2048/',TEST_FILES[fn]))
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE_val
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE_val
            end_idx = (batch_idx+1) * BATCH_SIZE_val

            rotated_data = provider.rotate_point_cloud_so3(current_data[start_idx:end_idx, :, :]) # so3 rotation
            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['prediction_op']], feed_dict=feed_dict)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE_val
            loss_sum += (loss_val*BATCH_SIZE_val)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)

    class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    acc_mean_cls = np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))
    print('eval time: %f' % (time() - eval_start_time))
    return (total_correct / float(total_seen)), acc_mean_cls, class_accuracies, loss_sum / float(total_seen)

if __name__ == "__main__":
    train()
    LOG_FOUT.close()