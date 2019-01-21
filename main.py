from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import json
import sys
import os
# import dataset_input_pipline
# import time
import JointNet
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
# sys.setrecursionlimit(100000)
parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default='./SeBi_bonesym_rp2_models/',
                    help='The directory where the model will be stored.')
parser.add_argument('--train_epochs', type=int, default=300,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=1,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--batch_size', type=int, default=64,
                    help='The number of images per batch.')

Joint_num = 17 # joints
n_steps = 60 # frame
n_channels = 20
n_hidden = 128 # hidden layer num of features
n_classes = 10
_WEIGHT_DECAY = 0
# log_dir = './tensorboard/NUCLA/JointNet'
Num_samples = {'train': 10500, 'test': 2800}
data_mean = {'train': None, 'test': None}
data_std = {'train': None, 'test': None}

def loadJasondata(filepath):
    assert os.path.exists(filepath), (
        'Can not find data at given directory!!')
    with open(filepath) as f:
        data = json.load(f)
    return data

def get_truelabel(labelpath):
  """Returns a list of filenames."""
  labels = []
  with open(labelpath) as f:
    for line in f:
        labels.append(int(line.strip()))
  f.close()
  return labels

def Norm(pose_array):
    Norm_pos = np.zeros([pose_array.shape[0], 17, 3])
    mean_pos = np.zeros([3])
    std_pos = np.zeros([3])
    mean_pos[0] = np.mean(pose_array[:, :, 0])
    mean_pos[1] = np.mean(pose_array[:, :, 1])
    mean_pos[2] = np.mean(pose_array[:, :, 2])
    std_pos[0] = np.std(pose_array[:, :, 0])
    std_pos[1] = np.std(pose_array[:, :, 1])
    std_pos[2] = np.std(pose_array[:, :, 2])
    for i in range(pose_array.shape[0]):
        for j in range(17):
            Norm_pos[i, j, 0] = (pose_array[i, j, 0] - mean_pos[0])/std_pos[0]
            Norm_pos[i, j, 1] = (pose_array[i, j, 1] - mean_pos[1])/std_pos[1]
            Norm_pos[i, j, 2] = (pose_array[i, j, 2] - mean_pos[2])/std_pos[2]
    return Norm_pos, mean_pos, std_pos

def Norm2(train_pose, mean, stddev):
    Norm_pos = np.zeros([train_pose.shape[0], 17, 3])
    for i in range(train_pose.shape[0]):
        for j in range(17):
            Norm_pos[i, j, 0] = (train_pose[i, j, 0] - mean[0])/stddev[0]
            Norm_pos[i, j, 1] = (train_pose[i, j, 1] - mean[1])/stddev[1]
            Norm_pos[i, j, 2] = (train_pose[i, j, 2] - mean[2])/stddev[2]
    return Norm_pos

def DeNorm(normpos, is_training):
    """The input pos must be normalized position."""
    if is_training == True:
        if data_mean['train'].any() == None:
            raise ValueError("Pls. calculate the mean position of training data first!")
            return
        if data_std['train'].any() == None:
            raise ValueError("Pls. calculate the standard deviation of training data first!")
            return
        denormPos = normpos * data_std['train'] + data_mean['train']
    else:
        if data_mean['test'].any() == None:
            raise ValueError("Pls. calculate the mean position of testing data first!")
            return
        if data_std['test'].any() == None:
            raise ValueError("Pls. calculate the standard deviation of testing data first!")
            return
        denormPos = normpos * data_std['test'] + data_mean['test']
    return denormPos

def input_fn(is_training, num_epochs, batch_size):
    APE_train_load = np.reshape(np.asarray(loadJasondata('./APEdataset/json_file/train/APE_train.json')), [-1, 17, 3])
    APE_train_gt_load = np.reshape(np.asarray(loadJasondata('./APEdataset/json_file/train/APE_train_gt.json')), [-1, 17, 3])
    APE_train_label = loadJasondata('./APEdataset/json_file/train/APE_drop_train_label.json')

    APE_train_gt, gt_train_mean, gt_train_stddev = Norm(APE_train_gt_load)
    data_mean['train'] = gt_train_mean
    data_std['train'] = gt_train_stddev
    APE_train = Norm2(APE_train_load, gt_train_mean, gt_train_stddev)

    APE_test_load = np.reshape(np.asarray(loadJasondata('./APEdataset/json_file/test/APE_test.json')), [-1, 17, 3])
    APE_test_gt_load = np.reshape(np.asarray(loadJasondata('./APEdataset/json_file/test/APE_test_gt.json')), [-1, 17, 3])
    APE_test_label = loadJasondata('./APEdataset/json_file/test/APE_drop_test_label.json')

    APE_test_gt, gt_test_mean, gt_test_stddev = Norm(APE_test_gt_load)
    data_mean['test'] = gt_train_mean
    data_std['test'] = gt_train_stddev
    APE_test = Norm2(APE_test_load, gt_test_mean, gt_test_stddev)

    if is_training == True:
        pos = APE_train
        position = tf.transpose(pos, [0, 2, 1])
        ground_truth = APE_train_gt
        pos_gt = tf.transpose(ground_truth, [0, 2, 1])
        dropout_labels = tf.reshape(APE_train_label, [-1])
    else:
        pos = APE_test
        position = tf.transpose(pos, [0, 2, 1])
        ground_truth = APE_test_gt
        pos_gt = tf.transpose(ground_truth, [0, 2, 1])
        dropout_labels = tf.reshape(APE_test_label, [-1])

    dataset = tf.data.Dataset.from_tensor_slices((position, pos_gt, dropout_labels))
    if is_training:
        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes have better performance. Because the scores
        # is a relatively small dataset, we choose to shuffle the full epoch.
        dataset = dataset.shuffle(Num_samples['train'])
    else:
        dataset = dataset.shuffle(Num_samples['test'])
    dataset = dataset.prefetch(2 * batch_size)
    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    # Batch results by up to batch_size, and then fetch the tuple from the
    # iterator.
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    # features, gtpos, dropout_labels = iterator.get_next()
    features, gtpos, dplabels = iterator.get_next()
    labels = [gtpos, dplabels]
    # print(labels[0])
    # print(labels)
    return features, labels

def findParentJoint(joint):
    if joint == 0:
        return 0
    elif joint == 16:
        return 1
    elif joint == 2:
        return 16
    elif joint == 4:
        return 16
    elif joint == 7:
        return 16
    elif joint == 10:
        return 0
    elif joint == 13:
        return 0
    else:
        return joint-1

def Cal_Bonelengths(pos):
    # for 17 joints structure
    # input pos must be a tensor with a shape of jointNum * batchsize * 3
    Bone_lens = [None] * (Joint_num-1)
    for i in range(Joint_num-1):
        Bone_lens[i] = tf.norm(pos[i+1] - pos[findParentJoint(i+1)], axis=1)
    return Bone_lens

def model_fn(features, labels, mode, params):
    pos = features
    pos_gt = tf.cast(labels[0], dtype=tf.float32)
    pos_gt = tf.unstack(pos_gt, Joint_num, 2)
    # stop calculating the gradient about ground truth position
    pos_gt = tf.stop_gradient(pos_gt, name="gtpos_stop_gradient")
    dp_labels = labels[1]

    pred_pos = JointNet.SingleRecursivemodel(pos, params['batch_size'], n_hidden, Joint_num)


    predictions = {
        'prediction_pos': tf.transpose(tf.convert_to_tensor(pred_pos), [1, 0, 2]),
        'dropLabel': dp_labels
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    posloss = 0
    for i in range(Joint_num):
        regression_error = tf.nn.l2_loss(tf.subtract(pred_pos[i], pos_gt[i])) #l2_loss is (x^2 + y^2 + z^2)/2
        # print("regression_error:", regression_error)
        posloss = posloss + regression_error
    tf.identity(posloss, name='position_loss')
    tf.summary.scalar('position_loss', posloss)

    # calculating bone length loss
    pred_boneLens = tf.convert_to_tensor(Cal_Bonelengths(pred_pos))
    gt_boneLens = tf.convert_to_tensor(Cal_Bonelengths(pos_gt))
    boneloss = tf.nn.l2_loss(tf.subtract(pred_boneLens, gt_boneLens))
    tf.identity(boneloss, name='bonelength_loss')
    tf.summary.scalar('bonelength_loss', boneloss)
    # calculating bone symmetry loss by considering the length of left limbs are equal to length of right limbs
    left = [4, 5, 6, 10, 11, 12]
    right = [7, 8, 9, 13, 14, 15]
    symmetryloss = 0
    for i in range(len(left)):
        symmetryloss = symmetryloss + tf.nn.l2_loss(tf.subtract(pred_boneLens[left[i]], pred_boneLens[right[i]]))
    tf.identity(symmetryloss, name='symmetry_loss')
    tf.summary.scalar('symmetry_loss', symmetryloss)

    # Define loss and optimizer
    loss_op = posloss + 5 * boneloss + 10 * symmetryloss + _WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    if mode == tf.estimator.ModeKeys.TRAIN:
        initial_learning_rate = 0.0002
        global_step = tf.train.get_or_create_global_step()
        batches_per_epoch = Num_samples['train'] / params['batch_size']
        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        boundaries = [int(batches_per_epoch * epoch) for epoch in [180, 240, 300]]
        values = [initial_learning_rate * decay for decay in [1, 0.5, 0.1, 0.05]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values)
        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op, global_step)

    else:
        train_op = None

    MPJPE = 0
    for i in range(Joint_num):
        pred_pos_denorm = DeNorm(pred_pos[i], is_training=(mode == tf.estimator.ModeKeys.TRAIN))
        gt_pos_denorm = DeNorm(pos_gt[i], is_training=(mode == tf.estimator.ModeKeys.TRAIN))
        regression_error_denorm = tf.norm(tf.subtract(pred_pos_denorm, gt_pos_denorm), axis=1)
        MPJPE = MPJPE + regression_error_denorm

    MPJPE = tf.metrics.mean(MPJPE)
    metrics = {'accuracy': MPJPE}
    tf.identity(MPJPE[1], name='train_MPJPE')
    tf.summary.scalar('train_MPJPE', MPJPE[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops=metrics)

def main(unused_argv):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
    motionrcg_classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={'batch_size': FLAGS.batch_size})

    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        tensors_to_log = {
            'learning_rate': 'learning_rate',
            'position_loss': 'position_loss',
            'bonelength_loss': 'bonelength_loss',
            'train_MPJPE': 'train_MPJPE',
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

        motionrcg_classifier.train(
            input_fn=lambda: input_fn(
                True, FLAGS.epochs_per_eval, FLAGS.batch_size),
            hooks=[logging_hook])

        # Evaluate the model and print results
        eval_results = motionrcg_classifier.evaluate(
            input_fn=lambda: input_fn(False, FLAGS.epochs_per_eval, FLAGS.batch_size))
        # best_ckpt_saver.handle(eval_results["metric"], sess, global_step_tensor)
        print(eval_results)


if __name__ == '__main__':
    # main()
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)