# coding: utf-8
# author: Yuanpeng Xiong
# time: 2018-07-04

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from imageIO import DataLoader
from config import *
import sys
from utils import *
from model import *
import warnings
warnings.filterwarnings("ignore")


def train(dataloader):

    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    valid_dir = "samples/{}_{}".format(tl.global_flag['mode'], tl.global_flag['name'])
    tl.files.exists_or_mkdir(valid_dir)

    sess = tf.InteractiveSession()
    # prepare network for training
    lr_image = tf.placeholder('float32', [None, HEIGHT, WIDTH, C + T], name='LR_image')
    hr_image = tf.placeholder('float32', [None, HEIGHT * SCALE, WIDTH * SCALE, C], name='HR_image')
    if TWO_FRAME:

        pre_image = tf.placeholder('float32', [None, HEIGHT, WIDTH, C + T], name='PRE_image')
        srmd = SRMD_double(lr_image, pre_image, is_train = True, is_iir = True, scale = SCALE, C = C)
    else:

        srmd = SRMD(lr_image, is_train=True, reuse=False, scale=SCALE, C=C)
    srmd.print_params(False)
    srmd.print_layers()

    if tl.files.file_exists(checkpoint_dir + '/%s.npz' % tl.global_flag['name']):
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/%s.npz' % tl.global_flag['name'], network=srmd)
        print('Got model pretrained!!!\n')

    # network optimizer
    cost = tl.cost.mean_squared_error(srmd.outputs, hr_image, is_mean=True)

    rmse = tf.sqrt(tl.cost.mean_squared_error(255 * srmd.outputs, 255 * hr_image, is_mean=True))

    train_params = srmd.all_params

    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_params)

    tl.layers.initialize_global_variables(sess)
    print('Start training')

    PRE_VALID = 100

    for epoch in range(MAX_EPOCH + 1):

        iter = int(dataloader.get_capacity() / BATCH_SIZE)

        train_loss = 0.0
        RMSE = 0.0
        for idx in range(iter):

            if TWO_FRAME:
                lr,pre,hr = dataloader.get_next_train_batch()
                out = sess.run([cost, train_op, rmse], {lr_image: lr, hr_image: hr, pre_image:pre})
            else:
                lr, hr = dataloader.get_next_train_batch()
                out = sess.run([cost, train_op, rmse], {lr_image: lr, hr_image: hr})

            train_loss += out[0]
            RMSE += out[2]
            percent = (idx + 1) * 100 / iter
            num_arrow = int(percent)
            num_line = 100 - num_arrow
            progress_bar = '[' + '>' * num_arrow + '-' * num_line + ']' + '%.2f%%' % percent + ' %.6f' % out[0] + '\r'
            sys.stdout.write(progress_bar)
            sys.stdout.flush()

        print("\nEpoch %d training loss is %.6f, RMSE is %.2f\n" % (epoch + 1, train_loss / iter, RMSE / iter))

        # evaluate model every epoch
        if epoch % TEST_PER_EPOCH == 0:
            if TWO_FRAME:
                lr, pre, hr = dataloader.get_next_valid_batch()

                valid_cost = sess.run([cost, srmd.outputs, rmse], {lr_image: lr, hr_image: hr, pre_image:pre})

            else:
                lr, hr = dataloader.get_next_valid_batch()
    
                valid_cost = sess.run([cost, srmd.outputs, rmse], {lr_image: lr, hr_image: hr})

            print('\nValidatation loss is %.6f, RMSE is %.2f\n' % (valid_cost[0], valid_cost[2]))

            save_image(lr, hr, valid_cost[1], valid_dir, epoch)

            # Avoid overfitting
            if valid_cost[0] > PRE_VALID:
                return
            else:
                tl.files.save_npz(train_params, name=checkpoint_dir + '/%s.npz' % tl.global_flag['name'], sess=sess)
                PRE_VALID = valid_cost[0]

        # MODEL SAVING HERE
        if epoch == 0 or epoch % 10 == 0:
            tl.files.save_npz(train_params, name=checkpoint_dir + '/%s.npz' % tl.global_flag['name'], sess=sess)


def evaluate(dataloader):
    checkpoint_dir = "checkpoint"
    if not tl.files.file_exists(checkpoint_dir + '/%s.npz' % tl.global_flag['name']):
        print('No model trained!!!')

    save_dir = "samples/{}_{}".format(tl.global_flag['mode'], tl.global_flag['name'])
    tl.files.exists_or_mkdir(save_dir)

    lr_image = tf.placeholder('float32', [None, HEIGHT, WIDTH, C + T], name='LR_image')
    hr_image = tf.placeholder('float32', [None, HEIGHT * SCALE, WIDTH * SCALE, C], name='HR_image')
    if TWO_FRAME:
        pre_image = tf.placeholder('float32', [None, HEIGHT, WIDTH, C + T], name='PRE_image')
        srmd = SRMD_double(lr_image, pre_image, is_train = True, is_iir = True, scale = SCALE, C = C)
    else:

        srmd = SRMD(lr_image, is_train=True, reuse=False, scale=SCALE, C=C)

    #srmd = SRMD(lr_image, is_train=False, reuse=False, scale=SCALE, C=C)
    sess = tf.InteractiveSession()

    rmse = tf.sqrt(tl.cost.mean_squared_error(255 * srmd.outputs, 255 * hr_image, is_mean=True))

    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/%s.npz' % tl.global_flag['name'], network=srmd)
    count = 0
    RMSE = 0
    for idx in range(int(dataloader.get_test_size() / BATCH_SIZE)):
        if TWO_FRAME:
            lr, pre, hr = dataloader.get_next_test_batch()

            out = sess.run([srmd.outputs, rmse], {lr_image: lr, hr_image: hr, pre_image:pre})

        else:
            lr, hr = dataloader.get_next_test_batch()

            out = sess.run([srmd.outputs, rmse], {lr_image: lr, hr_image: hr})

        #lr, hr = dataloader.get_next_test_batch()
        #out = sess.run([srmd.outputs, rmse], {lr_image: lr, hr_image: hr})
        save_image(lr, hr, out[0], save_dir, idx)
        RMSE += out[1]
        count += 1
    print('RMSE of the evaluation is : ', RMSE / count)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='train, evaluate')
    parser.add_argument('--name', type=str, default='srmd', help='srmd_CHANNEL_STACK')
    args = parser.parse_args()
    tl.global_flag['mode'] = args.mode
    tl.global_flag['name'] = args.name
    dataloader = DataLoader()
    if tl.global_flag['mode'] == 'train':
        train(dataloader)
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate(dataloader)
        pass
    else:
        raise Exception("Unknow --mode")
