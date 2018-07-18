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

import keras
#from keras.layers import *

def keras_train(dataloader, model):
    
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    
    #t_image = Input((HEIGHT, WIDTH, C + T))
    #pre_image = Input((HEIGHT, WIDTH, C + T))
    #pre_pre_image = Input((HEIGHT, WIDTH, C + T))

    train_generator = dataloader.train_generator
    valid_generator = dataloader.valid_generator
    
    # callbacks
    ckp = keras.callbacks.ModelCheckpoint(checkpoint_dir + '/{}_{}.ckpt'.format(tl.global_flag['model'], tl.global_flag['name']), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    estp = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
    
    # n_steps
    train_n_steps = int(dataloader.get_capacity() / BATCH_SIZE)
    valid_n_steps = int(dataloader.get_valid_size() / BATCH_SIZE)
    
    #model = RecurrentModel(t_image, pre_image, pre_pre_image)
    
    model.fit_generator(train_generator(), steps_per_epoch = train_n_steps, epochs = MAX_EPOCH, validation_data = valid_generator(), validation_steps = valid_n_steps, callbacks = [ckp, estp])
    
def train(dataloader):
    warnings.filterwarnings("ignore")
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    valid_dir = "samples/{}_{}_{}".format(tl.global_flag['mode'], tl.global_flag['model'], tl.global_flag['name'])
    tl.files.exists_or_mkdir(valid_dir)

    sess = tf.InteractiveSession()
    # prepare network for training
    lr_image = tf.placeholder('float32', [None, HEIGHT, WIDTH, C + T], name='LR_image')
    hr_image = tf.placeholder('float32', [None, HEIGHT * SCALE, WIDTH * SCALE, C], name='HR_image')
    if TWO_FRAME:

        pre_image = tf.placeholder('float32', [None, HEIGHT, WIDTH, C + T], name='PRE_image')
        srmd = SRMD_double(lr_image, pre_image, is_train = True, is_iir = True, scale = SCALE, C = C)
    else:
        if tl.global_flag['model'] == 'srmd':
            srmd = SRMD(lr_image, is_train=True, reuse=False, scale=SCALE, C=C)
            
        elif tl.global_flag['model'] == 'edsr':
        
            srmd = EDSR(lr_image, is_train=True, reuse=False, scale=SCALE, C=C)
        elif tl.global_flag['model'] == 'edsr_reuse':
        
            srmd = EDSR_reuse(lr_image, is_train=True, reuse=False, scale=SCALE, C=C)
            
        elif tl.global_flag['model'].startswith('lstm'):
            pre_image = tf.placeholder('float32', [None, HEIGHT, WIDTH, C + T], name='PRE_image')
            pre_pre_image = tf.placeholder('float32', [None, HEIGHT, WIDTH, C + T], name='PRE_PRE_image')
            #srmd = RecurrentModel(lr_image, pre_image, pre_pre_image, is_train=True, reuse=False)
            if tl.global_flag['model'].endswith('bn'):
                srmd = RecurrentModel_bn(lr_image, pre_image, pre_pre_image, is_train=True, reuse=False)
            elif tl.global_flag['model'].endswith('bn_relu'):
                srmd = RecurrentModel_bn_relu(lr_image, pre_image, pre_pre_image, is_train=True, reuse=False) 
            else:
                #print('FIXME')
                srmd = RecurrentModel(lr_image, pre_image, pre_pre_image, is_train=True, reuse=False)
        elif tl.global_flag['model'] == 'srmd_bin':
            srmd = SRMD_binary(lr_image, is_train=True, reuse=False, scale=SCALE, C=C)
        else:
            print('Model %s not implemented!'%tl.global_flag['model'])
    #srmd.print_params(False)
    srmd.print_params(False)
    srmd.print_layers()
    # network optimizer
    if MSE_LOSS:
        cost = tl.cost.mean_squared_error(srmd.outputs, hr_image, is_mean=True)
    else:
        cost = tl.cost.absolute_difference_error(srmd.outputs, hr_image, is_mean=True)

    rmse = tf.sqrt(tl.cost.mean_squared_error(255 * srmd.outputs, 255 * hr_image, is_mean=True))

    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=srmd.all_params)

    train_params = srmd.all_params
    # first initialize op parameter
    tl.layers.initialize_global_variables(sess)
    # load network parameter if existed
    if tl.files.file_exists(checkpoint_dir + '/%s_%s.npz' % (tl.global_flag['model'], tl.global_flag['name'])):
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/%s_%s.npz' % (tl.global_flag['model'], tl.global_flag['name']), network=srmd)
        print('Got model pretrained!!!\n')
    else:
        print('No model pretrained!')
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
            elif tl.global_flag['model'].startswith('lstm'):
                lr,pre,prepre,hr = dataloader.get_next_train_batch()
                out = sess.run([cost, train_op, rmse], {lr_image: lr, hr_image: hr, pre_image:pre, pre_pre_image:prepre})
            else:            
                lr, hr = dataloader.get_next_train_batch()
                out = sess.run([cost, train_op, rmse], {lr_image: lr, hr_image: hr})
        
            train_loss += out[0]
            RMSE += out[2]
            percent = (idx + 1) * 50 / iter
            num_arrow = int(percent)
            num_line = 50 - num_arrow
            progress_bar = '[' + '>' * num_arrow + '-' * num_line + ']' + '%.2f%% ' % (percent*2) + ' %.6e' % out[0] + '\r'
            sys.stdout.write(progress_bar)
            sys.stdout.flush()

        print("\nEpoch %d training loss is %.6e, RMSE is %.2f\n" % (epoch + 1, train_loss / iter, RMSE / iter))

        # evaluate model every epoch
        if epoch % TEST_PER_EPOCH == 0:
            max_val = int(dataloader.get_valid_size()/BATCH_SIZE)
            VAL = 0
            V_MSE = 0
            for valiter in range(max_val):
                if TWO_FRAME:
                    lr, pre, hr = dataloader.get_next_valid_batch()

                    valid_cost = sess.run([cost, srmd.outputs, rmse], {lr_image: lr, hr_image: hr, pre_image:pre})
                    
                elif tl.global_flag['model'].startswith('lstm'):
                    lr, pre, prepre, hr = dataloader.get_next_valid_batch()

                    valid_cost = sess.run([cost, srmd.outputs, rmse], {lr_image: lr, hr_image: hr, pre_image:pre, pre_pre_image:prepre})
                else:
                    lr, hr = dataloader.get_next_valid_batch()
        
                    valid_cost = sess.run([cost, srmd.outputs, rmse], {lr_image: lr, hr_image: hr})
                VAL += valid_cost[0]
                V_MSE += valid_cost[2]
            print('\nValidatation loss is %.6e, RMSE is %.2f\n' % (VAL / max_val, V_MSE / max_val))

            save_image(lr, hr, valid_cost[1], valid_dir, epoch)
            
            # Avoid overfitting
            if VAL / max_val > PRE_VALID:
                pass
            else:
                tl.files.save_npz(srmd.all_params, name=checkpoint_dir + '/%s_%s.npz' % (tl.global_flag['model'], tl.global_flag['name']), sess=sess)
                PRE_VALID = VAL / max_val
                  
def evaluate(dataloader, is_patch = False):
    checkpoint_dir = "checkpoint"

    save_dir = "samples/{}_{}_{}".format(tl.global_flag['mode'], tl.global_flag['model'], tl.global_flag['name'])
    tl.files.exists_or_mkdir(save_dir)

    lr_image = tf.placeholder('float32', [None, HEIGHT, WIDTH, C + T], name='LR_image')
    hr_image = tf.placeholder('float32', [None, HEIGHT * SCALE, WIDTH * SCALE, C], name='HR_image')
    if TWO_FRAME:
        pre_image = tf.placeholder('float32', [None, HEIGHT, WIDTH, C + T], name='PRE_image')
        srmd = SRMD_double(lr_image, pre_image, is_train = False, is_iir = True, scale = SCALE, C = C)
    else:
        if tl.global_flag['model'] == 'srmd':
            srmd = SRMD(lr_image, is_train=False, reuse=False, scale=SCALE, C=C)
        elif tl.global_flag['model'] == 'srmd_bin':
            srmd = SRMD_binary(lr_image, is_train=False, reuse=False, scale=SCALE, C=C)
        elif tl.global_flag['model'] == 'edsr':
            srmd = EDSR(lr_image, is_train=False, reuse=False, scale=SCALE, C=C)
        elif tl.global_flag['model'] == 'edsr_reuse':
            srmd = EDSR_reuse(lr_image, is_train=False, reuse=False, scale=SCALE, C=C)
        elif tl.global_flag['model'].startswith('lstm'):
            pre_image = tf.placeholder('float32', [None, HEIGHT, WIDTH, C + T], name='PRE_image')
            pre_pre_image = tf.placeholder('float32', [None, HEIGHT, WIDTH, C + T], name='PRE_PRE_image')
            if tl.global_flag['model'].endswith('bn'):
                srmd = RecurrentModel_bn(lr_image, pre_image, pre_pre_image, is_train=True, reuse=False)
            elif tl.global_flag['model'].endswith('bn_relu'):
                srmd = RecurrentModel_bn_relu(lr_image, pre_image, pre_pre_image, is_train=True, reuse=False) 
            else:
                #print('FIXME')
                srmd = RecurrentModel(lr_image, pre_image, pre_pre_image, is_train=True, reuse=False)
        else:
            print('Model %s not implemented!'%tl.global_flag['model'])
    sess = tf.InteractiveSession()

    rmse = tf.sqrt(tl.cost.mean_squared_error(255 * srmd.outputs, 255 * hr_image, is_mean=True))
    #tl.layers.initialize_global_variables(sess)
    if not tl.files.file_exists(checkpoint_dir + '/%s_%s.npz' % (tl.global_flag['model'], tl.global_flag['name'])):
        print('No model trained!!!')
        exit()
    else:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/%s_%s.npz' % (tl.global_flag['model'], tl.global_flag['name']), network=srmd)
        print('Got model pretrained!')
    count = 0
    RMSE = 0
    for idx in range(int(dataloader.get_test_size(is_patch) / BATCH_SIZE)):
        if TWO_FRAME:
            lr, pre, hr = dataloader.get_next_test_batch(is_patch)

            out = sess.run([srmd.outputs, rmse], {lr_image: lr, hr_image: hr, pre_image:pre})
        elif tl.global_flag['model'].startswith('lstm'):
            lr, pre, prepre, hr = dataloader.get_next_test_batch(is_patch)

            out = sess.run([srmd.outputs, rmse], {lr_image: lr, hr_image: hr, pre_image:pre, pre_pre_image:prepre})
        else:
            lr, hr = dataloader.get_next_test_batch(is_patch)

            out = sess.run([srmd.outputs, rmse], {lr_image: lr, hr_image: hr})

        #lr, hr = dataloader.get_next_test_batch()
        #out = sess.run([srmd.outputs, rmse], {lr_image: lr, hr_image: hr})
        save_image(lr, hr, out[0], save_dir, idx, is_patch)
        RMSE += out[1]
        count += 1
    print('RMSE of the evaluation is : ', RMSE / count)
   
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='tl_train', help='tl_train, tl_evaluate_patch, tl_evaluate')
    parser.add_argument('--name', type=str, default='srmd', help='srmd_CHANNEL_STACK, anything you want')
    parser.add_argument('--model', type=str, default='srmd', help='srmd, edsr, lstm, etc...')
    args = parser.parse_args()
    tl.global_flag['mode'] = args.mode
    tl.global_flag['name'] = args.name
    tl.global_flag['model'] = args.model
    
    
    if tl.global_flag['mode'] == 'tl_train':
        if tl.global_flag['model'].startswith('lstm'):
            dataloader = DataLoader(one_frame = False)  
        else:
            dataloader = DataLoader()  
        train(dataloader)
    elif tl.global_flag['mode'] == 'tl_evaluate':
        dataloader = DataLoader()
        evaluate(dataloader)
    elif tl.global_flag['mode'] == 'tl_evaluate_patch':
        if tl.global_flag['model'].startswith('lstm'):
            dataloader = DataLoader(one_frame = False)  
        else:
            dataloader = DataLoader()        
        evaluate(dataloader, is_patch = True)
            
    elif tl.global_flag['mode'] == 'keras_train':
        if tl.global_flag['model'] == 'srmd':
            dataloader = DataLoader(one_frame = True)
            model = SRMD_keras()
        elif tl.global_flag['model'].startswith('lstm'):
            dataloader = DataLoader(one_frame = False)
            model = RecurrentModel_keras()
        elif tl.global_flag['model'] == 'edsr':
            dataloader = DataLoader(one_frame = True)
            model = EDSR_keras()
        else:
            raise Exception("Unknow --model")
        keras_train(dataloader, model)    
    else:
        raise Exception("Unknow --mode")
