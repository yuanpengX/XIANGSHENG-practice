# coding: utf-8
# author: Yuanpeng Xiong
# time: 2018-07-03
# time: 2018-07-06

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from config import *
import sys
from utils import *

def SRMD(t_image, is_train=False, reuse=False, C = 1, scale = 2):
    """
    Super resolution with degradtion.
  
    Tensorlayer implementation of the SRMD 

    Main model code
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
 
    with tf.variable_scope("SRMD", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        n = InputLayer(t_image, name='in')
        for i in range(MIDDLE_STACK):
            n = Conv2d(n, MIDDLE_CHANNEL, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n%d/c'%i)
            n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n%d/b'%i)

        # super resolution parts
        n = Conv2d(n, scale*scale*C, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        n = SubpixelConv2d(n, scale=scale, n_out_channel = C, act=relu1, name='pixelshufflerx2/2')  # type: object
        return n



def SRMD_reuse(t_image, is_train=False, reuse=False):
    """
    Super resolution with degradtion.

    Tensorlayer implementation of the SRMD

    Main model code
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("SRMD_reuse", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+    
        n = t_image
        for i in range(MIDDLE_STACK):
            n = Conv2d(n, MIDDLE_CHANNEL, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n%d/c'%i)
            n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n%d/b'%i)
        return n

def SRMD_double(t_image, pre_image, is_train = True, C = 1, scale = 2, is_iir = True):


    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("STACK") as vs:
        pre_response = InputLayer(pre_image, name='preimage')
        pre_response = SRMD_reuse(pre_response, is_train = is_train, reuse = False)
        if is_iir:
            current_response = InputLayer(t_image, name = 'in')
        else:
            current_response = SRMD_reuse(pre_image, is_train = is_train, reuse = True) 
        n = ConcatLayer([pre_response, current_response])
        
        n = Conv2d(n, T + C, (3,3), (1,1), act = lrelu, padding = 'SAME', W_init = w_init, name = 'merge_c')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='merge_bn')

        n = SRMD_reuse(n, is_train = is_train, reuse = True)
        n = Conv2d(n, scale * scale * C, (3,3), (1,1), act = None, padding = 'SAME', W_init = w_init, name = 'n/c')

        n = SubpixelConv2d(n, scale=scale, n_out_channel = C, act=relu1, name='pixelshufflerx2/2')  # type: object


        return n


