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

# To utilizer keras interfaces
import keras
from keras.layers import Conv2D,Input,Lambda, Add, ConvLSTM2D
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import initializers

def SRMD(t_image, is_train=True, reuse=False, C = 1, scale = 2):
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
        
def SRMD_binary(t_image, is_train=True, reuse=False, C = 1, scale = 2):
    """
    Super resolution with degradtion.

    Tensorlayer implementation of the SRMD 

    Main model code
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.sign(x)

    with tf.variable_scope("SRMD_binary", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        n = InputLayer(t_image, name='in')
        for i in range(MIDDLE_STACK):
            n = BinaryConv2d(n, MIDDLE_CHANNEL, (3, 3), (1, 1), act=lrelu, padding='SAME', name='n%d/c'%i)
            n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n%d/b'%i)

        # super resolution parts
        n = BinaryConv2d(n, scale*scale*C, (3, 3), (1, 1), act=None, padding='SAME',name='n256s1/2')
        n = SubpixelConv2d(n, scale=scale, n_out_channel = C, act=relu1, name='pixelshufflerx2/2')  # type: object
        print(n.all_params)
        return n
        
def SRMD_keras(is_train=True, reuse=False, C = 1, scale = 2):
    """
    Super resolution with degradtion.
  
    Tensorlayer implementation of the SRMD 

    Main model code
    """
    pre = C + T
    t_image = Input([HEIGHT, WIDTH, C+T])
    
    n = t_image
    for i in range(MIDDLE_STACK):
        #n = Conv2D(MIDDLE_CHANNEL, (3,3), strides = (1, 1), activation = PRELU(),kernel_constraint = OrthogonalRegularizer(cin = pre, filters = MIDDLE_CHANNEL, window = 3), padding = 'SAME', data_format = 'channels_last', kernel_initializer = keras.initializers.Orthogonal(gain=1.0, seed=None))(n)
        n = Conv2D(MIDDLE_CHANNEL, (3,3), strides = (1, 1), activation = PLeakyReLU(alpha=0.2), padding = 'SAME', data_format = 'channels_last', kernel_initializer = initializers.random_normal(stddev=0.02))(n)
        n = BatchNormalization()(n)
        pre = MIDDLE_CHANNEL

    # super resolution parts
    #n = Conv2D(scale * scale * C, (3,3), strides = (1, 1), activation = PRELU(max_value=1),kernel_constraint= OrthogonalRegularizer(cin = MIDDLE_CHANNEL, filters = scale * scale * C, window = 3), padding = 'SAME', data_format = 'channels_last', kernel_initializer = keras.initializers.Orthogonal(gain=1.0, seed=None))(n)
    n = Conv2D(scale * scale * C, (3,3), strides = (1, 1), activation = PRELU(max_value=1), padding = 'SAME', data_format = 'channels_last', kernel_initializer = initializers.random_normal(stddev=0.02))(n)
    #n = SubpixelConv2d(n, scale=scale, n_out_channel = C, act=relu1, name='pixelshufflerx2/2')  # type: object
    subpix  = Lambda(lambda x: tf.depth_to_space(x, SCALE))
    n = subpix(n)
    

    model = keras.models.Model(inputs = t_image, outputs = n)
    
    model.summary()
    
    adam = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    model.compile(loss='mse', optimizer=adam)
    
    return model    

def SRMD_reuse(t_image, is_train=True, reuse=False):
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
            n = Conv2D(MIDDLE_CHANNEL, (3,3), strides = (1, 1), activation = PRELU(), padding = 'SAME', data_format = 'channels_last', kernel_initializer = keras.initializers.Orthogonal(gain=1.0, seed=None))(n)
            n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n%d/b'%i)(n)
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

def RecurrentModel_keras(is_train = True, C = 1, scale = 2):
    '''
    To implement classical convolutional LSTM structure
    '''
    t_image = Input((HEIGHT, WIDTH, C + T))
    pre_image = Input((HEIGHT, WIDTH, C + T))
    pre_pre_image = Input((HEIGHT, WIDTH, C + T))
    share_conv1 = Conv2D(MIDDLE_CHANNEL, (3,3), strides = (1, 1), activation = PRELU(), padding = 'SAME', data_format = 'channels_last')
    share_bn1 = BatchNormalization()
    relu1 =PRELU()
    
    share_conv2 = Conv2D(MIDDLE_CHANNEL, (3,3), strides = (1, 1), activation = PRELU(), padding = 'SAME', data_format = 'channels_last')
    share_bn2 = BatchNormalization()
    relu2 = PRELU()
    
    lstm1 = ConvLSTM2D(MIDDLE_CHANNEL, (3,3), padding = 'SAME', data_format = 'channels_last',recurrent_activation = PRELU(), activation = PRELU(), return_sequences = True)
    lstm2 = ConvLSTM2D(MIDDLE_CHANNEL, (3,3), padding = 'SAME', data_format = 'channels_last', return_sequences = False, recurrent_activation = PRELU(), activation = PRELU())
    #Lambda(lambda x: relu1(x))
    conv = Conv2D(C * scale * scale , (3,3),strides = (1, 1), activation = PRELU(max_value = 1), padding = 'SAME', data_format = 'channels_last')
    
    features1 = relu1(share_bn1(share_conv1(t_image)))
    features1 = relu2(share_bn2(share_conv2(features1)))
    
    features2 = relu1(share_bn1(share_conv1(pre_image)))
    features2 = relu2(share_bn2(share_conv2(features2)))
    
    features3 = relu1(share_bn1(share_conv1(pre_pre_image)))
    features3 = relu2(share_bn2(share_conv2(features3)))
    
    
    stack = Lambda(lambda x:tf.keras.backend.stack(x, axis = 1))
    concat = stack([features1, features2, features3])
   
    recur = lstm1(concat)
    
    recur2 = lstm2(recur)
    
    feature_last = conv(recur2)
    #print(type(feature_last))
    subpix  = Lambda(lambda x: tf.depth_to_space(x, SCALE))
    hr = subpix(feature_last)
    #print(type(hr))
    adam = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model = keras.models.Model(inputs = [t_image, pre_image, pre_pre_image], outputs = hr)
    
    model.summary()
    
    #optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    model.compile(loss='mse', optimizer=adam)
    
    return model
    
def RecurrentModel_reuse(image, is_train=True, reuse=False):
    '''
    This layer is the initial layer for all models
    And this part is just the same as EDSR
    And this is reusable model
    '''
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

    with tf.variable_scope("RES_BLOCK", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+               
        first_conv = Conv2d(image, MIDDLE_CHANNEL, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='conv1')
        pre = first_conv
        for i in range(RES_BLOCK):
            n = Conv2d(pre, MIDDLE_CHANNEL, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n%d/c'%i)
            n = Conv2d(n, MIDDLE_CHANNEL, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='noact%d/c'%i)
            n = ElementwiseLayer([n, pre], combine_fn=tf.add, name='add%d'%i)
            pre = n
        n = Conv2d(n, MIDDLE_CHANNEL, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='conv2')
        n = ElementwiseLayer([n, first_conv], combine_fn=tf.add, name='outadd')
        return n, first_conv
        
def RecurrentModel(t_image,pre_image, pre_pre_image, is_train=True, reuse=False):
    
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

    with tf.variable_scope("RecurrentModel", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        t_input = InputLayer(t_image, name='t')
        pre_input = InputLayer(pre_image, name='pre')
        pre_pre_input = InputLayer(pre_pre_image, name='pre_pre')
        
        t_features, t_firt = RecurrentModel_reuse(t_input, is_train = True, reuse = False)
        pre_features, pre_firts = RecurrentModel_reuse(pre_input, is_train = True, reuse = True)
        pre_pre_features, pre_pre_first = RecurrentModel_reuse(pre_input, is_train = True, reuse = True)
        
        stack = StackLayer([t_features, pre_features, pre_pre_features], axis=1, name='stack')
        
        time_features = ConvLSTMLayer(stack, (32,40), n_steps=2, name = 'lstm1', return_last = False, feature_map = MIDDLE_CHANNEL)
        
        low_res_features = ConvLSTMLayer(time_features, (32,40), n_steps=2, name = 'lstm2', return_last = True, feature_map = MIDDLE_CHANNEL)
        
        low_res_features =  ElementwiseLayer([t_firt, low_res_features], combine_fn=tf.add, name='outadd')
                
        hr = Conv2d(low_res_features, SCALE*SCALE*C, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='lastconv')
        hr = SubpixelConv2d(hr, scale=SCALE, n_out_channel = C, act=relu1, name='pixelshufflerx2/2')  # type: object
        return hr
        
def RecurrentModel_bn_reuse(image, is_train=True, reuse=False):
    '''
    This layer is the initial layer for all models
    And this part is just the same as EDSR
    And this is reusable model
    '''
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

    with tf.variable_scope("RES_BLOCK_bn", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+               
        first_conv = Conv2d(image, MIDDLE_CHANNEL, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='conv1')
        pre = first_conv
        for i in range(RES_BLOCK):
            n = Conv2d(pre, MIDDLE_CHANNEL, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n%d/c'%i)
            #n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n%d/b1'%i)
            n = Conv2d(n, MIDDLE_CHANNEL, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='noact%d/c'%i)
            #n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n%d/b2'%i)
            n = ElementwiseLayer([n, pre], combine_fn=tf.add, name='add%d'%i)
            pre = n
        n = Conv2d(n, MIDDLE_CHANNEL, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='conv2')
        #n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='outbn'%i)
        n = ElementwiseLayer([n, first_conv], combine_fn=tf.add, name='outadd')
        return n, first_conv
        
def RecurrentModel_bn(t_image,pre_image, pre_pre_image, is_train=True, reuse=False):
    
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

    with tf.variable_scope("RecurrentModel_bn", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        t_input = InputLayer(t_image, name='t')
        pre_input = InputLayer(pre_image, name='pre')
        pre_pre_input = InputLayer(pre_pre_image, name='pre_pre')
        
        t_features, t_firt = RecurrentModel_bn_reuse(t_input, is_train = True, reuse = False)
        pre_features, pre_firts = RecurrentModel_bn_reuse(pre_input, is_train = True, reuse = True)
        pre_pre_features, pre_pre_first = RecurrentModel_bn_reuse(pre_input, is_train = True, reuse = True)
        
        stack = StackLayer([t_features, pre_features, pre_pre_features], axis=1, name='stack')
        
        time_features = ConvLSTMLayer(stack, (32,40), n_steps=2, name = 'lstm1', return_last = False, feature_map = MIDDLE_CHANNEL)
        
        low_res_features = ConvLSTMLayer(time_features, (32,40), n_steps=2, name = 'lstm2', return_last = True, feature_map = MIDDLE_CHANNEL)
        
        low_res_features =  ElementwiseLayer([t_firt, low_res_features], combine_fn=tf.add, name='outadd')
                
        hr = Conv2d(low_res_features, SCALE*SCALE*C, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='lastconv')
        hr = SubpixelConv2d(hr, scale=SCALE, n_out_channel = C, act=relu1, name='pixelshufflerx2/2')  # type: object
        return hr

def RecurrentModel_bn_relu_reuse(image, is_train=True, reuse=False):
    '''
    This layer is the initial layer for all models
    And this part is just the same as EDSR
    And this is reusable model
    '''
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

    with tf.variable_scope("RES_BLOCK_bn", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+               
        first_conv = Conv2d(image, MIDDLE_CHANNEL, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='conv1')
        pre = first_conv
        for i in range(RES_BLOCK):
            n = Conv2d(pre, MIDDLE_CHANNEL, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n%d/c'%i)
            n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n%d/b1'%i)
            n = Conv2d(n, MIDDLE_CHANNEL, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='noact%d/c'%i)
            n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n%d/b2'%i)
            n = ElementwiseLayer([n, pre], combine_fn=tf.add, name='add%d'%i)
            pre = n
        n = Conv2d(n, MIDDLE_CHANNEL, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='conv2')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='outbn')
        n = ElementwiseLayer([n, first_conv], combine_fn=tf.add, name='outadd')
        return n, first_conv
        
def RecurrentModel_bn_relu(t_image,pre_image, pre_pre_image, is_train=True, reuse=False):
    
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

    with tf.variable_scope("RecurrentModel_bn", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        t_input = InputLayer(t_image, name='t')
        pre_input = InputLayer(pre_image, name='pre')
        pre_pre_input = InputLayer(pre_pre_image, name='pre_pre')
        
        t_features, t_firt = RecurrentModel_bn_relu_reuse(t_input, is_train = True, reuse = False)
        pre_features, pre_firts = RecurrentModel_bn_relu_reuse(pre_input, is_train = True, reuse = True)
        pre_pre_features, pre_pre_first = RecurrentModel_bn_relu_reuse(pre_input, is_train = True, reuse = True)
        
        stack = StackLayer([t_features, pre_features, pre_pre_features], axis=1, name='stack')
        
        time_features = ConvLSTMLayer(stack, (32,40), n_steps=2, name = 'lstm1', return_last = False, feature_map = MIDDLE_CHANNEL)
        
        low_res_features = ConvLSTMLayer(time_features, (32,40), n_steps=2, name = 'lstm2', return_last = True, feature_map = MIDDLE_CHANNEL)
        
        low_res_features =  ElementwiseLayer([t_firt, low_res_features], combine_fn=tf.add, name='outadd')
                
        hr = Conv2d(low_res_features, SCALE*SCALE*C, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='lastconv')
        hr = SubpixelConv2d(hr, scale=SCALE, n_out_channel = C, act=relu1, name='pixelshufflerx2/2')  # type: object
        return hr
  
def EDSR(t_image, is_train=True, reuse=False, C = 1, scale = 2):
    """
    Super resolution with degradtion.

    Tensorlayer implementation of the SRMD 

    Main model code
    """
    #MIDDLE_CHANNEL = 128
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

    with tf.variable_scope("EDSR", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        IN = InputLayer(t_image, name='in')
        first_conv = Conv2d(IN, MIDDLE_CHANNEL, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='conv1')
        pre = first_conv
        for i in range(RES_BLOCK):
            n = Conv2d(pre, MIDDLE_CHANNEL, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n%d/c'%i)
            n = Conv2d(n, MIDDLE_CHANNEL, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='noact%d/c'%i)
            n = ElementwiseLayer([n, pre], combine_fn=tf.add, name='n%d/add'%i)
            pre = n
        # super resolution parts
        n = Conv2d(n, MIDDLE_CHANNEL, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='conv2')
        n = ElementwiseLayer([n, first_conv], combine_fn=tf.add, name='outadd')
        n = Conv2d(n, scale*scale*C, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='conv3')
        n = SubpixelConv2d(n, scale=scale, n_out_channel = C, act=relu1, name='pixelshufflerx1')  # type: object
        n = Conv2d(n, 1, (3, 3), (1, 1), act=relu1, padding='SAME', W_init=w_init, name='conv4')
        return n

def EDSR_reuse(t_image, is_train=True, reuse=False, C = 1, scale = 2):
    """
    Super resolution with degradtion.

    Tensorlayer implementation of the SRMD 

    Main model code
    """
    #MIDDLE_CHANNEL = 128
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

    with tf.variable_scope("EDSR_reuse", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        IN = InputLayer(t_image, name='in')
        first_conv = Conv2d(IN, MIDDLE_CHANNEL, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='conv1')
        pre = first_conv
        for i in range(RES_BLOCK):
            n = Conv2d(pre, MIDDLE_CHANNEL, (3, 3), (1, 1), act=lrelu, padding='SAME', W_init=w_init, name='n%d/c'%i)
            n = Conv2d(n, MIDDLE_CHANNEL, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n%d/c'%i)
            n = ElementwiseLayer([n, pre], combine_fn=tf.add, name='fdadd')
            pre = n
        # super resolution parts
        n = Conv2d(n, MIDDLE_CHANNEL, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='conv2')
        n = ElementwiseLayer([n, first_conv], combine_fn=tf.add, name='fdadd')
        n = Conv2d(n, scale*scale*C, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='conv3')
        n = SubpixelConv2d(n, scale=scale, n_out_channel = C, act=relu1, name='pixelshufflerx1')  # type: object
        n = Conv2d(n, 1, (3, 3), (1, 1), act=relu1, padding='SAME', W_init=w_init, name='conv4')
        print(len(n.all_params))
        return n
           
def EDSR_keras():
    
    t_image = Input([HEIGHT, WIDTH, C+T])
    
    #MY_CHANNEL = 256#MIDDLE_CHANNEL * 2
    
    after_conv = Conv2D(MY_CHANNEL, (3,3), strides = (1, 1), activation = PLeakyReLU(alpha=0.2), padding = 'SAME', data_format = 'channels_last')(t_image)
    
    pre = after_conv
    
    # Res block
    for i in range(RES_BLOCK):
        
        next = Conv2D(MY_CHANNEL, (3,3), strides = (1, 1), activation = PLeakyReLU(alpha=0.2), kernel_initializer=initializers.random_normal(stddev=0.02), padding = 'SAME', data_format = 'channels_last')(pre)
        next = Conv2D(MY_CHANNEL, (3,3), strides = (1, 1), activation = PLeakyReLU(alpha=0.2),kernel_initializer=initializers.random_normal(stddev=0.02), padding = 'SAME', data_format = 'channels_last')(next)
        pre = Add()([pre ,next])
    
    pre = Conv2D(MY_CHANNEL, (3,3), strides = (1, 1), activation = PLeakyReLU(alpha=0.2),kernel_initializer=initializers.random_normal(stddev=0.02), padding = 'SAME', data_format = 'channels_last')(pre)
    pre = Add()([pre ,after_conv])
    pre = Conv2D(C* SCALE * SCALE, (3,3), strides = (1, 1), activation = PLeakyReLU(alpha=0.2),kernel_initializer=initializers.random_normal(stddev=0.02), padding = 'SAME', data_format = 'channels_last')(pre)
    
    subpix  = Lambda(lambda x: tf.depth_to_space(x, SCALE))
    
    pre = subpix(pre)
    
    pre = Conv2D(1, (3,3), strides = (1, 1), activation = PRELU(max_value = 1), kernel_initializer=initializers.random_normal(stddev=0.02),padding = 'SAME', data_format = 'channels_last')(pre)
    adam = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    model = keras.models.Model(inputs = t_image, outputs = pre)
    
    model.summary()
    
    #optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    model.compile(loss='mae', optimizer=adam)
    
    return model
