# coding: utf-8
# author: Yuanpeng Xiong
# time: 2018-07-04

import numpy as np
import os
import h5py as matio
from PIL import Image
from random import randint
from config import *
from datetime import datetime
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from config import *
import sys
import warnings
from keras.layers import ReLU
from keras.constraints import Constraint
from keras.layers.advanced_activations import LeakyReLU
warnings.filterwarnings("ignore")

from tensorlayer.layers.recurrent import *


class OrthogonalRegularizer(Constraint):
    """MaxNorm weight constraint.
    Constrains the weights incident to each hidden unit
    to have a norm less than or equal to a desired value.
    # Arguments
        m: the maximum norm for the incoming weights.
        axis: integer, axis along which to calculate weight norms.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    """

    def __init__(self, update_per = 20, filters = 128, window = 3, cin = 10):
        self.update_per = update_per
        self.count = 0
        self.window = window
        self.filters = filters 
        self.cin = cin
        
    def __call__(self, w):
        filters = self.filters
        window = self.window
        cin = self.cin
        self.count +=1
        if self.count % self.update_per == 0:
            w_re = tf.reshape(w, [window * window * cin, filters])
            s, u, v = tf.linalg.svd(w)
            w_approx =  tf.matmul(u, tf.matmul(tf.linalg.diag(np.ones(filters)), v, adjoint_b=True))
            w = tf.reshape(w_approx, [window, window, cin, filters])
            return w
        else:
            return w
        return w

    def get_config(self):
        return {'update_per': self.update_per
                }
                         
class PRELU(ReLU):
    def __init__(self, **kwargs):
        self.__name__ = "PRELU"
        super(PRELU, self).__init__(**kwargs)

class PLeakyReLU(LeakyReLU):
    def __init__(self, **kwargs):
        self.__name__ = "PLeakyReLU"
        super(PLeakyReLU, self).__init__(**kwargs)
       
def orthogonal_regularizer(w):
    
    '''
    My implementation of orthogonal_regularizer
    
    https://github.com/ajbrock/Neural-Photo-Editor/blob/master/train_IAN_simple.py#L279
    
    '''
    w_re = tf.reshape(w, [-1, w.shape[-1]])
    
    w_mat = tf.matmul(tf.matrix_transpose(w_re),w_re)
    
    r = w_mat - tf.eye(w.shape[-1])
    
    return tf.norm(t)
       
def subpix(x):
    return tf.depth_to_space(x[0], x[1])

def relu1(x, name='relu1'):
    with tf.name_scope(name, "relu6") as name_scope:
        x = tf.convert_to_tensor(x, name="features")
        return tf.minimum(tf.maximum(x, 0), 1, name=name_scope)

def save_image(lr, hr, recover, directory, epoch, is_patch = False):
    '''
    Save batch image to test
    '''
    if is_patch:
        '''
        is_path is True, then the whole batch will be a part of image
        '''
        save_dir = directory + '/test/'
        tl.files.exists_or_mkdir(save_dir)
        lr_full = np.ones([HEIGHT*PATCH_SCALE_X, WIDTH*PATCH_SCALE_Y,1])
        hr_full = np.ones([HEIGHT*PATCH_SCALE_X* SCALE, WIDTH*PATCH_SCALE_Y*SCALE,1])
        rec_full = np.ones([HEIGHT*PATCH_SCALE_X*SCALE, WIDTH*PATCH_SCALE_Y*SCALE,1])
        
        for i in range(PATCH_SCALE_X):
            for j in range(PATCH_SCALE_Y):
            
                lr_full[i*HEIGHT:i*HEIGHT+HEIGHT,j*WIDTH:j*WIDTH+WIDTH,0] = lr[i*PATCH_SCALE_Y + j,:,:,0].reshape(HEIGHT,WIDTH)
                hr_full[i*HEIGHT*SCALE:i*HEIGHT*SCALE+HEIGHT*SCALE,j*WIDTH*SCALE:j*WIDTH*SCALE+WIDTH*SCALE,0] = hr[i*PATCH_SCALE_Y + j].reshape(HEIGHT*SCALE,WIDTH*SCALE)
                rec_full[i*HEIGHT*SCALE:i*HEIGHT*SCALE+HEIGHT*SCALE,j*WIDTH*SCALE:j*WIDTH*SCALE+WIDTH*SCALE,0] = recover[i*PATCH_SCALE_Y + j].reshape(HEIGHT*SCALE, WIDTH*SCALE)
        tl.vis.save_image(lr_full, save_dir + '/%d_lr.bmp'%epoch)
        tl.vis.save_image(hr_full, save_dir + '/%d_hr.bmp'%epoch)
        tl.vis.save_image(rec_full, save_dir + '/%d_hr_rec.bmp'%epoch)
    else:
        save_dir = directory + '/test' + str(epoch)
        tl.files.exists_or_mkdir(save_dir)
        for i in range(BATCH_SIZE):
            tl.vis.save_image(lr[i], save_dir + '/%d_lr.bmp' % i)
            tl.vis.save_image(hr[i], save_dir + '/%d_hr.bmp' % i)
            tl.vis.save_image(recover[i], save_dir + '/%d_hr_rec.bmp' % i)

def load_mat(filename):
    '''
    SRMD author provides a PCA vector, and other kernels
    These kernels will be used to generate Degrade channel
    '''

    dic = matio.File(filename)

    P = np.array(dic['net/meta/P']).T
    print('Shape of PCA is ', P.shape)

    AtrpGaussianKernel = np.array(dic['net/meta/AtrpGaussianKernel']).T
    print('Shape of AtrpGaussianKernel is', AtrpGaussianKernel.shape)

    directKernel = np.array(dic['net/meta/directKernel']).T
    print('Shape of directKernel is', directKernel.shape)

    return P, AtrpGaussianKernel, directKernel

def construct_degration(P, kernel, noise_level=25):
    '''
    This function is used to construct degration channels
    '''
    if USE_NOISE:
        if DEBLUR:  
            projected = list(np.dot(P, kernel)) + [noise_level / 255]
        else:
            projected = [noise_level/255]
    else:
        if DEBLUR:
            projected = list(np.dot(P, kernel))
        else:
            return None

    t = len(projected)

    degration = np.ones([HEIGHT, WIDTH, t])

    for i in range(t):
        degration[:, :, i] = degration[:, :, i] * projected[i]

    return degration

def find_pre_image(image_name):

    lists = image_name.split('/')
    image_names = lists[-1].split('.')
    #print(image_names)
    if PATCH:
        names = image_names[0].split('_')
        name = int(names[0])
        #print(name)
        if(name % FRAME == 0):
            name = name
        else:
            name = name - 1
        names[0] = str(name)
        name = '_'.join(names)
    else:   
        name = int(image_names[0])    
        if((name - 1)%FRAME == 0):
            name = name
        else:
            name = name - 1
    #print(name)
    image_names[0] = str(name)
    lists[-1] = '.'.join(image_names)

    return '/'.join(lists)
    
def load_image(image_name):
    image = np.array(Image.open(image_name))

    # convert image to [0,1]
    if image.mean() > 255:
        image = image / 65535
    else:
        image = image / 255
    size = image.shape
    if len(image.shape) == 3:
        image = image[:, :, 0]
    image = image.reshape(size[0], size[1], 1)
    return image

def generate_patch(image_dir, scale = PATCH_SCALE, overlap = False):

    lr_files = [root + '/' + filename for root,dirs, files in os.walk(BASE_DIR + LR_DIR) for filename in files]

    hr_files = [root + '/' + filename for root,dirs, files in os.walk(BASE_DIR + HR_DIR) for filename in files]

    lr_files = sorted(lr_files)
    hr_files = sorted(hr_files)

    total = len(lr_files)

    patch_height = int(HEIGHT / PATCH_SCALE_X)
    patch_width = int(WIDTH / PATCH_SCALE_Y)

    for i in range(total):

        lr_image = np.array(Image.open(lr_files[i]))
        hr_image = np.array(Image.open(hr_files[i]))
        #print(lr_image.shape)
        for x in range(PATCH_SCALE_X):
            for y in range(PATCH_SCALE_Y):

                low_patch = lr_image[x * patch_height: x* patch_height + patch_height, y * patch_width:y * patch_width + patch_width,:]
                high_patch = hr_image[x * patch_height*SCALE : x* patch_height*SCALE + patch_height*SCALE, y * patch_width*SCALE :y * patch_width*SCALE + patch_width*SCALE,:]
                #print(low_patch.shape)
                Image.fromarray(low_patch).save(image_dir + '/LowRes/' + '%d_%d_%d.bmp'%(i,x,y),'bmp')
                Image.fromarray(high_patch).save(image_dir + '/HighRes/' + '%d_%d_%d.bmp'%(i,x,y),'bmp')
                
                
class SeperableConvLSTMCell(ConvRNNCell):
    """Basic Conv LSTM recurrent network cell.

    Parameters
    -----------
    shape : tuple of int
        The height and width of the cell.
    filter_size : tuple of int
        The height and width of the filter
    num_features : int
        The hidden size of the cell
    forget_bias : float
        The bias added to forget gates (see above).
    input_size : int
        Deprecated and unused.
    state_is_tuple : boolen
        If True, accepted and returned states are 2-tuples of the `c_state` and `m_state`.
        If False, they are concatenated along the column axis. The latter behavior will soon be deprecated.
    act : activation function
        The activation function of this layer, tanh as default.

    """

    def __init__(
            self, shape, filter_size, num_features, forget_bias=1.0, input_size=None, state_is_tuple=False,
            act=tf.nn.tanh
    ):
        """Initialize the basic Conv LSTM cell."""
        # if not state_is_tuple:
        # logging.warn("%s: Using a concatenated state is slower and will soon be "
        #             "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self.shape = shape
        self.filter_size = filter_size
        self.num_features = num_features
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = act

    @property
    def state_size(self):
        """State size of the LSTMStateTuple."""
        return (LSTMStateTuple(self._num_units, self._num_units) if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        """Number of units in outputs."""
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                # print state
                # c, h = tf.split(3, 2, state)
                c, h = tf.split(state, 2, 3)
            concat = _conv_sep([inputs, h], self.filter_size, self.num_features * 4, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            # i, j, f, o = tf.split(3, 4, concat)
            i, j, f, o = tf.split(concat, 4, 3)

            new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) * self._activation(j))
            new_h = self._activation(new_c) * tf.nn.sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h], 3)
            return new_h, new_state

def _conv_sep(args, filter_size, num_features, bias, bias_start=0.0, scope=None):
    """seperable convolution for LSTM:

    Parameters
    ----------
    args : tensor
        4D Tensor or a list of 4D, batch x n, Tensors.
    filter_size : tuple of int
        Filter height and width.
    num_features : int
        Nnumber of features.
    bias_start : float
        Starting value to initialize the bias; 0 by default.
    scope : VariableScope
        For the created subgraph; defaults to "Linear".

    Returns
    --------
    - A 4D Tensor with shape [batch h w num_features]

    Raises
    -------
    - ValueError : if some of the arguments has unspecified or wrong shape.

    """
    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[3]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Conv"):
        matrix = tf.get_variable(
            "Matrix", [filter_size[0], filter_size[1], total_arg_size_depth, 32], dtype=dtype
        )
        
        matrix_point = tf.get_variable(
            "Point_Matrix", [1, 1, total_arg_size_depth * 32, num_features], dtype=dtype
        )
        
        if len(args) == 1:
            res = tf.nn.separable_conv2d(args[0], matrix, matrix_point, strides=[1, 1, 1, 1], padding='SAME')
        else:
            res = tf.nn.separable_conv2d(tf.concat(args, 3), matrix, matrix_point, strides=[1, 1, 1, 1], padding='SAME')
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [num_features], dtype=dtype, initializer=tf.constant_initializer(bias_start, dtype=dtype)
        )
    return res + bias_term
    
if __name__ == '__main__':
    generate_patch('./Data/Patch/Square')

