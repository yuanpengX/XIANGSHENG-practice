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

warnings.filterwarnings("ignore")

def relu1(x, name='relu1'):
    with tf.name_scope(name, "relu6") as name_scope:
        x = tf.convert_to_tensor(x, name="features")
        return tf.minimum(tf.maximum(x, 0), 1, name=name_scope)

def save_image(lr, hr, recover, directory, epoch):
    '''
    Save batch image to test
    '''
    save_dir = directory + '/test' + str(epoch)
    tl.files.exists_or_mkdir(save_dir)
    for i in range(BATCH_SIZE):
        tl.vis.save_image(lr[i], save_dir + '/lr_%d.png' % i)
        tl.vis.save_image(hr[i], save_dir + '/hr_%d.png' % i)
        tl.vis.save_image(recover[i], save_dir + '/hr_rec_%d.png' % i)


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
        projected = list(np.dot(P, kernel)) + [noise_level / 255]
    else:
        projected = list(np.dot(P, kernel))

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
        if((name - 1)%256 == 0):
            name = name
        else:
            name = name - 1
        names[0] = str(name)
        name = '_'.join(names)
    else:   
        name = int(image_names[0])    
        if((name - 1)%256 == 0):
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



def generate_patch(image_dir, scale = 8, overlap = False):

    lr_files = [root + '/' + filename for root,dirs, files in os.walk(BASE_DIR + LR_DIR) for filename in files]

    hr_files = [root + '/' + filename for root,dirs, files in os.walk(BASE_DIR + HR_DIR) for filename in files]

    lr_files = sorted(lr_files)
    hr_files = sorted(hr_files)

    total = len(lr_files)

    patch_height = int(HEIGHT / scale)
    patch_width = int(WIDTH / scale)

    for i in range(total):

        lr_image = np.array(Image.open(lr_files[i]))
        hr_image = np.array(Image.open(hr_files[i]))
        #print(lr_image.shape)
        for x in range(scale):
            for y in range(scale):

                low_patch = lr_image[x * patch_height: x* patch_height + patch_height, y * patch_width:y * patch_width + patch_width,:]
                high_patch = hr_image[x * patch_height*2 : x* patch_height*2 + patch_height*2, y * patch_width*2 :y * patch_width*2 + patch_width*2,:]
                #print(low_patch.shape)
                Image.fromarray(low_patch).save(image_dir + '/LowRes/' + '%d_%d_%d.png'%(i,x,y),'png')
                Image.fromarray(high_patch).save(image_dir + '/HighRes/' + '%d_%d_%d.png'%(i,x,y),'png')

if __name__ == '__main__':
    generate_patch('./Data/Patch')

