# Image Configuration
PATCH = True
PATCH_SCALE = 8
if PATCH:
    HEIGHT =  int(256/8) #input dimenstion of low resolution image

    WIDTH = int(320/8)
else:
    HEIGHT =  256 #input dimenstion of low resolution image
    WIDTH = 320

C = 1# channel of image

SCALE = 2# super resolution scale

# model parameter

MIDDLE_STACK = 12 # paper provides 12
MIDDLE_CHANNEL = 128

# Degration Parameter
USE_NOISE = True

DEBLUR = False
if USE_NOISE:
    if DEBLUR:
        T = 16 # dimension of projected kernel
    else:
        T = 1
else:
    if not DEBLUR:
        T = 0
    else:
        T = 15
KERNEL_NUMBER =  2

# Training setting
DEBUG = False
PRE_LOAD = False
if DEBUG:
    MAX_EPOCH = 2
else:
    MAX_EPOCH = 200
TWO_FRAME = False
TEST_PER_EPOCH = 5

BATCH_SIZE = 16
MSE_LOSS = False
RES_BLOCK = 32
# Directory
BASE_DIR = './'

if PATCH:
    LR_DIR = '/Data/Patch/LowRes'
    HR_DIR = '/Data/Patch/HighRes'
else:   
    LR_DIR = '/Data/LowRes'
    HR_DIR = '/Data/HighRes'
