# Image Configuration
PATCH = True
if PATCH:
    HEIGHT =  int(256/8) #input dimenstion of low resolution image

    WIDTH = int(320/8)
else:
    HEIGHT =  256 #input dimenstion of low resolution image
    WIDTH = 320

C = 1# channel of image

SCALE = 2# super resolution scale

# model parameter

MIDDLE_STACK = 6 # paper provides 12
MIDDLE_CHANNEL = 32

# Degration Parameter
USE_NOISE = True
if USE_NOISE:
    T = 16 # dimension of projected kernel
else:
    T = 15
KERNEL_NUMBER =  2

# Training setting
DEBUG = False
PRE_LOAD = False

TWO_FRAME = False
TEST_PER_EPOCH = 5
MAX_EPOCH = 200
BATCH_SIZE = 4


# Directory
BASE_DIR = './'

if PATCH:
    LR_DIR = '/Data/Patch/LowRes'
    HR_DIR = '/Data/Patch/HighRes'
else:   
    LR_DIR = '/Data/LowRes'
    HR_DIR = '/Data/HighRes'
