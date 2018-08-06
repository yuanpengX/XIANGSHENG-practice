# Image Configuration
PATCH = True
PATCH_SCALE = 8
SQUARE = True
if PATCH:
    if SQUARE:
        HEIGHT = 32#input dimenstion of low resolution image
        WIDTH = 32
    else:
        HEIGHT = int(256 / PATCH_SCALE)
        WIDTH = int(320 / PATCH_SCALE)        
else:
    HEIGHT =  256 #input dimenstion of low resolution image
    WIDTH = 320
    
if SQUARE:
        PATCH_SCALE_X = int(256/32)
        PATCH_SCALE_Y = int(320/32)
else:
        PATCH_SCALE_X = PATCH_SCALE
        PATCH_SCALE_Y = PATCH_SCALE
print(PATCH_SCALE_X)
print(PATCH_SCALE_Y)
C = 1# channel of image

SCALE = 2# super resolution scale

# frame per video
FRAME = 256
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
LR = 1e-5
BATCH_SIZE = PATCH_SCALE_X * PATCH_SCALE_Y
MSE_LOSS = False
RES_BLOCK = 32
# Directory
BASE_DIR = './'

if PATCH:
    if SQUARE:
        LR_DIR = '/Data/Patch/Square/LowRes'
        HR_DIR = '/Data/Patch/Square/HighRes'
    else:
        LR_DIR = '/Data/Patch/LowRes'
        HR_DIR = '/Data/Patch/HighRes'
else:   
    LR_DIR = '/Data/LowRes'
    HR_DIR = '/Data/HighRes'
