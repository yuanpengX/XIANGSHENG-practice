# coding: utf-8
# author: Yuanpeng Xiong
# time: 2018-07-03

import numpy as np 
import os
from PIL import Image
from random import randint
from config import *
from utils import *
from datetime import datetime

class DataLoader():
	
	def __init__(self):
		'''
		'''
		
		# Get basic data from SRMD setting
		self.P, AtrpGaussianKernel, directKernel = load_mat('srmdModel/SRMDx3.mat')
		self.kernels = [AtrpGaussianKernel, directKernel]
		
		self.batch_size = BATCH_SIZE
		
		self.debug = DEBUG  # if mode i set debug, only load small number of images
	
		self.hr_files = [os.path.join(root, filename) for root,dirs,files in os.walk(BASE_DIR + HR_DIR) for filename in files]
		self.lr_files = [os.path.join(root, filename) for root,dirs,files in os.walk(BASE_DIR + LR_DIR) for filename in files]
		
		self.hr_files = sorted(self.hr_files)
		self.lr_files = sorted(self.lr_files)
		print('Total %d images in the given file directory' % (len(self.lr_files)))
		if DEBUG:
			self.hr_files = self.hr_files[:4000]
			self.lr_files = self.lr_files[:4000]
			
		# split data into training set/ valid set and test set
		
		self.total = len(self.hr_files)
		
		self.capacity = int(self.total * 0.7)
		

		self.valid_size = int(self.total * 0.2)
		
		self.test_size = self.total - self.capacity - self.valid_size
		
		
		self.train_lr_files = self.lr_files[:self.capacity]
		self.train_hr_files = self.hr_files[:self.capacity]
		
		self.valid_lr_files = self.lr_files[self.capacity: self.capacity + self.valid_size]
		self.valid_hr_files = self.hr_files[self.capacity: self.capacity + self.valid_size]
		
		self.test_lr_files = self.lr_files[self.capacity + self.valid_size:]
		self.test_hr_files = self.hr_files[self.capacity + self.valid_size:]
		
		if PRE_LOAD:
			if DEBUG:
				start = datetime.now()
			self.train_lr_images = [load_image(file) for file in self.train_lr_files]
			self.train_hr_images = [load_image(file) for file in self.train_hr_files]
			
			self.valid_lr_images = [load_image(file) for file in self.valid_lr_files]
			self.valid_hr_images = [load_image(file) for file in self.valid_hr_files]
				
			self.test_lr_images = [load_image(file) for file in self.test_lr_files]
			self.test_hr_images = [load_image(file) for file in self.test_hr_files]

		else:
			self.train_lr_images = []
			self.train_hr_images = []
			
			self.valid_lr_images = []
			self.valid_hr_images = []
				
			self.test_lr_images = []
			self.test_hr_images = []
		self.train_index = 0
		self.valid_index = 0
		self.test_index = 0
	
	def get_batch(self, index, capacity, lr_images, hr_images, lr_files, hr_files):
	
	
		if index + self.batch_size > capacity :
			index = 0
					
		start = index
		end = index + self.batch_size
		if PRE_LOAD:
				
			_lr_images = [np.concatenate([image, self.generate_kernel()],2) for image in lr_images[start:end]]
			_hr_images = hr_images[start:end]
		else:
			_lr_images = [np.concatenate([load_image(image), self.generate_kernel()],2) for image in lr_files[start:end]]
			_hr_images = [load_image(image) for image in hr_files[start:end]]
				
		index = end	
		return index, _lr_images, _hr_images
		
	def get_two_frame(self, index, capacity, lr_images, hr_images, lr_files, hr_files):

		if PRE_LOAD:
			print('PRELOAD NOT IMPLEMENTED UNDER TWO FRAME MODE')

		if index + self.batch_size > capacity :
			index = 0

		start = index
		end = index + self.batch_size
		if PRE_LOAD:
			
			_lr_images = [np.concatenate([image, self.generate_kernel()],2) for image in lr_images[start:end]]
			_hr_images = hr_images[start:end]
		else:
			_lr_images = [np.concatenate([load_image(image), self.generate_kernel()],2) for image in lr_files[start:end]]
					
			_pr_images = [np.concatenate([load_image(find_pre_image(image)), self.generate_kernel()], 2) for image in lr_files[start:end]]
			
			_hr_images = [load_image(image) for image in hr_files[start:end]]

		index = end
		return index, _lr_images, _pr_images, _hr_images

				
	def get_next_train_batch(self):
		'''
		
		'''
		if DEBUG:
			start = datetime.now()
		if TWO_FRAME:
			self.train_index, lr_images, pre_images, hr_images = self.get_two_frame(self.train_index, self.capacity, self.train_lr_images, self.train_hr_images, self.train_lr_files, self.train_hr_files)
			return np.array(lr_images), np.array(pre_images), np.array(hr_images)
			
		else:
			self.train_index, lr_images, hr_images = self.get_batch(self.train_index, self.capacity, self.train_lr_images, self.train_hr_images, self.train_lr_files, self.train_hr_files)
			return np.array(lr_images), np.array(hr_images)
			
		
	def get_next_test_batch(self):
		'''
		
		'''
		if TWO_FRAME:
			self.test_index, lr_images, pre_images, hr_images = self.get_two_frame(self.test_index, self.test_size, self.test_lr_images, self.test_hr_images, self.test_lr_files, self.test_hr_files)
			return np.array(lr_images), np.array(pre_images), np.array(hr_images)

		else:

			self.test_index, lr_images, hr_images = self.get_batch(self.test_index, self.test_size, self.test_lr_images, self.test_hr_images, self.test_lr_files, self.test_hr_files)
			return np.array(lr_images), np.array(hr_images)
	
	def get_next_valid_batch(self):
		'''
		
		'''
		if TWO_FRAME:
			self.valid_index, lr_images, pre_images, hr_images = self.get_two_frame(self.valid_index, self.valid_size, self.valid_lr_images, self.valid_hr_images, self.valid_lr_files, self.valid_hr_files)
			return np.array(lr_images), np.array(pre_images), np.array(hr_images)

		else:

			self.valid_index, lr_images, hr_images = self.get_batch(self.valid_index, self.valid_size, self.valid_lr_images, self.valid_hr_images, self.valid_lr_files, self.valid_hr_files)
		
			return np.array(lr_images), np.array(hr_images)
		
	def get_capacity(self):
		'''
		
		'''
		return self.capacity

	def get_valid_size(self):
		return self.valid_size

	def get_test_size(self):
		return self.test_size
	def generate_kernel(self):
	
		'''
		By now, we only provids Two types of kernel
		'''
		
		kernel_type = self.kernels[randint(0,KERNEL_NUMBER - 1)]
		
		kernel_number = kernel_type.shape[-1]
		
		kernel = kernel_type[:,:,:,randint(0, kernel_number - 1)].reshape([-1,1])
		
		noise_level = randint(0, 75)
		
		degration = construct_degration(self.P, kernel, noise_level) 

		return degration

if __name__ == '__main__':
	dataloader = DataLoader()
	
	a,b = dataloader.get_next_test_batch()
	print('test lr shape: ', a.shape)
	print('test hr shape:', b.shape)
	print('max lr is: ', a.max())
	print('max hr is:', b.max())
	a,b = dataloader.get_next_train_batch()
	print('train lr shape: ', a.shape)
	print('train hr shape:', b.shape)
	
	a,b = dataloader.get_next_valid_batch()
	print('train lr shape: ', a.shape)
	print('train hr shape:', b.shape)
	
	
	
