"""
@Authors 
Ashutosh Upreti 	: 2014B4A70784G
Himanshu Singhvi 	: 2014B2A70716G
Abhijeet Khasnis 	: 2014B3A30529G

"""

import math
import numpy as np
import cv2
import keras
import os
import scipy
from keras.models import Sequential
from keras.layers import Input, Dense ,Dropout
from keras.layers import Conv2D, Dense ,Conv2DTranspose
import keras.backend as K
from keras.models import model_from_json
import json
from keras import models


def psnr2(a,b):

	MSE = np.sum((np.array(a,np.int32)-np.array(b,np.int32))**2)/(256*256*3)
	return 20*math.log10(255)-10*math.log10(MSE)
	
class Derain(object):

	def __init__(self, data_dir,checkpoint_dir="./checkpoints2"):
		self.images = []
		self.dimension = []
		self.derainedImage = []
		self.rainedImage = []
		self.mcheckpoint = keras.callbacks.ModelCheckpoint(checkpoint_dir + "/" + "weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
		self.n = 0;

		for fname in os.listdir(data_dir):	

			img = cv2.imread(os.path.join(data_dir,fname))
			if img is not None:
				self.images.append(img)
				self.dimension.append(list(img.shape))

				columnSize=self.dimension[self.n][1]/2
				self.derainedImage.append(img[:, 0:columnSize, :])
				self.rainedImage.append(img[:, columnSize:, :])

			self.n = self.n + 1


		self.x = 256
		self.y = 256

		for i in range(self.n):
			self.derainedImage[i] = cv2.resize(self.derainedImage[i], (self.y, self.x))
			self.rainedImage[i] = cv2.resize(self.rainedImage[i], (self.y, self.x))

	def psnr(self, rainImage,derainImage):
		MSE = K.mean(K.square(rainImage-derainImage))

		if(MSE == 0):
		   return 50
		else:
		   return 20*K.log(255.0/K.sqrt(MSE)) / K.log(K.constant(10))

	def load_model(self, **params):
		files = os.listdir("./checkpoints")
		fileName = [i for i in files if '.hdf5' in i]
		print fileName
		print("#"*5+" Using saved model-" + "weights.50-939.48.hdf5" + " " + "#"*5)
		md = models.load_model(os.path.join("./checkpoints2/", "weights.10-687.26.hdf5"), custom_objects = {"psnr" : self.psnr})
		print("#"*5+" Model Loaded " + "#"*5)
		self.model = md
		# self.d={'rainy':np.asarray(self.rainedImage),'original':np.asarray(self.derainedImage)}

		# self.model.fit(self.d['rainy'],self.d['original'],  epochs=10, batch_size=4,validation_split=0.3, callbacks=[self.mcheckpoint])



	#Run for starting 100 images in training data set
	def test(self):
		# print("#"*5+ " Testing " + "#"*5)
		# lossMetric = self.model.evaluate(self.d['rainy'].reshape(-1,256,256,3), self.data['original'].reshape(-1,256,256,3), batch_size = 1)
		# count = 0

		# for in lossMetric:
		# 	print(self.model.metrics_names[count], 1)
		# 	count += 1
		psnr_t=0
		for i in range(100):
			di=self.model.predict(np.reshape(self.rainedImage[i],[1,256,256,3]))

			#print "-----"
			#print self.psnr(np.reshape(self.rainedImage[i],[1,256,256,3]),di)


			dd=np.reshape(di,[256,256,3])
			cv2.imwrite("./outtest4/"+str(i)+"rained.jpg",self.rainedImage[i])
			cv2.imwrite("./outtest4/"+str(i)+"ground.jpg",self.derainedImage[i])
			cv2.imwrite("./outtest4/"+str(i)+"predict.jpg",dd)
			psnr_=psnr2(dd,self.derainedImage[i])
			psnr_t=psnr_t+psnr_
			print psnr_
		print "final"
		print psnr_t/100	

	#provide location of the sample test data to check the psnr
	def sample_test(self):
		# print("#"*5+ " Testing " + "#"*5)
		# lossMetric = self.model.evaluate(self.d['rainy'].reshape(-1,256,256,3), self.data['original'].reshape(-1,256,256,3), batch_size = 1)
		# count = 0

		# for in lossMetric:
		# 	print(self.model.metrics_names[count], 1)
		# 	count += 1
		data_dir="./test_sampe"
		n=0
		images=[]
		dimension=[]
		derainedImage=[]
		rainedImage=[]

		for fname in os.listdir(data_dir):	

			img = cv2.imread(os.path.join(data_dir,fname))
			if img is not None:
				images.append(img)
				dimension.append(list(img.shape))

				columnSize=dimension[n][1]/2
				derainedImage.append(img[:, 0:columnSize, :])
				rainedImage.append(img[:, columnSize:, :])

			n = n + 1


		x = 256
		y = 256

		for i in range(n):
			derainedImage[i] = cv2.resize(derainedImage[i], (y, x))
			rainedImage[i] = cv2.resize(rainedImage[i], (y, x))


		psnr_t=0
		for i in range(n):
			di=self.model.predict(np.reshape(rainedImage[i],[1,256,256,3]))

			#print "-----"
			#print psnr(np.reshape(rainedImage[i],[1,256,256,3]),di)


			dd=np.reshape(di,[256,256,3])
			cv2.imwrite("./outtest3/"+str(i)+"rained.jpg",rainedImage[i])
			cv2.imwrite("./outtest3/"+str(i)+"ground.jpg",derainedImage[i])
			cv2.imwrite("./outtest3/"+str(i)+"predict.jpg",dd)
			psnr_=psnr2(dd,derainedImage[i])
			psnr_t=psnr_t+psnr_
			print psnr_
		print "final"
		print psnr_t/n	

		

	def train(self, training_steps=10):

		input_shape=(256,256,3)
		epochs = 10
		batch_size = 4
		
		self.d={'rainy':np.asarray(self.rainedImage),'original':np.asarray(self.derainedImage)}

		self.model= Sequential()

		self.model.add(keras.layers.InputLayer(input_shape=input_shape))

		self.model.add(keras.layers.convolutional.Conv2D(64, 2, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))

		self.model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.1, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))

		self.model.add(keras.layers.convolutional.Conv2D(128, 2, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))

		self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))

		self.model.add(keras.layers.convolutional.Conv2D(256, 3, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))

		self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))

		self.model.add(keras.layers.convolutional.Conv2D(512, 2, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))

		self.model.add(Dropout(0.5))

		self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))

		self.model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.1, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))

		self.model.add(keras.layers.UpSampling2D(size=(2, 2), data_format=None))

		self.model.add(Conv2DTranspose(512, 2, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))

		self.model.add(keras.layers.UpSampling2D(size=(2, 2), data_format=None))

		self.model.add(Conv2DTranspose(256, 3, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))

		self.model.add(keras.layers.UpSampling2D(size=(2, 2), data_format=None))

		self.model.add(Conv2DTranspose(128, 2, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))

		self.model.add(Conv2DTranspose(3, 2, strides=(1,1),padding='same',data_format="channels_last",activation='relu'))

		#adaDeltaOptimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
		#self.model.compile(loss='mean_squared_error', optimizer=adaDeltaOptimizer, metrics=['accuracy',psnr])

		self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy',self.psnr])

		self.model.fit(self.d['rainy'],self.d['original'],  epochs=epochs, batch_size=batch_size,validation_split=0.3, callbacks=[self.mcheckpoint])
		

		for i in range(10):
			di=self.model.predict(np.reshape(self.rainedImage[i],[1,256,256,3]))
			dd=np.reshape(di,[256,256,3])
			cv2.imwrite("./outtest2/"+str(i)+"rained.jpg",self.rainedImage[i])
			cv2.imwrite("./outtest2/"+str(i)+"ground.jpg",self.derainedImage[i])
			cv2.imwrite("./outtest2/"+str(i)+"predict.jpg",dd)



if __name__=='__main__':

	p = Derain("./training", "./checkpoints2")
	# p.load_model()
	p.load_model()
	p.test()
	#p.train()
	# p.sample_test()
