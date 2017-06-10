import tensorflow as tf 
import numpy as np 
import os 
import sys
import model_utils as mu 
import get_frames as gf
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
from keras import backend as K
from scipy.misc import imresize
from keras.preprocessing.image import Iterator
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import threading
import glob




class Iterator(object):
	def __init__(self, n, batch_size, shuffle, seed):
		self.n = n
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.batch_index = 0
		self.total_batches_seen = 0
		self.lock = threading.Lock()
		self.index_generator = self._flow_index(n, batch_size, shuffle, seed)
	def reset(self):
		self.batch_index = 0

	def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
		self.reset()
		while 1:
			if seed is not None:
				np.random.seed(seed + self.total_batches_seen)
			if self.batch_index == 0:
				index_array = np.arange(n)
				if shuffle:
					index_array = np.random.permutation(n)

			current_index = (self.batch_index * batch_size) % n
			if n > current_index + batch_size:
				current_batch_size = batch_size
				self.batch_index += 1
			else:
				current_batch_size = n - current_index
				self.batch_index = 0
			self.total_batches_seen += 1
			yield (index_array[current_index: current_index + current_batch_size],
					current_index, current_batch_size)

	def __iter__(self):
		return self

	def __next__(self, *args, **kwargs):
		return self.next(*args, **kwargs)

class UniqueConditionGenerator(Iterator):
	"""
	Example of use 
	import model_generator as md
	for X,Y in md.UniqueConditionIterator(10,len(clear_0_frames_path),'../dataset/EXTRASUNNY/12/1496614796'):
	    for i,x in enumerate(X):
	        print(Y[i])
	        plt.imshow(x)
	        plt.show()
	    break

	"""
	def __init__(self, batch_size, len_file, path, size=(120,360), shuffle=False, seed=None):
		self.batch_size=batch_size   
		self._len_file = len_file
		self._path = path
		self._size = size
		super(UniqueConditionGenerator, self).__init__(len_file, batch_size, shuffle, seed)

	def next(self):
		with self.lock:
			index_array, current_index, current_batch_size = next(self.index_generator)

		paths = gf.create_paths(self._path,index_array,True)
		data = gf.load_data_from_paths(paths,select=True)
		X = np.array([imresize(d['frame'],self._size) for d in data])
		Y = np.array([[d['throttle'],d['brake'],d['steering']] for d in data]) 
		return X,Y
"""
Order of FRAMES OUTUTS
[0 1 2 3 4 5 6 7 8 9] 0 10
[10 11 12 13 14 15 16 17 18 19] 
[20 21 22 23 24 25 26 27 28 29] 
[30 31 32 33 34 35 36 37 38 39] 
[40 41 42 43 44 45 46 47 48 49] 
[50 51 52 53 54 55 56 57 58 59] 
[60 61 62 63 64 65 66 67 68 69] 
[70 71 72 73 74 75 76 77 78 79] 
[80 81 82 83 84 85 86 87 88 89] 
[90 91 92 93 94 95 96 97 98 99] 

"""
### THIS GENERATOR IS NOT RIGHT BECAUSE WE MIGHT WANT TO DO SEQ2SEQ WITH LSMT . 
### WE NEED TO BUILD A CUSTOM ITERATOR FOR SEQUENTIAL INPUTS.
        
class SequentialUniqueConditionIterator(object):
	def __init__(self, n, batch_size, shuffle, seed):
		self.n = n
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.batch_index = 0
		self.total_batches_seen = 0
		self.lock = threading.Lock()
		self.index_generator = self._flow_index(n, batch_size, shuffle, seed)
	def reset(self):
		self.batch_index = 0

	def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
		self.reset()
		while 1:
			if seed is not None:
				np.random.seed(seed + self.total_batches_seen)
			if self.batch_index == 0:
				index_array = np.arange(n)
				if shuffle:
					index_array = np.random.permutation(n)

			current_index = (self.batch_index + 1 ) % n ## Changed batch_size by 1 and * in +
			if n > current_index + batch_size:
				current_batch_size = batch_size
				self.batch_index += 1
			else:
				current_batch_size = n - current_index
				self.batch_index = 0
			self.total_batches_seen += 1
			yield (index_array[current_index: current_index + current_batch_size],
					current_index, current_batch_size)

	def __iter__(self):
		return self

	def __next__(self, *args, **kwargs):
		return self.next(*args, **kwargs)

class SequentialUniqueConditionGenerator(SequentialUniqueConditionIterator):
	"""
	Example of use 
	import model_generator as md
	for X,Y in md.UniqueConditionIterator(10,len(clear_0_frames_path),'../dataset/EXTRASUNNY/12/1496614796'):
	    for i,x in enumerate(X):
	        print(Y[i])
	        plt.imshow(x)
	        plt.show()
	    break

	"""
	def __init__(self, batch_size, len_file, path, size=(120,360), shuffle=False, seed=None):
		self.batch_size=batch_size   
		self._len_file = len_file
		self._path = path
		self._size = size
		super(SequentialUniqueConditionGenerator, self).__init__(len_file, batch_size, shuffle, seed)

	def next(self):
		with self.lock:
			index_array, current_index, current_batch_size = next(self.index_generator)

		paths = gf.create_paths(self._path,index_array,True)
		data = gf.load_data_from_paths(paths,select=True)
		X = np.array([imresize(d['frame'],self._size) for d in data])
		Y = np.array([[d['throttle'],d['brake'],d['steering']] for d in data]) 
		return X,Y


class SequentialWeatherIterator(object):
	def __init__(self, n, batch_size, shuffle, seed):
		self.n = n
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.batch_index = 0
		self.total_batches_seen = 0
		self.lock = threading.Lock()
		self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

	def reset(self):
		self.batch_index = 0

	def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
		self.reset()
		while 1:
			if seed is not None:
				np.random.seed(seed + self.total_batches_seen)
			if self.batch_index == 0:
				index_array = np.arange(n)
				if shuffle:
					index_array = np.random.permutation(n)

			current_index = (self.batch_index + 1 ) % n ## Changed batch_size by 1 and * in +
			if n > current_index + batch_size:
				current_batch_size = batch_size
				self.batch_index += 1
			else:
				current_batch_size = n - current_index
				self.batch_index = 0
			self.total_batches_seen += 1
			yield (index_array[current_index: current_index + current_batch_size],
					current_index, current_batch_size)

	def __iter__(self):
		return self

	def __next__(self, *args, **kwargs):
		return self.next(*args, **kwargs)

class SequentialWeatherGenerator(SequentialWeatherIterator):
	"""
	Example of use 
	import model_generator as md
	for index_array in md.SequentialWeatherGenerator(4,PATH_DATA):
	    print(index_array)
	"""
	def __init__(self, batch_size, dataset_path, fixed_weather=True, size=(120,360), shuffle=False, seed=None):
		self._batch_size=batch_size   
		self._shuffe = shuffle
		self._seed = seed
		self._dataset_path = dataset_path
		self._size = size
		self._fixed_weather = fixed_weather
		self._weathers = [f for f in os.listdir(self._dataset_path) if f[0].upper() == f[0]]
		self._selected_weather = np.random.choice(self._weathers)
		self.random_directory()
		super(SequentialWeatherGenerator, self).__init__(self._number_frames, self._batch_size, self._shuffe, self._seed)

	def random_directory(self):
		if not self._fixed_weather:
			self._selected_weather = np.random.choice(self._weathers)
		self._files = [f.replace('\\','/') for f in glob.glob(self._dataset_path+self._selected_weather+'/*/*')]
		self._selected_file = np.random.choice(self._files)
		self._number_frames = len(os.listdir(self._selected_file+'/'))

	def next(self):
		with self.lock:
			index_array, current_index, current_batch_size = next(self.index_generator)
		#super(SequentialWeatherGenerator, self).reset()
		paths = gf.create_paths(self._selected_file,index_array,True)
		data = gf.load_data_from_paths(paths,select=True)
		X = np.array([imresize(d['frame'],self._size) for d in data])
		Y = np.array([[d['throttle'],d['brake'],d['steering']] for d in data]) 
		if index_array[-1] == (self._number_frames-1):
			self.random_directory()
			#print(self._selected_file)
			super(SequentialWeatherGenerator, self).__init__(self._number_frames, self._batch_size, self._shuffe, self._seed)

		return X,Y



