import os
import sys
import glob
import numpy as np
import numpy as np
from numpy.lib.stride_tricks import as_strided
import gzip
import pickle


PATH_DATA = '../dataset/'

def get_dataset_number_frames(data_path=PATH_DATA):
	dict_path = {}
	weathers = os.listdir(PATH_DATA)
	for weather in weathers:
		if 'desktop.ini' not in weather:
			for hour in os.listdir(PATH_DATA+weather):
				for timestamp in os.listdir(PATH_DATA+weather+'/'+hour):
					_max = np.max([int(f[:-3]) for f in os.listdir(PATH_DATA+weather+'/'+hour+'/'+timestamp)])
					dict_path[PATH_DATA+weather+'/'+hour+'/'+timestamp] = _max
	return dict_path

def get_path_to_frames(dataset_frames,key):
	return [key+'/'+str(i)+'.pz' for i in range(1,dataset_frames[key]) if os.path.exists(key+'/'+str(i)+'.pz')]


def frame2numpy(frame, frameSize):
	buff = np.fromstring(frame, dtype='uint8')
	# Scanlines are aligned to 4 bytes in Windows bitmaps
	strideWidth = int((frameSize[0] * 3 + 3) / 4) * 4
	# Return a copy because custom strides are not supported by OpenCV.
	return as_strided(buff, strides=(strideWidth, 3, 1), shape=(frameSize[1], frameSize[0], 3)).copy()

def load_image(path,frameSize=(1280,640)):
	f = gzip.open(path, 'rb')
	frame = pickle.load(f)
	return frame2numpy(frame['frame'],frameSize=frameSize)

def get_frame(path,frameSize=(1280,640),dict_keys=['frame','speed','throttle','steering','brake'],active=False):
	f = gzip.open(path, 'rb')
	frame = pickle.load(f)
	frame['frame'] = frame2numpy(frame['frame'],frameSize=frameSize)
	if active == True:
		out = {}
		for key in dict_keys:
			out[key] = frame[key]
		return out
	return frame


