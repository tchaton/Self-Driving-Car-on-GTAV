#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import socket, struct
import pickle
import gzip
import numpy as np
import os
import time

class Targets:
	def __init__(self, datasetPath, compressionLevel):
		self.pickleFile = None
		self._seed = 0
		self._datasetPath = datasetPath

		#if datasetPath != None:
		#	self.pickleFile = gzip.open(datasetPath, mode='ab', compresslevel=compressionLevel)

	def parse(self, frame, jsonstr,infos):
		dct = json.loads(jsonstr)
		dct['frame'] = frame
		directory_path = self._datasetPath+infos['weather']+'/'+str(infos['hour'])+'/'+str(infos['time'])
		self.mk(directory_path)
		pickle.dump(dct, gzip.open(directory_path+'/'+str(self._seed)+'.pz', mode='ab', compresslevel=0))
		self._seed+=1
		return dct
	def mk(self,path):
		if not os.path.exists(path):
			os.makedirs(path)
			print('Directory created',path)
	def reset_seed(self):
		self._seed = 0



class Client:
	def __init__(self, ip='localhost', port=8000, datasetPath=None, compressionLevel=0):
		print('Trying to connect to DeepGTAV')
		
		self.targets = Targets(datasetPath, compressionLevel)

		try:
			self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self.s.connect((ip, int(port)))
		except:
			print('ERROR: Failed to connect to DeepGTAV')
		else:
			print('Successfully connected to DeepGTAV')

	def sendMessage(self, message):
		jsonstr = message.to_json().encode('utf-8')
		try:
			self.s.sendall(len(jsonstr).to_bytes(4, byteorder='little'))
			self.s.sendall(jsonstr)
		except Exception as e:
			print('ERROR: Failed to send message. Reason:', e)
			return False
		return True

	def recvMessage(self,infos):
		frame = self._recvall()
		if not frame: 
			print('ERROR: Failed to receive frame')		
			return None
		data = self._recvall()
		if not data: 
			print('ERROR: Failed to receive message')		
			return None
		return self.targets.parse(frame, data.decode('utf-8'),infos)

	def _recvall(self):
		#Receive first size of message in bytes
		data = b""
		while len(data) < 4:
			packet = self.s.recv(4 - len(data))
			if not packet: return None
			data += packet
		size = struct.unpack('I', data)[0]

		#We now proceed to receive the full message
		data = b""
		while len(data) < size:
			packet = self.s.recv(size - len(data))
			if not packet: return None
			data += packet
		return data

	def close(self):
		self.s.close()
