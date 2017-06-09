#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from deepgtav.messages import Start, Stop, Dataset, frame2numpy,Scenario
from deepgtav.client import Client

import argparse
import time
import cv2
import json
import numpy as np

# Stores a dataset file with data coming from DeepGTAV
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=None)
	parser.add_argument('-l', '--host', default='localhost', help='The IP where DeepGTAV is running')
	parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
	parser.add_argument('-d', '--dataset_path', default='dataset.pz', help='Place to store the dataset')
	args = parser.parse_args()

	weathers = ["CLEAR", "EXTRASUNNY", "CLOUDS", "OVERCAST", "RAIN", "CLEARING", "THUNDER", "SMOG", "FOGGY", "XMAS", "SNOWLIGHT", "BLIZZARD", "NEUTRAL", "SNOW" ]
	hours = [0,4,8,12,16,20]
	## CREATE A REAL DATASET WITH WEATHER CONDITIONS IN DIRECTORY.
	for weather in weathers:
		# Creates a new connection to DeepGTAV using the specified ip and port. 
		# If desired, a dataset path and compression level can be set to store in memory all the data received in a gziped pickle file.
		for hour in hours:
			client = Client(ip=args.host, port=args.port, datasetPath=args.dataset_path, compressionLevel=9)
			infos = {}
			infos['weather'] = weather
			infos['hour'] = hour
			infos['time'] = int(time.time())

			# Configures the information that we want DeepGTAV to generate and send to us. 
			# See deepgtav/messages.py to see what options are supported
			dataset = Dataset(rate=4, frame=[1280,640], throttle=True, brake=True, steering=True, vehicles=True, peds=True, trafficSigns=True, reward=[15.0, 0.0], direction=None, speed=True, yawRate=True, location=True, time=True)
			# Send the Start request to DeepGTAV.
			scenario = Scenario(time=[hour,0],weather=weather,drivingMode=[786603,15.0]) # Driving style is set to normal, with a speed of 15.0 mph. All other scenario options are random.
			client.sendMessage(Start(dataset=dataset,scenario=scenario))
			stoptime = time.time() + 2*60
			while time.time() < stoptime:
				try:
					message = client.recvMessage(infos)	
				except KeyboardInterrupt:
				# We tell DeepGTAV to stop
					break
		# We tell DeepGTAV to stop
	client.sendMessage(Stop())
	client.close()
