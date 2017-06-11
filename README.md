# VPilot
Scripts and tools to easily communicate with [DeepGTAV](https://github.com/ai-tor/DeepGTAV). In the future a self-driving agent will be implemented.

<img src="http://forococheselectricos.com/wp-content/uploads/2016/07/tesla-autopilot-1.jpg" alt="Self-Driving Car" width="900px">

## How it works

VPilot uses JSON over TCP sockets to start, configure and send/receive commands to/from [DeepGTAV](https://github.com/ai-tor/DeepGTAV) by using the Python DeepGTAV libraries. 

_dataset.py_ and _drive.py_ serve as examples to collect a dataset using DeepGTAV and giving the control to an agent respectively.

## How to use it

DATASET.PY :
python dataset.py -d dataset/ 

### Simulation configuration
weathers = ["CLEAR", "EXTRASUNNY", "CLOUDS", "OVERCAST", "RAIN", "CLEARING", "THUNDER", "SMOG", "FOGGY", "XMAS", "SNOWLIGHT", "BLIZZARD", "NEUTRAL", "SNOW" ]

hours = [0,4,8,12,16,20]

### DATASET FORMAT:
dataset/weather/hour/timestamp_simulation/index_of_frames.pz

### PLUGIN DEV

ADD _IS_VEHICLE_DAMAGED for instance in order to get vehicule damage.

See http://www.dev-c.com/nativedb/ for a full list of all the methods that can be used.


