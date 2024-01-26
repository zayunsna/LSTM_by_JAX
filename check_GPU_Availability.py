#!/usr/bin/env python
# -*- coding: utf-8 -*-

## This script will check GPU Usage in both library "tensorflow" & "torch"
## the printed info represent not only your system had GPU driver, cuda 
## and also both library is detected GPU device.

import os
## In order to reduce the log message what we don't need now
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.python.client import device_lib
import torch as tt


## It needs a Nvidia Graphic Driver
print("#"*50)
print(" ==> Chech GPU basic info")
os.system('nvidia-smi')
print("")

## It required CUDA Toolkit
print("#"*50)
print(" ==> Check Installed CUDA Version")
os.system('nvcc -V')
print("")
## To check Tensorflow can find the GPU device
## the printed list contained the "GPU" then you can use GPU in tensorflow library
print("#"*50)
print(" ==> Check GPU can running at tensorflow")
print(" Connected devices info ")
print(device_lib.list_local_devices())
print("")

## To check Torch can use GPU device
print("#"*50)
print(" ==> Check GPU can running at torch ")
is_okay_torch = tt.cuda.is_available()
print(" Status : {}".format(is_okay_torch))
print("")