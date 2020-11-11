import random
from random import shuffle
import numpy as np
import tensorflow as tf
import datetime
import time
import queue
import logging
from PIL import Image
import itertools
import re
import os
import glob
import shutil
import sys
import copy
import h5py

target_city = 'BERLIN'
model_id_list = ['berlin_model1_1', 'berlin_model1_2', 'berlin_model2_1', 'berlin_model2_2', 'berlin_model3_1', 'berlin_model3_2',] 
test_runs_base_folder_path = '../test_runs/'
SEED = 0
num_train_file = 181
num_frame_per_day = 288
num_frame_before = 12
num_frame_sequence = 24
num_frame_out = 6   
num_sequence_per_day = num_frame_per_day - num_frame_sequence + 1 
height=495
width =436
num_channel=9
num_channel_out=8
num_channel_static = 7
visual_input_channels=115
visual_output_channels=48
vector_input_channels=1


def write_data(data, filename):
    f = h5py.File(filename, 'w', libver='latest')
    dset = f.create_dataset('array', shape=(data.shape), data = data, compression='gzip', compression_opts=9)
    f.close()

if __name__ == '__main__':
  num_model = len(model_id_list)
  try:
    if not os.path.exists('output'):
      os.makedirs('output')
    os.makedirs('output/' + target_city)
  except Exception:
    exit(-1)  
    
  test_file_id_list = []
  if 1:
    model_id = model_id_list[0]
    model_folder_path = test_runs_base_folder_path + model_id
    for filename in os.listdir(model_folder_path):
      if filename.split('.')[-1] != 'h5':
        continue
      file_id = filename.split('.')[0]
      test_file_id_list.append(file_id)
    test_file_id_list = sorted(test_file_id_list)

  for f, file_id in enumerate(test_file_id_list):
    out_file_path = 'output/' + target_city + '/' + file_id + '.h5'
    input_file_path = test_runs_base_folder_path + model_id_list[0] + '/' + file_id + '.h5'
    fr = h5py.File(input_file_path, 'r')
    a_group_key = list(fr.keys())[0]
    out_data = fr[a_group_key]
    out_data = out_data.astype(np.float32)
    for m in range(1, num_model):
      model_id = model_id_list[m]
      input_file_path = test_runs_base_folder_path + model_id + '/' + file_id + '.h5'
      fr = h5py.File(input_file_path, 'r')
      a_group_key = list(fr.keys())[0]
      out_data += ((fr[a_group_key]).astype(np.float32))
    prediction = out_data.astype(np.float32) / float(num_model)   
    prediction = np.rint(prediction)
    prediction = np.clip(prediction, 0.0, 255.0).astype(np.uint8)
    write_data(prediction, out_file_path)
