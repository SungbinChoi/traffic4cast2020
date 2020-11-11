import random
import numpy as np
import tensorflow as tf
import datetime
import time
import queue
import threading
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
from net_all import *
from trainer_all import *

target_city   = 'BERLIN'          # 'ISTANBUL'  'MOSCOW'
load_model_id = 'berlin_model1_1' # 
load_model_path  = load_model_id
test_save_folder_path = '../test_runs/' + load_model_id
input_test_data_folder_path  = '../0_data/' + target_city + '/'  + 'testing'
input_static_data_path       = '../0_data/' + target_city + '/'  + target_city + "_static_2019.h5" 
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
save_model_path = './models/1'
summary_path    = './summaries'
learning_rate = 3e-4
is_training = False


def write_data(data, filename):
    f = h5py.File(filename, 'w', libver='latest')
    dset = f.create_dataset('array', shape=(data.shape), data = data, compression='gzip', compression_opts=9)
    f.close()


def get_data_filepath_list(input_data_folder_path):
  data_filepath_list = []
  for filename in os.listdir(input_data_folder_path):
    if filename.split('.')[-1] != 'h5':     
      continue
    data_filepath_list.append(os.path.join(input_data_folder_path, filename))
  data_filepath_list = sorted(data_filepath_list)
  return data_filepath_list


if __name__ == '__main__':
  random.seed(SEED)
  np.random.seed(SEED)
  tf.set_random_seed(SEED)
  trainer = Trainer(height, width, visual_input_channels, visual_output_channels, vector_input_channels, learning_rate, 
                    save_model_path, load_model_path, summary_path, is_training)
  tf.reset_default_graph() 

  test_data_filepath_list = get_data_filepath_list(input_test_data_folder_path)
  test_output_filepath_list = list()
  for test_data_filepath in test_data_filepath_list:
    filename = test_data_filepath.split('/')[-1]
    test_output_filepath_list.append(test_save_folder_path + '/' + filename)
    
  try:
    if not os.path.exists(test_save_folder_path):
      os.makedirs(test_save_folder_path)
  except Exception:
    exit(-1)  

  static_data = None
  if 1:
          file_path = input_static_data_path
          fr = h5py.File(file_path, 'r')
          a_group_key = list(fr.keys())[0]
          data = np.asarray(fr[a_group_key], np.uint8)         
          static_data = data[np.newaxis,:,:,:]
          static_data = static_data.astype(np.float32)
          static_data = static_data / 255.0

  for i in range(len(test_data_filepath_list)):
    file_path = test_data_filepath_list[i]
    out_file_path = test_output_filepath_list[i]
    fr = h5py.File(file_path, 'r')
    a_group_key = list(fr.keys())[0]
    data = fr[a_group_key]
    data = np.array(data, np.uint8)      
    batch_size_test = data.shape[0]
    test_data_batch_list  = []  
    for j in range(batch_size_test):
      data_sliced = data[j,:,:,:,:]
      test_data_batch_list.append(data_sliced[np.newaxis,:,:,:,:])
    input_data = np.concatenate(test_data_batch_list, axis=0).astype(np.float32)  
    input_data = input_data / 255.0
    input_data = np.moveaxis(input_data, 1, -1).reshape((batch_size_test, height, width, -1))
    input_data = np.concatenate(( input_data, np.repeat(static_data, batch_size_test, axis=0)), axis=3)
    input_time = np.zeros((batch_size_test, 1), np.float32)

    prediction_list = []
    for b in range(batch_size_test):
      run_out_one = trainer.infer(input_data[b,:,:,:][np.newaxis,:,:,:], input_time[b,:][np.newaxis,:], )  
      prediction_one = run_out_one['predict']
      prediction_list.append(prediction_one)
    prediction = np.concatenate(prediction_list, axis=0)    
    prediction = np.moveaxis(np.reshape(prediction, (batch_size_test, height, width, num_channel_out, num_frame_out, )), -1, 1)
    prediction = prediction.astype(np.float32) * 255.0
    prediction = np.rint(prediction)   
    prediction = np.clip(prediction, 0.0, 255.0).astype(np.uint8)
    write_data(prediction, out_file_path) 

