import random
from random import shuffle
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

SEED = int(time.time())
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
target_city = 'BERLIN'  # 'ISTANBUL'  'MOSCOW'
input_train_data_folder_path = '../0_data/' + target_city + '/'  + 'training'
input_val_data_folder_path   = '../0_data/' + target_city + '/'  + 'validation'
input_static_data_path = '../0_data/' + target_city + '/'  + target_city + "_static_2019.h5"
save_model_path = './models/1'
summary_path    = './summaries'
batch_size = 2
batch_size_val = 1
learning_rate = 3e-4
load_model_path = ''
is_training = True
num_epoch_to_train = 100000000
save_per_iteration = 10000
num_thread=2

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

  try:
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
  except Exception:
            exit(-1)

  train_data_filepath_list = get_data_filepath_list(input_train_data_folder_path)
  val_data_filepath_list   = get_data_filepath_list(input_val_data_folder_path)
  train_data_filepath_list += val_data_filepath_list

  train_set = [] 
  for i in range(len(train_data_filepath_list)):           
    for j in range(num_sequence_per_day):
      train_set.append( (i,j) )
  num_iteration_per_epoch = int(len(train_set) / batch_size)                   
  
  val_set = []
  for i in range(len(val_data_filepath_list)):
    for j in range(0, num_sequence_per_day, num_frame_sequence):
      val_set.append( (i,j) )
    for j in range(num_frame_sequence//2, num_sequence_per_day, num_frame_sequence):
      val_set.append( (i,j) )
  num_val_iteration_per_epoch = int(len(val_set) / batch_size_val)     

  static_data = None
  if 1:
          file_path = input_static_data_path
          fr = h5py.File(file_path, 'r')
          a_group_key = list(fr.keys())[0]
          data = np.asarray(fr[a_group_key], np.uint8)     
          static_data = data[np.newaxis,:,:,:]
          static_data = static_data.astype(np.float32)
          static_data = static_data / 255.0

  train_input_queue  = queue.Queue()
  train_output_queue = queue.Queue()
  
  def load_train_multithread():
    
    while True:
      if train_input_queue.empty() or train_output_queue.qsize() > 8:
        time.sleep(0.1)
        continue
      i_j_list = train_input_queue.get()
      
      train_orig_data_batch_list  = []
      train_data_batch_list = []  
      train_data_mask_list = []   
      train_data_time_list  = []
      train_stat_batch_list = [] 
      for train_i_j in i_j_list:
          (i,j) = train_i_j
          file_path = train_data_filepath_list[i]
          fr = h5py.File(file_path, 'r')
          a_group_key = list(fr.keys())[0]
          data = fr[a_group_key]
          train_data_time_list.append(float(j)/float(num_frame_per_day))                               
          train_data_batch_list.append(data[j:j+num_frame_sequence,:,:,:][np.newaxis,:,:,:,:])  

      train_data_time_list = np.asarray(train_data_time_list)
      input_time = np.reshape(train_data_time_list, (batch_size, 1))      
      train_data_batch = np.concatenate(train_data_batch_list, axis=0)   
      
      input_data = train_data_batch[:,:num_frame_before ,:,:,:]                
      orig_label = train_data_batch[:, num_frame_before:,:,:,:num_channel_out] 
      true_label = np.concatenate((                                            
            orig_label[:, 0:3, :,:,:],  
            orig_label[:, 5::3,:,:,:]   
                                      ), axis=1)

      input_data = input_data.astype(np.float32)
      true_label = true_label.astype(np.float32)
      orig_label = orig_label.astype(np.float32)
      
      
      input_data = input_data / 255.0
      true_label = true_label / 255.0
      orig_label = orig_label / 255.0
      
      input_data = np.moveaxis(input_data, 1, -1).reshape((batch_size, height, width, -1))  
      true_label = np.moveaxis(true_label, 1, -1).reshape((batch_size, height, width, -1))  
      orig_label = np.moveaxis(orig_label, 1, -1).reshape((batch_size, height, width, -1))  

      input_data = np.concatenate((                                
                input_data,
                np.repeat(static_data, batch_size, axis=0)         
                                     ), axis=3)
      orig_label_mask = None
      true_label_mask = np.ones((batch_size, height, width, num_frame_out*num_channel_out), np.float32)
      train_output_queue.put( (input_data, true_label, input_time) )


  thread_list = []
  for i in range(num_thread):

    t = threading.Thread(
                        target=load_train_multithread, 
                        )
    t.start()

  
  global_step = 0
  for epoch in range(num_epoch_to_train):
    np.random.shuffle(train_set)

    for a in range(num_iteration_per_epoch):
      
      i_j_list = []      
      for train_i_j in train_set[a * batch_size : (a+1) * batch_size]:
        i_j_list.append(train_i_j)
      train_input_queue.put(i_j_list)
    
    
    for a in range(num_iteration_per_epoch):
      
      while train_output_queue.empty():
        time.sleep(0.1)
      
      (input_data, true_label, input_time) = train_output_queue.get()

      run_out = trainer.update(input_data, true_label, input_time)
      global_step += 1

      if global_step % save_per_iteration == 0:
      
        eval_loss_list = list()
        for a in range(num_val_iteration_per_epoch):
        
          val_orig_data_batch_list  = []
          val_data_batch_list = []   
          val_data_mask_list = [] 
          val_data_time_list  = []
          val_stat_batch_list = []   
          for i_j in val_set[a * batch_size_val : (a+1) * batch_size_val]:
            (i,j) = i_j
            file_path = val_data_filepath_list[i]
            fr = h5py.File(file_path, 'r')
            a_group_key = list(fr.keys())[0]
            data = fr[a_group_key] 
            val_data_time_list.append(float(j)/float(num_frame_per_day))      
            val_data_batch_list.append(data[j:j+num_frame_sequence,:,:,:][np.newaxis,:,:,:,:])  

          val_data_time_list = np.asarray(val_data_time_list)
          input_time = np.reshape(val_data_time_list, (batch_size_val, 1))
          val_data_batch = np.concatenate(val_data_batch_list, axis=0)  
          
          input_data = val_data_batch[:,:num_frame_before ,:,:,:]                
          orig_label = val_data_batch[:, num_frame_before:,:,:,:num_channel_out]  
          true_label = np.concatenate((                                           
            orig_label[:, 0:3, :,:,:],  
            orig_label[:, 5::3,:,:,:]   
                                      ), axis=1)

          input_data = input_data.astype(np.float32)
          true_label = true_label.astype(np.float32)
          orig_label = orig_label.astype(np.float32)
          
          input_data = input_data / 255.0
          true_label = true_label / 255.0
          orig_label = orig_label / 255.0
          
          input_data = np.moveaxis(input_data, 1, -1).reshape((batch_size_val, height, width, -1))  
          true_label = np.moveaxis(true_label, 1, -1).reshape((batch_size_val, height, width, -1))  
          orig_label = np.moveaxis(orig_label, 1, -1).reshape((batch_size_val, height, width, -1))  

          input_data = np.concatenate((                                
                input_data,
                np.repeat(static_data, batch_size_val, axis=0)         
                                     ), axis=3)

          orig_label_mask = None

          run_out = trainer.evaluate(input_data, true_label, input_time)
          eval_loss_list.append(run_out['loss'])

        print('global_step:', global_step, '\t', 'epoch:', epoch, '\t', 'eval_loss:', np.mean(eval_loss_list))
        
        trainer.save_model(global_step)
        trainer.write_summary(global_step)
      