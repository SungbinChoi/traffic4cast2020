from random import shuffle
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
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as c_layers

GPU='0'
num_frame_out = 6    
num_channel_out=8   
logger = logging.getLogger("traffic")

class NetA(object):

    def __init__(self, height, width, visual_input_channels, visual_output_channels, vector_input_channels, learning_rate):

        self.batch_size = tf.placeholder(shape=None, dtype=tf.int32, name='batch_size')
        self.learning_rate = learning_rate
        self.is_train_mode = tf.placeholder(shape=None, dtype=tf.bool, name='is_train_mode')
        self.vector_in  = tf.placeholder(shape=[None, vector_input_channels],                dtype=tf.float32, name='vector_in')
        self.visual_in  = tf.placeholder(shape=[None, height, width, visual_input_channels], dtype=tf.float32, name='visual_in')
        self.true_label = tf.placeholder(shape=[None, height, width, visual_output_channels],dtype=tf.float32, name='true_label')

    #https://github.com/taki0112/Group_Normalization-Tensorflow
    @staticmethod
    def group_norm(x, G=8, eps=1e-6, scope='group_norm') :
          
        with tf.variable_scope(scope):

          original_shape = x.get_shape().as_list()
          original_shape = [-1 if s is None else s for s in original_shape]
          if len([s for s in original_shape if s == -1]) > 1:
              raise ValueError('Only one axis dimension can be undefined in the input tensor')
          N, H, W, C = original_shape
          
          G = min(G, C)

          x = tf.reshape(x, [N, H, W, G, C // G])
          
          mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
          x = (x - mean) / tf.sqrt(var + eps)

          gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
          beta  = tf.get_variable('beta',  [1, 1, 1, C], initializer=tf.constant_initializer(0.0))

          x = tf.reshape(x, [N, H, W, C]) * gamma + beta

        return x
        
        
    def create_conv1x1_bn_layer(self, x, h_size, layer_i):

            x = tf.layers.conv2d(x, h_size, kernel_size=[1, 1], strides=[1, 1], padding='same', activation=tf.nn.elu, reuse=False, name="conv1x1_"+str(layer_i))
            x = self.group_norm(x, scope=str(layer_i)+'1x1_group_norm') 
            return x
          
          
    def create_conv_bn_layer(self, x, h_size, layer_i):

            x = tf.layers.conv2d(x, h_size, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=tf.nn.elu, reuse=False, name="conv_"+str(layer_i))
            x = self.group_norm(x, scope=str(layer_i)+'_group_norm') 
            return x
          
    def create_pool_conv_bn_layer(self, x, h_size, layer_i):

            x = tf.layers.max_pooling2d(x, pool_size=[3,3], strides=[2,2], padding='same', data_format='channels_last', name="pool_"+str(layer_i))
            x = tf.layers.conv2d(x, h_size, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=tf.nn.elu, reuse=False, name="conv_"+str(layer_i))
            x = self.group_norm(x, scope=str(layer_i)+'_group_norm') 
            return x
      
      
    def create_upscale_concat_conv_bn_layer(self, x, y, h_size, layer_i):

            x = tf.layers.conv2d_transpose(x, h_size, kernel_size=[3,3], strides=[2, 2], padding='same',
                                           activation=tf.nn.elu, reuse=False, name="deconv_"+str(layer_i))
            x = tf.image.resize_images(x, y.get_shape()[1:3])
            x = tf.concat([x, y], axis=-1)

            x = tf.layers.conv2d(x, h_size, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                     activation=tf.nn.elu, reuse=False, name="conv_"+str(layer_i))
            x = self.group_norm(x, scope=str(layer_i)+'_group_norm') 
            return x

    def build(self,): 
     
     with tf.device('/gpu:' + GPU):
      out_size = num_frame_out*num_channel_out
      self.predict = self.create_visual_observation_encoder(self.visual_in, out_size, 'netA') 
      self.loss = tf.losses.mean_squared_error(self.true_label, self.predict)
      self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      gradient_limit = 1.0
      grad = self.optimizer.compute_gradients(self.loss)
      clipped_grad = list()
      for g, var in grad:
          if g is not None:
            clipped_grad.append( (tf.clip_by_value(g, -gradient_limit, gradient_limit), var) )
      self.update_batch = self.optimizer.apply_gradients(clipped_grad)
      self.update_batch = tf.group([self.update_batch, update_ops])

    def dense_block(self, x, h_size, layer_i):  

            branch_list = list()
            branch_list.append(x)
            i=0
            
            x1 = tf.layers.conv2d(x, h_size, kernel_size=[1, 1], strides=[1, 1], padding='same', activation=tf.nn.elu, reuse=False, name="conv_"+str(layer_i)+'_'+str(i))
            x1 = self.group_norm(x1, scope=str(layer_i)+'_group_norm_'+str(i))
            i += 1
            branch_list.append(x1)
            
            x2 = tf.layers.conv2d(x1, h_size//2, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=tf.nn.elu, reuse=False, name="conv_"+str(layer_i)+'_'+str(i))
            x2 = self.group_norm(x2, scope=str(layer_i)+'_group_norm_'+str(i))
            i += 1
            branch_list.append(x2)
            
            x9 = tf.layers.conv2d(x2, h_size//2, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=tf.nn.elu, reuse=False, name="conv_"+str(layer_i)+'_'+str(i))
            x9 = self.group_norm(x9, scope=str(layer_i)+'_group_norm_'+str(i))
            i += 1
            branch_list.append(x9)

            x3 = tf.layers.conv2d(x, h_size//2, kernel_size=[1, 1], strides=[1, 1], padding='same', activation=tf.nn.elu, reuse=False, name="conv_"+str(layer_i)+'_'+str(i))
            x3 = self.group_norm(x3, scope=str(layer_i)+'_group_norm_'+str(i))
            i += 1
            x3 = tf.layers.max_pooling2d(x3, pool_size=[3,3], strides=[1,1], padding='same', data_format='channels_last', name="maxpool_"+str(layer_i)+'_'+str(i))
            i += 1
            branch_list.append(x3)
            
            x = tf.concat(branch_list, axis=-1)   
            x = tf.layers.conv2d(x, h_size, kernel_size=[1, 1], strides=[1, 1], padding='same', activation=tf.nn.elu, reuse=False, name="conv1x1_"+str(layer_i)+'_'+str(i))

            return x     


    def conv_pool_dense_block(self, x, h_size, layer_i):  
            x = tf.layers.conv2d(x, h_size, kernel_size=[3, 3], strides=[2, 2], padding='same', activation=tf.nn.elu, reuse=False, name="conv_"+str(layer_i)+'_'+'convpool')
            x = self.group_norm(x, scope=str(layer_i)+'_group_norm_'+'convpool')
            return self.dense_block(x, h_size, layer_i)    
          
          
          
          

    def create_visual_observation_encoder(self, image_input, out_size, scope, ):

        with tf.variable_scope(scope):
            
            x1 = self.dense_block(image_input,  128, 1)  
            x2 = self.conv_pool_dense_block(x1, 128, 2)
            x3 = self.conv_pool_dense_block(x2, 128, 3)
            x4 = self.conv_pool_dense_block(x3, 128, 4)
            x5 = self.conv_pool_dense_block(x4, 128, 5)
            x6 = self.conv_pool_dense_block(x5, 128, 6)
            x7 = self.conv_pool_dense_block(x6, 128, 7)
            x8 = self.conv_pool_dense_block(x7, 128, 8)
            
            x100 = self.create_conv_bn_layer(x8,     128, 9, )
            
            x107 = self.create_upscale_concat_conv_bn_layer(x100, x7, 128, 107, )
            x106 = self.create_upscale_concat_conv_bn_layer(x107, x6, 128, 106, )
            x105 = self.create_upscale_concat_conv_bn_layer(x106, x5, 128, 105, )
            x104 = self.create_upscale_concat_conv_bn_layer(x105, x4, 128, 104, )
            x103 = self.create_upscale_concat_conv_bn_layer(x104, x3, 128, 103, )
            x102 = self.create_upscale_concat_conv_bn_layer(x103, x2, 128, 102, )
            x101 = self.create_upscale_concat_conv_bn_layer(x102, x1, 128, 101, )
            
            out = tf.layers.conv2d(x101, out_size, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=tf.nn.sigmoid, reuse=False, name="out")

        return out    
