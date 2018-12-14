#!/usr/bin/python
# -*- coding: utf-8 -*-
import tarfile
import os.path
from os import listdir
from os.path import isfile, join
import json
import re, bz2, random, csv
from bz2 import BZ2File
from io import BytesIO
import nltk.data, csv
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
import numpy as np
from tensorflow.contrib.slim.python.slim.data import parallel_reader
from tensorflow.python.ops import io_ops
#shuffle queue and bucketing and parallel read

import tensorflow as tf


def read_and_decode(filename_queue,tfrec_len): 
    batch_size = 5
    IMAGE_WIDTH = 10
    common_queue = tf.RandomShuffleQueue(
          capacity=10+3*batch_size,
          min_after_dequeue=10+batch_size,
          dtypes=[tf.string,tf.string])
    
    p_reader = parallel_reader.ParallelReader(io_ops.TFRecordReader, common_queue, num_readers=tfrec_len)

    _, serialized_example = p_reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
        })

    bucket_boundaries = [7,11,18,23,33]
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    
    height = tf.cast(features['height'], tf.int32)
    
    image_shape = tf.stack([height, IMAGE_WIDTH])
    print("*********image shape****",image_shape)
        
    image = tf.reshape(image, image_shape)
    
    (l, images) = tf.contrib.training.bucket_by_sequence_length(height, [image], batch_size,\
                bucket_boundaries,capacity=1 * batch_size, dynamic_pad=True)
    
    table_index = tf.train.range_input_producer(
        batch_size, shuffle=True).dequeue_many(batch_size)
    
    ret = tf.gather(images[0], table_index)
    
    return (l,ret)

mydir = 'tfrecData'
files = [f for f in listdir(mydir) if isfile(join(mydir, f))]

tfrecords_filename = sorted(files, key=lambda x: int(x.split('.')[0].split('_')[0])*100+int(x.split('.')[0].split('_')[1]))

tfrecords_filename = [join(mydir, f) for f in tfrecords_filename]
print(tfrecords_filename)
filename_queue = tf.train.string_input_producer(tfrecords_filename, num_epochs=2,shuffle=True)

# Even when reading in multiple threads, share the filename
# queue.
image = read_and_decode(filename_queue,len(tfrecords_filename))

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

def load_vocab():
    #mean padding, BOS, EOS, and OOV
    vocab = u'''␀␂␃⁇ ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789().?!,:'-`;'''
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

char2idx, idx2char = load_vocab()

with tf.Session()  as sess:
    
    sess.run(init_op)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        # Let's read off 3 batches just for example
        for i in xrange(6):

            imgs = sess.run([image])
            #print(np.array(imgs).shape)
            ret= []
            for j,img in enumerate(imgs[0][0]):  
                print(img.shape)
                for row in img:
                    x = [idx2char.get(idx, "").encode('utf-8') for idx in row]
                    print(x)
                 
            
    except Exception,e:
         coord.request_stop(e)        
    finally:
        coord.request_stop()
        coord.join(threads)