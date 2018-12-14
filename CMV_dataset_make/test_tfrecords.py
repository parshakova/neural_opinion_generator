# coding: utf-8
from __future__ import print_function 
import sugartensor as tf
import time
import numpy as np
from datetime import datetime
import codecs, re, os
import nltk.data, csv
from os import listdir
from os.path import isfile, join

from op_util_classes_novan import LSTMCell, ConvLSTMCell
from LSTMCell import BasicLSTMCell2

from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops, tensor_array_ops, io_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.contrib.slim.python.slim.data import parallel_reader
from tensorflow.python.client import device_lib


# bucketing and saving states

tf.sg_verbosity(10)
batch_steps = 0
counter = 0
local_device_protos = device_lib.list_local_devices()
_gpus = len([x.name for x in local_device_protos if x.device_type == 'GPU'])
num_gpus =  max(_gpus, 1)

class Hp:
    batch_size = 2 # batch size
    hd = 650 # hidden dimension
    c_maxlen = 150 # Maximum sentence length
    w_maxlen = 25
    char_vocab = u'''␀␂␃⁇ abcdefghijklmnopqrstuvwxyz0123456789().?!,:'-;'''
    char_vs = len(char_vocab) 
    par_maxlen = [3]
    num_blocks = 3     # dilated blocks
    keep_prob = tf.placeholder(tf.float32)
    w_emb_size = 300
    num_gpus =  num_gpus
    rnn_hd = 300

def read_and_decode(filename_queue,batch_size):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'word_opinion': tf.FixedLenFeature([Hp.w_maxlen*6], dtype=tf.int64, default_value=[-1]*Hp.w_maxlen*6),
        'char_opinion': tf.FixedLenFeature([], tf.string)
    })

    char_opinion = tf.decode_raw(features['char_opinion'], tf.uint8)    
    height = tf.cast(features['height'], tf.int32) 
    word_opinion = tf.cast(features['word_opinion'], tf.int32)  

    char_opinion = tf.reshape(char_opinion, tf.stack([6, Hp.c_maxlen]) ) 
    word_opinion = tf.reshape(word_opinion, tf.stack([6, Hp.w_maxlen])) 
    print("word ",word_opinion.get_shape().as_list(), "char ",char_opinion.get_shape().as_list())

    words, chars = tf.train.shuffle_batch( [word_opinion, char_opinion],
                                                 batch_size=batch_size,
                                                 capacity=2*batch_size,
                                                 num_threads=1,
                                                 min_after_dequeue=batch_size)
    
    return (words, chars)


mydir = 'tfrc150char_wrd'  
files = [f for f in listdir(mydir) if isfile(join(mydir, f))]
tfrecords_filename = []
for fl in files:
    if int(fl.split('.')[0].split('_')[1]) in Hp.par_maxlen:
        tfrecords_filename.append(fl)
tfrecords_filename = sorted(tfrecords_filename, key=lambda x: int(x.split('.')[0].split('_')[0])*100+int(x.split('.')[0].split('_')[1]))
tfrecords_filename = [join(mydir, 'short_3.tfrecords')]

c = 0
for fn in tfrecords_filename:
  for record in tf.python_io.tf_record_iterator(fn):
     c += 1

print(tfrecords_filename)
num_epochs = 1
filename_queue = tf.train.string_input_producer(tfrecords_filename, num_epochs=num_epochs,shuffle=False,capacity=1)

(words,chars) = read_and_decode(filename_queue, Hp.batch_size * Hp.num_gpus)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)
    wo = sess.run(words)
print(wo)
