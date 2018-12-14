"""Converts symbol csv to indexed one """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tarfile
import os.path
from os import listdir
from os.path import isfile, join
import json
import re, bz2, random, csv
import nltk.data, csv
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
import numpy as np

import argparse
import os
import sys, collections
from tqdm import tqdm
import tensorflow as tf



class flag:
    def __init__(self,fname):
        if not os.path.exists(fname):
            os.makedirs(fname)
        self.directory = fname

from tensorflow.contrib.learn.python.learn.datasets import mnist

def _bytes_feature(value):
    ret = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    #print(ret.ByteSize())
    return ret

def _int64_feature(value):
    #print(np.array(value).shape)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def convert_to1(ds_path,name):
    """Converts a dataset to tfrecords."""
   
    with open(ds_path, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        filename = os.path.join(FLAGS.directory, name + '.tfrecords')
        print('Writing', filename)
        ln = 0
        writer = tf.python_io.TFRecordWriter(filename)
        for index, paragraphs in enumerate(reader):
            ln = index
            sents = paragraphs[0].split("|")
            data = []
            for snt in sents:
                data.append(map(int,snt.split(" ")))
            flag=False
            for kelem in data:           
                    kk = len(kelem)
                    if kk !=200: 
                        flag = True
                        print(index, kk)
            if flag:
                print(data)    
                #print(sents)
                continue  
            data = np.array(data).astype(np.uint8)
            #print(index,data.shape)
            image_raw = data.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(data.shape[0]),
                'image_raw': _bytes_feature(image_raw)}))
                
            writer.write(example.SerializeToString())          
    print(ln)
                
    writer.close()
              
#FLAGS = flag('tfrc200eos')

def main1():
    # Get the data.
    mydir = 'datasets/changeMV/csv198/numb198'
    ffiles = [f for f in listdir(mydir) if isfile(join(mydir, f))]
    files = []
    for el in ffiles:
        if el.split('.')[0].split('_')[0].isdigit():
            files.append(el)
    onlyfiles = sorted(files, key=lambda x: int(x.split('.')[0].split('_')[0])*100+int(x.split('.')[0].split('_')[1]))
    print(onlyfiles)
    for i,el in enumerate(onlyfiles):
        path = os.path.join(mydir, el)
        print(path, el.split('.')[0])
        convert_to(path,el.split('.')[0])

def convert_topairs(ds_path,name):
    """Converts a dataset to tfrecords."""   
    with open(ds_path, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        filename = os.path.join(FLAGS.directory, name + '.tfrecords')
        print('Writing', filename)
        ln = 0
        writer = tf.python_io.TFRecordWriter(filename)
        for index, paragraphs in enumerate(reader):
            ln = index
            datas = []
            for i in range(len(paragraphs)):
                sents = paragraphs[i].split("|")
                data = []
                for snt in sents:
                    data.append(map(int,snt.split(" ")))
                #data[-1].extend([0,0])
                flag=False
                for kelem in data:           
                        kk = len(kelem)
                        if kk !=200: 
                            flag = True
                            print(index, kk)
                datas.append(data)
            #if flag:
                #print(datas)   
                #continue  
            data1 = np.array(datas[0]).astype(np.uint8)
            statement = data1.tostring()
            data2 = np.array(datas[1]).astype(np.uint8)
            response = data2.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height1': _int64_feature(data1.shape[0]),
                'statement': _bytes_feature(statement),
                'height2': _int64_feature(data2.shape[0]),
                'response': _bytes_feature(response)}))
                
            writer.write(example.SerializeToString())          
    print(ln)
                
    writer.close()

def convert_topairssolo(ds_path,name):
    """Converts a dataset to tfrecords.""" 
    cmaxlen = 150
    wmaxlen = 25  
    num_sents = 3
    with open(ds_path, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        filename = os.path.join(FLAGS.directory, name + '.tfrecords')
        print('Writing', filename)
        ln = 0
        writer = tf.python_io.TFRecordWriter(filename)
        for index, paragraphs in tqdm(enumerate(reader)):
            ln = index
            if index > 50:
                print("FINISHED 50 samples")
                break
            datas = []
            for i in range(len(paragraphs)): # statement and response in character level AND statement and response in word level
                sents = paragraphs[i].split("|")
                data = []
                for snt in sents:
                    if i <2: # statement and response in character level
                        snt = re.sub(r'(5)+(?=(\s5))',"",snt)
                    inpt = re.sub(r'(3)+(?=(\s3))',"",snt).strip()
                    inpt = re.sub(r'(4)+(?=(\s4))',"",inpt).strip()
                    inpt=map(int,re.sub(r"\s{2,}", " ", inpt).split(" "))
                    if i<2: 
                        inpt += [0] * (cmaxlen - len(inpt))
                        inpt = inpt[:cmaxlen]
                    else: 
                        inpt += [0] * (wmaxlen - len(inpt))
                        inpt = inpt[:wmaxlen]
                    #print(inpt, snt)
                    data.append(inpt)

                flag=False
                if i<2:
                    for kelem in data:           
                            kk = len(kelem)
                            if kk !=cmaxlen: 
                                flag = True
                                print(index, kk)
                    if len(data)<num_sents:
                        for lj in range(num_sents - len(data)):
                            data.append([0]*(cmaxlen))
                elif i>=2:
                    if len(data)<num_sents:
                        for lj in range(num_sents - len(data)):
                            data.append([0]*(wmaxlen))
                datas.append(data)
            if flag:
                print(datas)   
                continue  

            data1 = np.array(datas[0]).astype(np.uint8)
            data2 = np.array(datas[1]).astype(np.uint8)
            datac = np.concatenate((data1, data2),axis=0)
            statement = datac.tostring()

            data1 = np.array(datas[2]).astype(np.int64)
            data2 = np.array(datas[3]).astype(np.int64)
            data = np.concatenate((data1, data2),axis=0)
            w_statement = np.reshape(data, -1)

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature([datac.shape[0]]),
                'char_opinion': _bytes_feature(statement),
                'word_opinion': _int64_feature(w_statement)}))
                
            writer.write(example.SerializeToString())          
    print(ln)
                
    writer.close()
              
FLAGS = flag('tfrc150char_wrd')

def mainpairs():
    # Get the data.
    mydir = 'datasets/changeMV/csveq147/numb147_0621'
    ffiles = [f for f in listdir(mydir) if isfile(join(mydir, f))]
    files = []
    for el in ffiles:
        if el.split('.')[0].split('_')[0].isdigit():
            num1 =int(el.split('.')[0].split('_')[0])
            num2 = int(el.split('.')[0].split('_')[1])
            if num2 in {3}:
                files.append(el)

    onlyfiles = sorted(files, key=lambda x: int(x.split('.')[0].split('_')[0])*100+int(x.split('.')[0].split('_')[1]))
    print(onlyfiles)
    for i,el in enumerate(onlyfiles):
        path = os.path.join(mydir, el)
        print(path, el.split('.')[0])
        convert_topairssolo(path,"schshort_infer3")
        break


mainpairs()


