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

from LSTMCell import BasicLSTMCell2

from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops, tensor_array_ops, io_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.contrib.slim.python.slim.data import parallel_reader
from tensorflow.python.client import device_lib
from nltk.translate.bleu_score import sentence_bleu

import scipy.spatial.distance as sd
from skip_thoughts import configuration
from skip_thoughts import encoder_manager
import tensorflow.contrib.slim as slim


tf.sg_verbosity(10)
batch_steps = 0
local_device_protos = device_lib.list_local_devices()
_gpus = len([x.name for x in local_device_protos if x.device_type == 'GPU'])
num_gpus =  max(_gpus, 1)

class Hp:
    batch_size = 22 # batch size
    hd = 400 # hidden dimension
    c_maxlen = 150 # Maximum sentence length
    w_maxlen = 25
    char_vocab = u'''␀␂␃⁇N abcdefghijklmnopqrstuvwxyz'''
    char_vs = len(char_vocab) 
    par_maxlen = [3]
    num_blocks = 3     # dilated blocks
    keep_prob = tf.placeholder(tf.float32)
    w_emb_size = 300
    num_gpus =  num_gpus
    rnn_hd = 300
    sample = 0.25
    num_to_decode = 3

#len 150 OrderedDict([(3, 365703), (6, 391330), (10, 300479), (15, 192515), (20, 99041), (30, 88487), (40, 32657)])
input_datalen = {3:24000}
for el in Hp.par_maxlen:
    batch_steps += input_datalen[el]  #for batch of 16    #1457401 1394000
batch_steps //= Hp.batch_size
num_epochs = 10000



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
    #print("word ",word_opinion.get_shape().as_list(), "char ",char_opinion.get_shape().as_list())

    words, chars = tf.train.shuffle_batch( [word_opinion, char_opinion],
                                                 batch_size=batch_size,
                                                 capacity=3*batch_size,
                                                 num_threads=2,
                                                 min_after_dequeue=2*batch_size)
    
    return (words, chars)


# residual block
@tf.sg_sugar_func
def sg_res_block(tensor, opt):
    # default rate

    opt += tf.sg_opt(size=3, rate=1, causal=False, is_first=False)

    # input dimension
    in_dim = tensor.get_shape().as_list()[-1]
    
    with tf.sg_context(dev = opt.dev,reuse=opt.reuse_vars):
        #reduce dim
        input_ = (tensor
                  .sg_bypass_gpus(act='leaky_relu', ln=(not opt.is_first),name = "relu_"+opt.name)
                  .sg_conv1d_gpus(size=1, dim=in_dim/2, act='leaky_relu', ln=opt.causal,name = "convi_"+opt.name))

        # 1xk conv dilated
        out = input_.sg_aconv1d_gpus(size=opt.size, rate=opt.rate, causal=opt.causal, act='leaky_relu', ln=opt.causal, name="aconv_"+opt.name)

        # dimension recover and residual connection
        out = out.sg_conv1d_gpus(size=1, dim=in_dim,name = "convo_"+opt.name) + tensor

    #print(out.name)

    return out

# inject residual multiplicative block
tf.sg_inject_func(sg_res_block)


@tf.sg_layer_func_gpus
def sg_quasi_conv1d(tensor, opt):

    opt += tf.sg_opt(is_enc=False, causal=True)

    # Split into H and H_zfo
    H = tensor[:Hp.batch_size]
    H_z = tensor[Hp.batch_size:2*Hp.batch_size]
    H_f = tensor[2*Hp.batch_size:3*Hp.batch_size]
    H_o = tensor[3*Hp.batch_size:]
    if opt.is_enc:
        H_z, H_f, H_o = 0, 0, 0
    
    # Convolution and merging
    with tf.sg_context(size=opt.size,act="linear", causal=opt.causal and (not opt.is_enc), dev = opt.dev, reuse=opt.reuse_vars):
        Z = H.sg_aconv1d_gpus(name = "aconvz_"+opt.name) + H_z # (b, seqlen, hd)
        F = H.sg_aconv1d_gpus(name = "aconvf_"+opt.name) + H_f # (b, seqlen, hd)
        O = H.sg_aconv1d_gpus(name = "aconvo_"+opt.name) + H_o # (b, seqlen, hd)

    # Activation
    with tf.sg_context(dev = opt.dev, reuse=opt.reuse_vars):
      Z = Z.sg_bypass_gpus(act="tanh",name = "tanhz_"+opt.name) # (b, seqlen, hd)
      F = F.sg_bypass_gpus(act="sigmoid",name = "sigmf_"+opt.name) # (b, seqlen, hd)
      O = O.sg_bypass_gpus(act="sigmoid",name = "sigmo_"+opt.name) # (b, seqlen, hd)
    
    # Concat
    ZFO = tf.concat([Z, F, O],0)
    
    return ZFO # (batch*3, seqlen, hiddim)

# injection
tf.sg_inject_func(sg_quasi_conv1d)
   

@tf.sg_rnn_layer_func_gpus
def sg_quasi_rnn(tensor, opt):
    # Split
    if opt.att:
        H, Z, F, O = tf.split(axis=0, num_or_size_splits=4, value=tensor) # (b, seqlen, hd) for all
    else:
        Z, F, O = tf.split(axis=0, num_or_size_splits=3, value=tensor) # (b, seqlen, hd) for all
    
    # step func
    def step(z, f, o, c):
        '''
        Runs fo-pooling at each time step
        '''
        c = f * c + (1 - f) * z
        
        if opt.att: # attention
            a = tf.nn.softmax(tf.einsum("ijk,ik->ij", H, c)) # alpha. (b, seqlen) 
            k = (a.sg_expand_dims() * H).sg_sum(axis=1) # attentional sum. (b, seqlen) 
            h = o * (k.sg_dense_gpus(act="linear",name = "k%d_%s"%(t,opt.name),dev = opt.dev,reuse=opt.reuse_vars)\
                + c.sg_dense_gpus(act="linear",name = "c%d_%s"%(t,opt.name),dev = opt.dev,reuse=opt.reuse_vars))
        else:
            h = o * c
        
        return h, c # hidden states, (new) cell memories
    
    # Do rnn loop
    c, hs = 0, []
    timesteps = tensor.get_shape().as_list()[1]
    for t in range(timesteps):
        z = Z[:, t, :] # (b, hd)
        f = F[:, t, :] # (b, hd)
        o = O[:, t, :] # (b, hd)

        # apply step function
        h, c = step(z, f, o, c) # (b, hd), (b, hd)
        
        # save result
        hs.append(h.sg_expand_dims(axis=1))
    
    # Concat to return        
    H = tf.concat(hs, 1) # (b, seqlen, hd)
    
    if opt.is_enc:
        H_z = tf.tile((h.sg_dense_gpus(act="linear",name = "z_%s"%(opt.name),dev=opt.dev,reuse=opt.reuse_vars).sg_expand_dims(axis=1)), [1, timesteps, 1])
        H_f = tf.tile((h.sg_dense_gpus(act="linear",name = "f_%s"%(opt.name),dev=opt.dev,reuse=opt.reuse_vars).sg_expand_dims(axis=1)), [1, timesteps, 1])
        H_o = tf.tile((h.sg_dense_gpus(act="linear",name = "o_%s"%(opt.name),dev=opt.dev,reuse=opt.reuse_vars).sg_expand_dims(axis=1)), [1, timesteps, 1])   
        concatenated = tf.concat(axis=0, values=[H, H_z, H_f, H_o]) # (b*4, seqlen, hd)
        
        return concatenated
    else:
        return H # (b, seqlen, hd)

# injection
tf.sg_inject_func(sg_quasi_rnn)

def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        with tf.device('/cpu:0'):
            matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
            bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


def tower_loss_xe(words,chars, scope, rnn_cell, dec_cell, word_emb, out_reuse_vars=False, dev='/cpu:0', save_activ = False):
    out_rvars = out_reuse_vars
    # make embedding matrix for source and target
    with tf.device(dev):
        with tf.variable_scope('embatch_size',reuse=out_reuse_vars):
          # (vocab_size, latent_dim)
          #emb_x = tf.get_variable('emb_x', dtype=tf.float32, shape=[Hp.vs, Hp.hd],initializer=tf.truncated_normal_initializer())
          emb_char = tf.sg_emb(name='emb_char', voca_size=Hp.char_vs, dim=Hp.hd, dev = dev)
          emb_word = tf.sg_emb(name='emb_word', emb=word_emb, voca_size=Hp.word_vs, dim=300, dev = dev)
          #emb_y = tf.get_variable('emb_y', dtype=tf.float32, shape=[Hp.vs, Hp.hd],initializer=tf.truncated_normal_initializer())

    chars = tf.cast(chars, tf.int32)

    time = tf.constant(0)
    losses_init = tf.constant(0.0)

    inputs = tf.transpose(chars, perm=[1, 0, 2])
    input_ta = tensor_array_ops.TensorArray(tf.int32, size=tf.shape(chars)[1], dynamic_size=True,clear_after_read = True)
    chars_sent = input_ta.unstack(inputs) #each element is (batch, sentlen)

    winputs = tf.transpose(words, perm=[1, 0, 2])
    winput_ta = tensor_array_ops.TensorArray(tf.int32, size=tf.shape(words)[1], dynamic_size=True,clear_after_read = True)
    words_sent = winput_ta.unstack(winputs)

    resp_steps = tf.shape(chars)[1] # number of sentences in paragraph
    statm_steps = resp_steps // 2

    rnn_state = rnn_cell.zero_state(Hp.batch_size, tf.float32)      #rnn_cell.rnn_state, rnn_cell.rnn_h

    summaries_act = []
    
    # -------------------------------------------- STATEMENT ENCODING -----------------------------------------------

    def rnn_cond_stat(time, rnn_state):
      return tf.less(time, statm_steps-1)    
      
    def rnn_body_stat(time, rnn_state):
        ch = chars_sent.read(time)
        ch =  tf.reverse_sequence(input=ch, seq_lengths=[Hp.c_maxlen]*Hp.batch_size, seq_dim=1)
        reuse_vars = out_reuse_vars

    	# --------------------------   BYTENET ENCODER   --------------------------

        with tf.variable_scope('encoder'):
            # embed table lookup
            enc = ch.sg_lookup(emb=emb_char)  #(batch, sentlen, latentdim)
            # loop dilated conv block            
            for i in range(Hp.num_blocks):
                enc = (enc
                       .sg_res_block(size=5, rate=1, name = "enc1_%d"%(i), is_first=True,reuse_vars=reuse_vars,dev=dev)
                       .sg_res_block(size=5, rate=2, name = "enc2_%d"%(i),reuse_vars=reuse_vars,dev=dev)
                       .sg_res_block(size=5, rate=4, name = "enc4_%d"%(i),reuse_vars=reuse_vars,dev=dev)
                       .sg_res_block(size=5, rate=8, name = "enc8_%d"%(i),reuse_vars=reuse_vars, dev=dev)
                       .sg_res_block(size=5, rate=16, name = "enc16_%d"%(i),reuse_vars=reuse_vars,dev=dev))
            byte_enc = enc
    	# --------------------------   QCNN + QPOOL ENCODER #1  --------------------------

            with tf.variable_scope('quazi'):
                d_byte_enc = tf.nn.dropout(byte_enc, Hp.keep_prob)

                #quasi cnn layer ZFO  [batch * 3, seqlen, dim2 ]
                conv = d_byte_enc.sg_quasi_conv1d(is_enc=True, size=4, name = "qconv_1", dev = dev, reuse_vars=reuse_vars)
                # c = f * c + (1 - f) * z, h = o*c [batch * 4, seqlen, hd]
                pool0 = conv.sg_quasi_rnn(is_enc=False, att=False, name="qrnn_1", reuse_vars=reuse_vars, dev=dev)

                qpool_last = pool0[:,-1,:]


        # --------------------------   MAXPOOL along time dimension   --------------------------

        inpt_maxpl = tf.expand_dims(byte_enc, 1) # [batch, 1, seqlen, channels]
        maxpool = tf.nn.max_pool(inpt_maxpl, [1, 1, Hp.c_maxlen, 1], [1, 1, 1, 1], 'VALID')
        maxpool = tf.squeeze(maxpool, [1, 2])

        # --------------------------   HIGHWAY   --------------------------

        concat = qpool_last + maxpool
        #concat = qpool_last + maxpool
        with tf.variable_scope('highway',reuse=reuse_vars):
            input_lstm = highway(concat, concat.get_shape()[-1], num_layers=1)

    	# --------------------------   CONTEXT LSTM  --------------------------
        input_lstm = tf.nn.dropout(input_lstm, Hp.keep_prob)

        with tf.variable_scope('contx_lstm', reuse= reuse_vars):
            output, rnn_state = rnn_cell(input_lstm, rnn_state)        

        return (time+1, rnn_state)



    loop_vars_stat = [time, rnn_state]

    time, rnn_state = tf.while_loop(rnn_cond_stat, rnn_body_stat, loop_vars_stat, swap_memory=False)

    reuse_vars_enc = True
    reuse_vars_dec = out_reuse_vars
    parSA, parMX, parGT, sample_logprobs = tf.ones([Hp.batch_size, 1], tf.int32), tf.ones([Hp.batch_size, 1], tf.int32),tf.ones([Hp.batch_size, 1], tf.int32), tf.zeros((1), tf.float32) #tf.zeros((1, Hp.w_maxlen-1), tf.float32)


    # -------------------------------------------- RESPONSE DECODING  ----------------------------------------------

    def rnn_cond_response(time, rnn_state, losses, parSA, parMX, parGT, sample_logprobs):
      return tf.less(time, resp_steps-1)    
      
    def rnn_body_response(time, rnn_state, losses, parSA, parMX, parGT, sample_logprobs):
        ch = chars_sent.read(time)
        wrd = words_sent.read(time+1)
        ch =  tf.reverse_sequence(input=ch, seq_lengths=[Hp.c_maxlen]*Hp.batch_size, seq_dim=1)
        reuse_vars = reuse_vars_enc

        # --------------------------   BYTENET ENCODER   --------------------------

        with tf.variable_scope('encoder'):
            # embed table lookup
            enc = ch.sg_lookup(emb=emb_char)  #(batch, sentlen, latentdim)
            # loop dilated conv block            
            for i in range(Hp.num_blocks):
                enc = (enc
                       .sg_res_block(size=5, rate=1, name = "enc1_%d"%(i), is_first=True,reuse_vars=reuse_vars,dev=dev)
                       .sg_res_block(size=5, rate=2, name = "enc2_%d"%(i),reuse_vars=reuse_vars,dev=dev)
                       .sg_res_block(size=5, rate=4, name = "enc4_%d"%(i),reuse_vars=reuse_vars,dev=dev)
                       .sg_res_block(size=5, rate=8, name = "enc8_%d"%(i),reuse_vars=reuse_vars, dev=dev)
                       .sg_res_block(size=5, rate=16, name = "enc16_%d"%(i),reuse_vars=reuse_vars,dev=dev))
            byte_enc = enc
        # --------------------------   QCNN + QPOOL ENCODER #1  --------------------------

            with tf.variable_scope('quazi'):
                d_byte_enc = tf.nn.dropout(byte_enc, Hp.keep_prob)

                #quasi cnn layer ZFO  [batch * 3, seqlen, dim2 ]
                conv = d_byte_enc.sg_quasi_conv1d(is_enc=True, size=4, name = "qconv_1", dev = dev, reuse_vars=reuse_vars)
                # c = f * c + (1 - f) * z, h = o*c [batch * 4, seqlen, hd]
                pool0 = conv.sg_quasi_rnn(is_enc=False, att=False, name="qrnn_1", reuse_vars=reuse_vars, dev=dev)

                qpool_last = pool0[:,-1,:]


        # --------------------------   MAXPOOL along time dimension   --------------------------

        inpt_maxpl = tf.expand_dims(byte_enc, 1) # [batch, 1, seqlen, channels]
        maxpool = tf.nn.max_pool(inpt_maxpl, [1, 1, Hp.c_maxlen, 1], [1, 1, 1, 1], 'VALID')
        maxpool = tf.squeeze(maxpool, [1, 2])

        # --------------------------   HIGHWAY   --------------------------

        concat = qpool_last + maxpool
        #concat = qpool_last+ maxpool
        with tf.variable_scope('highway',reuse=reuse_vars):
            input_lstm = highway(concat, concat.get_shape()[-1], num_layers=1)

        # --------------------------   CONTEXT LSTM  --------------------------

        input_lstm = tf.nn.dropout(input_lstm, Hp.keep_prob)

        with tf.variable_scope('contx_lstm', reuse= reuse_vars):
            output, rnn_state = rnn_cell(input_lstm, rnn_state) 

        reuse_vars = reuse_vars_dec

        w_src = wrd.sg_lookup(emb=emb_word)

        y = wrd[:, 1:] # ground truth for decoding

        w_input = [tf.squeeze(x, [1]) for x in tf.split(w_src, Hp.w_maxlen, 1)[:-1]]
        dec_stateCE = rnn_state #dec_cell.zero_state(Hp.batch_size, tf.float32)
        dec_outCE = []
        dec_stateSA = rnn_state
        dec_outSA = []
        dec_stateMX = rnn_state
        dec_outMX = []
        logprob = []

        sample_d =  tf.contrib.distributions.Categorical(logits=tf.log([[Hp.sample, 1-Hp.sample]])) 

        for idx, w_in in enumerate(w_input):
            if idx>0:
                d_outSA = tf.squeeze(tf.multinomial(tf.squeeze(tf.expand_dims(d_outSA, 1).sg_conv1d_gpus(size=1, dim=Hp.word_vs, name="out_conv", act="linear", dev=dev, reuse=True),1), 1),1).sg_lookup(emb=emb_word)
                d_outMX = tf.squeeze(tf.expand_dims(d_outMX,1).sg_conv1d_gpus(size=1, dim=Hp.word_vs, name="out_conv", act="linear", dev=dev, reuse=True),1).sg_argmax().sg_lookup(emb=emb_word)
                w_in = tf.cond(tf.equal(tf.squeeze(sample_d.sample(1)), 0), lambda:d_outSA, lambda:w_in)
            else:
                d_outSA, d_outMX = w_in, w_in
            

            with tf.variable_scope('dec_lstm', reuse= idx>0 or reuse_vars):
                dec_stateCE = tf.contrib.rnn.LSTMStateTuple( 
                        c = dec_stateCE.c,
                        h = dec_stateCE.h)
                d_outCE, dec_stateCE = dec_cell(tf.nn.dropout(w_in, Hp.keep_prob), dec_stateCE)

            with tf.variable_scope('dec_lstm', reuse= True):
                dec_stateSA = tf.contrib.rnn.LSTMStateTuple( 
                        c = dec_stateSA.c,
                        h = dec_stateSA.h)
                d_outSA, dec_stateSA = dec_cell(tf.nn.dropout(d_outSA, Hp.keep_prob), dec_stateSA)
                dec_stateMX = tf.contrib.rnn.LSTMStateTuple( 
                        c = dec_stateMX.c,
                        h = dec_stateMX.h)
                d_outMX, dec_stateMX = dec_cell(tf.nn.dropout(d_outMX, Hp.keep_prob), dec_stateMX)

            dec_outCE.append(d_outCE)
            
            valSA = tf.nn.log_softmax(tf.squeeze(tf.expand_dims(d_outSA,1).sg_conv1d_gpus(size=1, dim=Hp.word_vs, name="out_conv", act="linear", dev=dev, reuse=idx>0 or reuse_vars),1), dim=-1)

            indSA = tf.cast(tf.squeeze(tf.multinomial(valSA,1),1), tf.int32)

            batch_pos = tf.range(Hp.batch_size, dtype=tf.int32)
            g_indSA = tf.stack([batch_pos, indSA], axis=1)

            dec_outSA.append(indSA)
            logprob.append(tf.gather_nd(valSA, g_indSA))
            #print("logprob input shape ", tf.gather_nd(valSA, g_indSA).get_shape())
            dec_outMX.append(tf.cast(tf.squeeze(tf.expand_dims(d_outMX,1).sg_conv1d_gpus(size=1, dim=Hp.word_vs, name="out_conv", act="linear", dev=dev, reuse=idx>0 or True).sg_argmax(),1),tf.int32))
            

        dec_outCE = tf.stack(dec_outCE, 1)      # (batch x w_maxlen x rnn_hd)  cross-entropy
        decSA = tf.squeeze(tf.stack(dec_outSA, 1))      # (batch x w_maxlen)  sampling multinomial
        decMX = tf.squeeze(tf.stack(dec_outMX, 1))      # (batch x w_maxlen)  argmax from greedy decoding
        logprob = tf.squeeze(tf.stack(logprob, 1))      # (batch x w_maxlen)  logprobs on sampled indexes
        print("logprob shape ", logprob.get_shape())

        decCE = dec_outCE.sg_conv1d_gpus(size=1, dim=Hp.word_vs, name="out_conv", act="linear", dev=dev, reuse=True)
        print("dec shape ", decCE.get_shape().as_list(), "d_outSA shape ", decSA.get_shape().as_list())

        # argmin to find index from which padding starts
        indsSA =  tf.cast(tf.argmin(decSA, 1), tf.int32) # (batch)
        indsMX =  tf.cast(tf.argmin(decMX, 1),tf.int32)
        indsGT =  tf.cast(tf.argmin(y, 1), tf.int32)

        sentSA = tf.unstack(decSA, axis=0)  # list = batch * [w_maxlen]
        sentMX = tf.unstack(decMX, axis=0)
        logprob = tf.unstack(logprob, axis=0)
        sentGT = tf.unstack(y, axis=0) # ground truth sentence

        parSA = tf.concat([parSA, sentSA], 1)
        parMX = tf.concat([parMX, sentMX], 1)
        parGT = tf.concat([parGT, sentGT], 1)
        print("sentSA shape ", sentSA[1].get_shape().as_list(), "indsSA shape ", indsSA.get_shape().as_list(), "logprob shape ", logprob[1].get_shape().as_list())
        print("sample_logprobs shape ", sample_logprobs.get_shape().as_list(), "parSA shape ", parSA.get_shape().as_list(), "parGT shape ", parGT.get_shape().as_list())

        to_concat =[[] for _ in range(Hp.batch_size)]
        for b in range(Hp.batch_size):
            to_concat[b] = tf.cond(tf.equal(sentSA[b][indsSA[b]],0), lambda: tf.reduce_mean(logprob[b][:indsSA[b]]), lambda: tf.reduce_mean(logprob[b][:]))
        to_concat = tf.stack(to_concat)
        print("size to_cont ", to_concat.get_shape().as_list())

        sample_logprobs = tf.concat([sample_logprobs, to_concat], 0)

        ce_array = decCE.sg_ce(target=y, mask=True, name = "cross_entropy_example")        

        cross_entropy_mean = tf.reduce_mean(ce_array, name='cross_entropy_batches')
        tf.sg_summary_loss(cross_entropy_mean,"cross_entropy")
        
        losses = tf.add_n([losses,cross_entropy_mean], name='add_losses_batch')

        return (time+1, rnn_state, losses, parSA, parMX, parGT, sample_logprobs)


    time = statm_steps - 1

    loop_vars_response = [time, rnn_state, losses_init, parSA, parMX, parGT, sample_logprobs]
    shape_invs_in = [time.get_shape(), tf.contrib.rnn.LSTMStateTuple(tf.TensorShape([None,None]),tf.TensorShape([None,None])),  losses_init.get_shape(), \
                        tf.TensorShape([Hp.batch_size, None]), tf.TensorShape([Hp.batch_size, None]), tf.TensorShape([Hp.batch_size, None]), tf.TensorShape([None])]

    time, rnn_state, losses, parSA, parMX, parGT, sample_logprobs = tf.while_loop\
                      (rnn_cond_response, rnn_body_response, loop_vars_response, shape_invariants=shape_invs_in,swap_memory=False)


    variables   = tf.trainable_variables()
    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in variables if 'b' not in v.name ]) * 0.0000015

    losses = 0.5*(losses + tf.reduce_mean(lossL2))

    tf.add_to_collection('losses', losses)

    if save_activ:
        summaries_act.append(tf.summary.histogram("lstm/rnn_state", rnn_state))
        summaries_act.append(tf.summary.histogram("losses", losses))
        summaries_act.append(tf.summary.scalar("losses", losses))
    
    # Calculate the total loss for the current tower.
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
    tf.sg_summary_loss(total_loss, "losses")

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    
    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)

    sample_logprobs = sample_logprobs[1:]
    return (total_loss, summaries_act, parSA, parMX, parGT, sample_logprobs)


def tower_loss_rl2(sample_logprobs, rewards):
    # sample_logprobs shape        (batch_size * num_to_decode,  num_w_in_sents)
    # rewards shape                (batch_size, num_to_decode)
    rewards = tf.tile((tf.reshape(rewards,[-1]).sg_expand_dims(axis=1)), [1, Hp.w_maxlen-1])
    print('rewards shape ', rewards.get_shape().as_list(), 'sample_logprobs shape ', sample_logprobs.get_shape().as_list())
    scores = tf.reduce_sum(tf.multiply(sample_logprobs, rewards), 1) / tf.count_nonzero(sample_logprobs, 1, dtype=tf.float32)
    print('scores shape ', scores.get_shape().as_list())

    return scores


def tower_loss_rl(sample_logprobs, rewards):
    # sample_logprobs shape        (batch_size * num_to_decode)
    # rewards shape                (num_to_decode, batch_size)
    rewards = tf.reshape(rewards,[-1])
    print('rewards shape ', rewards.get_shape().as_list(), 'sample_logprobs shape ', sample_logprobs.get_shape().as_list())
    scores = tf.reduce_mean(tf.multiply(sample_logprobs, rewards))
    print('scores shape ', scores.get_shape().as_list())

    return scores


def compute_rewards_bleu(parSA, parMX, parGT):

    def compute_rewards_aid(candid, reference):

        def f1_score(p, r):
            return 2*p*r/(p+r)

        def rouge(h, r):
            h1 = set(h)
            r1 = set(r)
            return len(r1 & h1)*1.0/len(r1)
        scores = np.zeros((Hp.batch_size))

        for b in range(Hp.batch_size):
            cand = candid[b][np.nonzero(candid[b])]
            refer = reference[b][np.nonzero(reference[b])]
            rR  = rouge(cand, refer)        #recall
            bR = sentence_bleu([map(str, refer)], map(str, cand))  # precision;   references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25)
            scores[b] = f1_score(bR, rR)

        return scores
    # output num_gpus * (Hp.batch_size)
    rewards = []
    for g in range(Hp.num_gpus):
        # remove paddings
        rSA = compute_rewards_aid(parSA[g], parGT[g])  # reward from sampled
        rMX = compute_rewards_aid(parMX[g], parGT[g])
        rewards.append(rSA - rMX)

    return rewards

def compute_rewards_bleu_sents2(parSA, parMX, parGT):

    def compute_rewards_aid(candid, reference):

        def f1_score(p, r):
            if (p+r) != 0:
                return 2*p*r/(p+r)
            else:
                return 0

        def rouge(h, r):
            h1 = set(h)
            r1 = set(r)
            if len(r1) != 0:
                return len(r1 & h1)*1.0/len(r1)
            else: 
                return 0

        scores = np.zeros((Hp.batch_size, Hp.num_to_decode))
        ind = 0
        for r, c in zip(np.split(reference[:,1:],Hp.num_to_decode,1), np.split(candid[:,1:],Hp.num_to_decode,1)):
            for b in range(Hp.batch_size):
                cand = c[b][np.nonzero(c[b])]
                refer = r[b][np.nonzero(r[b])]
                rR  = rouge(cand, refer)        #recall
                bR = sentence_bleu([map(str, refer)], map(str, cand))  # precision;   references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25)
                scores[b, ind] = f1_score(bR, rR)
            ind += 1

        return scores
    # output num_gpus * (Hp.batch_size, Hp.num_to_decode)
    rewards = []
    for g in range(Hp.num_gpus):
        # remove paddings
        rSA = compute_rewards_aid(parSA[g], parGT[g])  # reward from sampled
        rMX = compute_rewards_aid(parMX[g], parGT[g])
        rewards.append(rSA - rMX)

    return rewards



def compute_rewards_bleu_sents(parSA, parMX, parGT, flag):

    def compute_rewards_aid(candid, reference):

        def f1_score(p, r):
            if (p+r) != 0:
                return 2*p*r/(p+r)
            else:
                return 0

        def rouge(h, r):
            h1 = set(h)
            r1 = set(r)
            if len(r1) != 0:
                return len(r1 & h1)*1.0/len(r1)
            else: 
                return 0

        scores = np.zeros((Hp.num_to_decode, Hp.batch_size))
        ind = 0
        for r, c in zip(np.split(reference[:,1:],Hp.num_to_decode,1), np.split(candid[:,1:],Hp.num_to_decode,1)):
            for b in range(Hp.batch_size):
                cand = c[b][np.nonzero(c[b])]
                refer = r[b][np.nonzero(r[b])]
                if len(set(refer)) != 0:
                    rR  = rouge(cand, refer)        #recall
                    bR = sentence_bleu([map(str, refer)], map(str, cand))  # precision;   references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25)
                    scores[ind, b] = f1_score(bR, rR)
            ind += 1

        return scores
    # output num_gpus * (Hp.batch_size, Hp.num_to_decode)
    rewards = []
    for g in range(Hp.num_gpus):
        # remove paddings
        rSA = compute_rewards_aid(parSA[g], parGT[g])  # reward from sampled
        rMX = compute_rewards_aid(parMX[g], parGT[g])
        rewards.append(rSA - rMX)
        if flag:
            print('rSA          rMX')
            print(rSA)
            print(rMX)

    return rewards



def compute_rewards_skipemb_sents(skip_encoder, parSA, parMX, parGT, flag):

    def compute_rewards_aid(candidate, reference):

        scores = np.zeros((Hp.num_to_decode, Hp.batch_size))
        #print(candidate.shape, reference.shape)
        ind = 0
        for r, c in zip(np.split(reference[:,1:],Hp.num_to_decode,1), np.split(candidate[:,1:],Hp.num_to_decode,1)):
            cand = skip_encoder.encode(c)
            refer = skip_encoder.encode(r)
            if flag:
                print(c,r)
            current_score = 2 - sd.cdist(cand, refer, "cosine")[0]      # 2 - perfect similarity; 0 -  worst
            scores[ind, :] = np.nan_to_num(current_score)
            ind += 1
        return scores 
    # output num_gpus * (Hp.batch_size)
    rewards = []
    for g in range(Hp.num_gpus):
        # remove paddings
        rSA = compute_rewards_aid(parSA[g], parGT[g])  # (batch_size, num_to_decode) reward from sampled
        rMX = compute_rewards_aid(parMX[g], parGT[g])
        #print(rSA, rMX)
        rewards.append(rSA - rMX)
        if flag:
            print('rSA          rMX')
            print(rSA)
            print(rMX)

    return rewards



def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, v in grad_and_vars:
          print(v.name,v.device, g) 
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def logging(ce):
    return ce

def makedir_date():    
    logfold = time.strftime("tnsrbrd/klm%dd%mm_%H%Mg"+str(Hp.num_gpus))
    if not os.path.exists(logfold):
        os.makedirs(logfold)
    else:
        logfold = time.strftime("tnsrbrd/klm%dd%mm_%H")+str(int(time.strftime("%M"))+1)+'g'+str(Hp.num_gpus+1)
        os.makedirs(logfold)
    return logfold
    

def load_vocab():
    #mean padding, BOS, EOS, and OOV
    vocab = u'''␀␂␃⁇N abcdefghijklmnopqrstuvwxyz'''
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

char2idx, idx2char = load_vocab()

def train():
    #  tf.truncated_normal_initializer(mean=-0.1, stddev=.01)
    # initializer=tf.contrib.layers.xavier_initializer()
    dev = '/cpu:0'
    with tf.device(dev):
        mydir = 'tfrc150char_wrd0704'  
        files = [f for f in listdir(mydir) if isfile(join(mydir, f))]
        
        tfrecords_filename = [join(mydir, 'short_3.tfrecords')]#[join(mydir, f) for f in tfrecords_filename]
        tfrecords_filename_inf = [join(mydir, '11_3.tfrecords')]
        
        print(tfrecords_filename)
        filename_queue = tf.train.string_input_producer(tfrecords_filename, num_epochs=num_epochs,shuffle=True,capacity=3)
        infer_queue = tf.train.string_input_producer(tfrecords_filename_inf, num_epochs=num_epochs,shuffle=True,capacity=1)

        optim = tf.train.AdamOptimizer(learning_rate=0.00004,beta1=0.9, beta2=0.99)

        # Calculate the gradients for each model tower.
        tower_grads = []
        reuse_vars = False
        with tf.variable_scope("dec_lstm") as scp:
            dec_cell = BasicLSTMCell2(Hp.w_emb_size, Hp.rnn_hd, state_is_tuple=True)
        with tf.variable_scope("contx_lstm") as scp:
            cell = BasicLSTMCell2(Hp.hd, Hp.rnn_hd, state_is_tuple=True)
            rnn_cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=Hp.keep_prob, output_keep_prob=Hp.keep_prob)

        (words,chars) = read_and_decode(filename_queue, Hp.batch_size * Hp.num_gpus)

        words_splits = tf.split(axis=0, num_or_size_splits=Hp.num_gpus, value=words)
        chars_splits = tf.split(axis=0, num_or_size_splits=Hp.num_gpus, value=chars)

        word_emb = np.loadtxt("glove300d_0704.txt")
        Hp.word_vs = word_emb.shape[0]

        rewards = [tf.placeholder(tf.float32, shape=(Hp.num_to_decode, Hp.batch_size)) for _ in range(Hp.num_gpus)]
        parSA, parMX, parGT = [], [], []

        # --------------------------------------------------------------------------------

        with tf.variable_scope(tf.get_variable_scope()) as scp:
          for i in xrange(Hp.num_gpus):
            with tf.device('/gpu:%d' % (i)):
              with tf.name_scope('%s_%d' % ("tower", i)) as scope:
                # loss for one tower
                save_activ = i ==Hp.num_gpus - 1
                (loss, summaries_act, iparSA, iparMX, iparGT, sample_logprobs) = tower_loss_xe(words_splits[i], chars_splits[i],scope,rnn_cell, dec_cell, word_emb=word_emb, out_reuse_vars=reuse_vars,dev=dev,save_activ=save_activ)
                reuse_vars = True

                parSA.append(iparSA) # (batch, num_words_in_opinion)
                parMX.append(iparMX)
                parGT.append(iparGT)

                lossRL = tf.stack([ tf.reduce_mean(- rewards[b] * sample_logprobs[b]) for b in Hp.batch_size], 0)
                losses = loss  - 0.5*tower_loss_rl(sample_logprobs, rewards[i]) 
                #losses = loss + 0.5* tf.reduce_mean(lossRL)

                single_grads = optim.compute_gradients(losses)     
                tower_grads.append(single_grads)

        # --------------------------------------------------------------------------------
        parSA = tf.stack(parSA, 0)
        parMX = tf.stack(parMX, 0)
        parGT = tf.stack(parGT, 0)

        summaries_grad = []
        summaries_var = []

        grads = average_gradients(tower_grads)

        capped_gvs = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in grads] 
        train_op = optim.apply_gradients(capped_gvs)
        
        for grad, var in single_grads:
                if grad.device == '/device:GPU:%d'%(Hp.num_gpus-1):
                    print(" summ "+ var.op.name+" "+grad.device)
                    summaries_grad.append(tf.summary.histogram(var.op.name + '/grad', grad))
                print(grad.device)
                if var.device == '/device:CPU:0':
                    print(" summ "+ var.op.name)
                    summaries_var.append(tf.summary.histogram(var.op.name, var))
        
        
        trainable_variables = []
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            trainable_variables.append(var.name)

        saver = tf.train.Saver(tf.trainable_variables())
        summary_op1 =  tf.summary.merge(summaries_var)  
        summary_op2 =  tf.summary.merge(summaries_grad)     
        summary_op3 =  tf.summary.merge(summaries_act) 

        session_config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False)
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.94

        logdir = makedir_date()

        session_config.gpu_options.allow_growth=False

        chkpt_dir = join(logdir,'last_chpt')
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        
        with open(join(logdir,'log_tfce_myce.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['TF', 'MY'])

        with tf.Session(config=session_config) as sess:

             # ---------------------------  ST  vectors   -------------------------------------

            CHECKPOINT_PATH = "skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/model.ckpt-501424"
            EMBEDDING_MATRIX_FILE ="skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/expand0722/embeddings.npy"
            VOCAB_FILE = "skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/expand0722/vocab.txt"

            skip_encoder = encoder_manager.EncoderManager()
            skip_encoder.load_model(sess, configuration.model_config(),
                       vocabulary_file=VOCAB_FILE,
                       embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                       checkpoint_path=CHECKPOINT_PATH)

            # ---------------------------------------------------------------------------------


            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            tf.train.start_queue_runners(sess=sess)
            #hn08d05m_0300g5
            saver.restore(sess, tf.train.latest_checkpoint('tnsrbrd/lec04d07m_2006g2/last_chpt'))  #  lec14d07m_1535g2 lec04d07m_2006g2
            tf.sg_set_train(sess) #sg_set_infer

            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord)

            summary_writer1 = tf.summary.FileWriter(join(logdir,'summary_vars'), sess.graph)
            summary_writer2 = tf.summary.FileWriter(join(logdir,'summary_grads'), sess.graph)
            summary_writer3 = tf.summary.FileWriter(join(logdir,'summary_act'), sess.graph)
            c = 0
            for fn in tfrecords_filename:
                for record in tf.python_io.tf_record_iterator(fn):
                    c += 1
            batch_steps = c // Hp.batch_size
            print(" NUMBER OF BATCH STEPS   ",batch_steps)

            #try:
            for ep in range(num_epochs):
                for step in xrange(batch_steps / Hp.num_gpus):
                    start_time = time.time()

                    parSA_val, parMX_val, parGT_val = sess.run([parSA, parMX, parGT],feed_dict={Hp.keep_prob : 0.75})
                    rewards_val = compute_rewards_skipemb_sents(skip_encoder, parSA_val, parMX_val, parGT_val, False) + compute_rewards_bleu_sents(parSA_val, parMX_val, parGT_val, False)
                    d1 = {Hp.keep_prob : 0.75}
                    d2 = {i: d for i, d in zip(rewards, rewards_val)}
                    d1.update(d2)

                    _, loss_value, wtext, ctext = sess.run([train_op, losses, words_splits[-1],chars_splits[-1]],feed_dict=d1)

                    #log_accumulation.append([loss_value, my_cess])
                    duration = time.time() - start_time
                    with open(join(logdir,'log_tfce_myce.csv'), 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([loss_value])


                    if step % 250 == 0:
                        summary_str1, summary_str2,summary_str3 = sess.run([summary_op1,summary_op2,summary_op3],feed_dict=d1)
                        summary_writer1.add_summary(summary_str1, ep*batch_steps / Hp.num_gpus+step)
                        summary_writer2.add_summary(summary_str2, ep*batch_steps / Hp.num_gpus+step)
                        summary_writer3.add_summary(summary_str3, ep*batch_steps / Hp.num_gpus+step)
                        num_examples_per_step = Hp.batch_size * Hp.num_gpus
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration / Hp.num_gpus # num gpus
                        format_str = ('ep: %d, %s: step %d, loss = %8.6f (%.1f examples/sec; %.3f sec/batch) | r_mean = %8.4f r_min = %8.4f r_max = %8.4f')
                        print (format_str % (ep, datetime.now(), step, loss_value,
                                           examples_per_sec, sec_per_batch, np.mean(rewards_val[0]),np.amin(rewards_val[0]),np.amax(rewards_val[0])))

                checkpoint_path = join(chkpt_dir,'model%d.ckpt'%(ep))
                saver.save(sess, checkpoint_path, global_step=ep*(batch_steps // Hp.num_gpus))



if __name__ == "__main__":
  train()
  print("IT IS OVER")


