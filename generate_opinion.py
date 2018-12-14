# coding: utf-8

from __future__ import print_function 
import sugartensor as tf
import time
import numpy as np
from datetime import datetime
import codecs, re, os
import nltk.data, csv, pickle
from os import listdir
from os.path import isfile, join

from LSTMCell import BasicLSTMCell2

from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops, tensor_array_ops, io_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.contrib.slim.python.slim.data import parallel_reader
from tensorflow.python.client import device_lib
import beam_search
from nltk.corpus import wordnet
from itertools import chain
from nltk.corpus import brown

tf.sg_verbosity(10)
batch_steps = 0
local_device_protos = device_lib.list_local_devices()
_gpus = len([x.name for x in local_device_protos if x.device_type == 'GPU'])
num_gpus =  max(_gpus, 1)


class Hp:
    batch_size = 51 # batch size
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

#len 150 OrderedDict([(3, 365703), (6, 391330), (10, 300479), (15, 192515), (20, 99041), (30, 88487), (40, 32657)])
input_datalen = {3:24000}
for el in Hp.par_maxlen:
    batch_steps += input_datalen[el]  #for batch of 16    #1457401 1394000
batch_steps //= Hp.batch_size
num_epochs = 1
word_tags = dict()
for word,pos in brown.tagged_words():
    word_tags[word]=pos


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
    words, chars = tf.train.batch( [word_opinion, char_opinion],
                                                 batch_size=batch_size,
                                                 capacity=3*batch_size,
                                                 num_threads=1)
    
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



def tower_infer_enc(chars, scope, rnn_cell, dec_cell, word_emb, out_reuse_vars=False, dev='/cpu:0'):
    out_rvars = out_reuse_vars

    # make embedding matrix for source and target
    with tf.device(dev):
        with tf.variable_scope('embatch_size',reuse=out_reuse_vars):
          # (vocab_size, latent_dim)
          emb_char = tf.sg_emb(name='emb_char', voca_size=Hp.char_vs, dim=Hp.hd, dev = dev)
          emb_word = tf.sg_emb(name='emb_word', emb=word_emb, voca_size=Hp.word_vs, dim=300, dev = dev)
          
    chars = tf.cast(chars, tf.int32)

    time = tf.constant(0)

    inputs = tf.transpose(chars, perm=[1, 0, 2])
    input_ta = tensor_array_ops.TensorArray(tf.int32, size=tf.shape(chars)[1], dynamic_size=True,clear_after_read = True)
    chars_sent = input_ta.unstack(inputs) #each element is (batch, sentlen)

    resp_steps = tf.shape(chars)[1] # number of sentences in paragraph
    statm_steps = resp_steps // 2

    rnn_state = rnn_cell.zero_state(Hp.batch_size, tf.float32)      #rnn_cell.rnn_state, rnn_cell.rnn_h
    maxdecode = 3
    
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

                #quasi cnn layer ZFO  [batch * 3, seqlen, dim2 ]
                conv = byte_enc.sg_quasi_conv1d(is_enc=True, size=4, name = "qconv_1", dev = dev, reuse_vars=reuse_vars)
                # c = f * c + (1 - f) * z, h = o*c [batch * 4, seqlen, hd]
                pool0 = conv.sg_quasi_rnn(is_enc=False, att=False, name="qrnn_1", reuse_vars=reuse_vars, dev=dev)

                qpool_last = pool0[:,-1,:]


        # --------------------------   MAXPOOL along time dimension   --------------------------

        inpt_maxpl = tf.expand_dims(byte_enc, 1) # [batch, 1, seqlen, channels]
        maxpool = tf.nn.max_pool(inpt_maxpl, [1, 1, Hp.c_maxlen, 1], [1, 1, 1, 1], 'VALID')
        maxpool = tf.squeeze(maxpool, [1, 2])

        # --------------------------   HIGHWAY   --------------------------

        concat = qpool_last + maxpool
        with tf.variable_scope('highway',reuse=reuse_vars):
            input_lstm = highway(concat, concat.get_shape()[-1], num_layers=1)

        # --------------------------   CONTEXT LSTM  --------------------------
        input_lstm = tf.nn.dropout(input_lstm, Hp.keep_prob)

        with tf.variable_scope('contx_lstm', reuse= reuse_vars):
            output, rnn_state = rnn_cell(input_lstm, rnn_state)
            

        return (time+1, rnn_state)


    loop_vars_stat = [time, rnn_state]

    time, rnn_state = tf.while_loop\
                      (rnn_cond_stat, rnn_body_stat, loop_vars_stat, swap_memory=False)

    return rnn_state


def tower_infer_dec(chars, scope, rnn_cell, dec_cell, word_emb, rnn_state, out_reuse_vars=False, dev='/cpu:0'):

    with tf.device(dev):
        with tf.variable_scope('embatch_size',reuse=True):
          # (vocab_size, latent_dim)
          emb_char = tf.sg_emb(name='emb_char', voca_size=Hp.char_vs, dim=Hp.hd, dev = dev)
          emb_word = tf.sg_emb(name='emb_word', emb=word_emb, voca_size=Hp.word_vs, dim=300, dev = dev)
          
    print(chars)
    ch = chars
    ch =  tf.reverse_sequence(input=ch, seq_lengths=[Hp.c_maxlen]*Hp.batch_size, seq_dim=1)
    reuse_vars = reuse_vars_enc =True

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

            #quasi cnn layer ZFO  [batch * 3, seqlen, dim2 ]
            conv = byte_enc.sg_quasi_conv1d(is_enc=True, size=4, name = "qconv_1", dev = dev, reuse_vars=reuse_vars)
            # c = f * c + (1 - f) * z, h = o*c [batch * 4, seqlen, hd]
            pool0 = conv.sg_quasi_rnn(is_enc=False, att=False, name="qrnn_1", reuse_vars=reuse_vars, dev=dev)

            qpool_last = pool0[:,-1,:]


    # --------------------------   MAXPOOL along time dimension   --------------------------

    inpt_maxpl = tf.expand_dims(byte_enc, 1) # [batch, 1, seqlen, channels]
    maxpool = tf.nn.max_pool(inpt_maxpl, [1, 1, Hp.c_maxlen, 1], [1, 1, 1, 1], 'VALID')
    maxpool = tf.squeeze(maxpool, [1, 2])

    # --------------------------   HIGHWAY   --------------------------

    concat = qpool_last+ maxpool
    with tf.variable_scope('highway',reuse=reuse_vars):
        input_lstm = highway(concat, concat.get_shape()[-1], num_layers=1)

    # --------------------------   CONTEXT LSTM  --------------------------

    input_lstm = tf.nn.dropout(input_lstm, Hp.keep_prob)

    with tf.variable_scope('contx_lstm', reuse= reuse_vars):
        output, rnn_state = rnn_cell(input_lstm, rnn_state) 

    beam_size = 8
    reuse_vars = out_reuse_vars


    greedy = False
    if greedy:

        dec_state = rnn_state
        dec_out = []
        d_out = tf.constant([1] * Hp.batch_size)
        for idx in range(Hp.w_maxlen):
            w_input = d_out.sg_lookup(emb=emb_word)
            dec_state = tf.contrib.rnn.LSTMStateTuple( 
                        c = dec_state.c,
                        h = dec_state.h)
            with tf.variable_scope('dec_lstm', reuse= idx>0 or reuse_vars):
                d_out, dec_state = dec_cell(w_input, dec_state)
            
            dec_out.append(d_out)
            d_out = tf.expand_dims(d_out, 1).sg_conv1d_gpus(size=1, dim=Hp.word_vs, name="out_conv", act="linear",dev=dev,reuse= idx>0 or reuse_vars)
            d_out = tf.squeeze(d_out).sg_argmax()
            
        dec_out = tf.stack(dec_out, 1)

        dec = dec_out.sg_conv1d_gpus(size=1, dim=Hp.word_vs, name="out_conv", act="linear",dev=dev,reuse=True)
        return dec.sg_argmax(), rnn_state

    else:

        # ------------------ BEAM SEARCH --------------------
        dec_state =  tf.contrib.rnn.LSTMStateTuple(tf.tile(tf.expand_dims(rnn_state[0], 1), [1,beam_size, 1]), tf.tile(tf.expand_dims(rnn_state[1], 1), [1, beam_size, 1]))
        initial_ids = tf.constant([1] * Hp.batch_size)

        def symbols_to_logits_fn(ids, dec_state):
            dec = []
            dec_c, dec_h = [], []
            # (batch x beam_size x decoded_seq)
            ids = tf.reshape(ids, [Hp.batch_size, beam_size, -1])
            print("dec_state ", dec_state[0].get_shape().as_list())
            for ind in range(beam_size):
                with tf.variable_scope('dec_lstm', reuse= ind>0 or reuse_vars):
                    w_input = ids[:, ind, -1].sg_lookup(emb=emb_word)
                    dec_state0 = tf.contrib.rnn.LSTMStateTuple(
                                        c = dec_state.c[:,ind,:],
                                        h = dec_state.h[:,ind,:])
                    dec_out, dec_state_i = dec_cell(w_input, dec_state0)
                    dec_out = tf.expand_dims(dec_out, 1)
                dec_i = dec_out.sg_conv1d_gpus(size=1, dim=Hp.word_vs, name="out_conv", act="linear",dev=dev,reuse= ind>0 or reuse_vars)

                dec.append(tf.squeeze(dec_i,1))
                dec_c.append(dec_state_i[0])
                dec_h.append(dec_state_i[1])
            return tf.stack(dec, 1), tf.contrib.rnn.LSTMStateTuple(tf.stack(dec_c,1), tf.stack(dec_h,1))

        final_ids, final_probs = beam_search.beam_search(
            symbols_to_logits_fn, dec_state,
            initial_ids,
            beam_size,
            Hp.w_maxlen-1,
            Hp.word_vs,
            3.5,
            eos_id=2) 
            
        return final_ids[:,0,:], rnn_state





def logging(ce):
    return ce

def makedir_date():    
    logfold = time.strftime("eval/sch%dd%mm_%H%Mg"+str(Hp.num_gpus))
    if not os.path.exists(logfold):
        os.makedirs(logfold)
    else:
        logfold = time.strftime("eval/sch%dd%mm_%H")+str(int(time.strftime("%M"))+1)+'g'+str(Hp.num_gpus+1)
        os.makedirs(logfold)
    return logfold
    

def load_vocab():
    #mean padding, BOS, EOS, and OOV
    vocab = u'''␀␂␃⁇N abcdefghijklmnopqrstuvwxyz'''
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def load_word_vocab():
    with open("word_vocab0703.pickle", "rb") as f:
        word_vocab = pickle.load(f)
            # pad BOS EOS OOV
    word_vocab =[u'␀',u'␂',u'␃', u'⁇', 'N'] + word_vocab[5:]
    word2idx = {char: idx for idx, char in enumerate(word_vocab)}
    idx2word = {idx: char for idx, char in enumerate(word_vocab)}
    return word2idx, idx2word


char2idx, idx2char = load_vocab()
word2ind, ind2word = load_word_vocab()

def make_synonyms(word):
    synonyms = wordnet.synsets(word)
    seq = chain.from_iterable([word.lemma_names() for word in synonyms])
    seen = set()
    seen_add = seen.add
    lemmas = [x for x in seq if not (x in seen or seen_add(x))]
    return lemmas

def adjust_char_vocab(source_sent):
    sent_list = source_sent.split()
    digits = set('0123456789')
    for ind in range(len(sent_list)):
        if sent_list[ind].isdigit() or (sent_list[ind] in word_tags and word_tags[sent_list[ind]]=='CD'): 
            sent_list[ind] = "N" #unic_dict[sent_list[ind]]
            continue

        if set(sent_list[ind]) & digits:
            lemmas = make_synonyms(sent_list[ind])
            for wrd in lemmas:
                if wrd in word_tags and word_tags[wrd] in {'OD', 'CD', 'NN'} :
                    sent_list[ind] = wrd
                    break
            if not set(sent_list[ind]) & digits: continue 

        if sent_list[ind] in word2ind and set(sent_list[ind]) & digits:
            word = re.sub(r"[^a-z]", "", sent_list[ind])
            if word in word2ind and word not in {'k', 'x', 'b'}:
                sent_list[ind] = word
                continue
            elif word == 'ww':
                sent_list[ind] = 'war'
                continue

            sent_list[ind] = "N"
            continue

        if sent_list[ind] not in word2ind:
            sent_list[ind] = u'⁇'


    source_sent = " ".join(sent_list)
    source_sent=re.sub(r"[^␀␂␃⁇N abcdefghijklmnopqrstuvwxyz]", " ", source_sent)

    return source_sent

def word2char_ids(wS):

    out = np.zeros((Hp.batch_size, Hp.c_maxlen))
    for b in range(Hp.batch_size):
        stop_ind = np.where(wS[b] == 2)[0]
        if stop_ind.size > 0:
            stop_ind = stop_ind[0]
            wS[b, stop_ind+1:] = wS[b, stop_ind+1:]*0
        temp = list(map(lambda x: ind2word[x], wS[b]))
        temp = [x for x in temp]
        temp_s = " ".join(temp)
        temp_s = adjust_char_vocab(temp_s)
        print(temp_s)
        temp_ind = list(map(lambda x: char2idx[x], temp_s))
        last_ind = min(len(temp_ind), Hp.c_maxlen)
        out[b,:last_ind] = np.array(temp_ind[:last_ind])
    return out

def idxword2txt(array):
    out = []
    ind2word[0] = '0'
    ind2word[1] = 'B'
    ind2word[2] = 'E'
    ind2word[3] = 'UNK'
    for b in range(len(array)):
        temp = list(map(lambda x: ind2word[x], array[b]))
        temp_s = " ".join(temp)
        out += [temp_s]

    return out


def generate():
    dev = '/cpu:0'
    with tf.device(dev):
        mydir = 'tfrc150char_wrd0704'  
        files = [f for f in listdir(mydir) if isfile(join(mydir, f))]
        tfrecords_filename = []
        tfrecords_filename = [join(mydir, 'short_infer3.tfrecords')]  #[join(mydir, f) for f in tfrecords_filename]
        tfrecords_filename_inf = [join(mydir, '11_3.tfrecords')]
        
        print(tfrecords_filename)
        filename_queue = tf.train.string_input_producer(tfrecords_filename, num_epochs=num_epochs,shuffle=True,capacity=1)
        infer_queue = tf.train.string_input_producer(tfrecords_filename_inf, num_epochs=num_epochs,shuffle=True,capacity=1)

        optim = tf.train.AdamOptimizer(learning_rate=0.0001,beta1=0.9, beta2=0.99)

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

        # --------------------------------------------------------------------------------
        with tf.name_scope('%s_%d' % ("tower", 0)) as scope:
            rnn_state = tower_infer_enc(chars_splits[0], scope, rnn_cell, dec_cell, word_emb, out_reuse_vars=False, dev='/cpu:0')

            chars_pl = tf.placeholder(tf.int32, shape=(None, Hp.c_maxlen))
            rnn_state_pl1 = [tf.placeholder(tf.float32, shape=(None, Hp.rnn_hd)),tf.placeholder(tf.float32, shape=(None, Hp.rnn_hd))]
            rnn_state_pl = tf.contrib.rnn.LSTMStateTuple(rnn_state_pl1[0], rnn_state_pl1[1])

            final_ids, rnn_state_dec = tower_infer_dec(chars_pl, scope, rnn_cell, dec_cell, word_emb, rnn_state_pl, out_reuse_vars=False, dev='/cpu:0')


        # --------------------------------------------------------------------------------
        
        saver = tf.train.Saver(tf.trainable_variables())
        session_config=tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False)
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.94

        session_config.gpu_options.allow_growth=False

        restore_dir =  'tnsrbrd/hin17d08m_1313g2'   #   lec30d07m_1634g2   lec04d07m_2006g2     lec28d07m_1221g2    lec31d07m_1548g2
        csv_file = join(restore_dir, time.strftime("hin%dd%mm_%H%M.csv"))
        csv_f = open(csv_file,'a') 
        csv_writer = csv.writer(csv_f)

        with tf.Session(config=session_config) as sess:
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            tf.train.start_queue_runners(sess=sess)
            saver.restore(sess, tf.train.latest_checkpoint(join(restore_dir, 'last_chpt')))  #    lec04d07m_2006g2

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for ep in range(num_epochs):

                tf.sg_set_infer(sess) 
                rnn_state_val, w_txt, ch_txt = sess.run([rnn_state, words_splits[0], chars_splits[0]],feed_dict={Hp.keep_prob : 1.0}) 
                
                predictions =  [] #[w_txt[:,2,:]]
                for idx in range(3):
                    char_inpt = word2char_ids(ids_val) if idx != 0 else ch_txt[:,2,:]
                    ids_val, rnn_state_val = sess.run([final_ids, rnn_state_dec],feed_dict={Hp.keep_prob : 1.0, rnn_state_pl1[0] : rnn_state_val[0], rnn_state_pl1[1] : rnn_state_val[1], chars_pl : char_inpt})
                    temp = np.zeros((Hp.batch_size, Hp.w_maxlen))
                    for b in range(Hp.batch_size):
                        stop_ind = np.where(ids_val[b] == 2)[0]
                        if stop_ind.size > 0:
                            stop_ind = stop_ind[0]
                            ids_val[b, stop_ind+1:] = ids_val[b, stop_ind+1:]*0
                    temp[:,:ids_val.shape[1]]  = ids_val
                    predictions.append(temp)
                
                # predictions are decode_sent x b x w_maxlen
                predictions = np.array(predictions)
                in_batches = [w_txt[b,:,:] for b in range(Hp.batch_size)]
                res_batches = [predictions[:,b,:] for b in range(Hp.batch_size)]

                for b in range(Hp.batch_size):
                    in_paragraph = idxword2txt(in_batches[b])
                    print("\n INPUT SAMPLE \n")
                    print(in_paragraph)

                    res_paragraph = idxword2txt(res_batches[b])
                    print("\n RESULTS \n")
                    print(res_paragraph)

                    csv_writer.writerow([" ".join(in_paragraph[:3]), " ".join(in_paragraph[3:]), " ".join(res_paragraph)])

            csv_f.close()




if __name__ == "__main__":
  generate()
  print("IT IS OVER")
