# coding: utf-8
import sugartensor as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops

def _state_size_with_prefix2(state_size, prefix=None):
    result_state_size = tensor_shape.as_shape(state_size).as_list()
    if prefix is not None:
        if not isinstance(prefix, list):
          raise TypeError("prefix of _state_size_with_prefix should be a list.")
        result_state_size = prefix + result_state_size
    return result_state_size

# layer normalization for rnn
def _ln_rnn(x, gamma, beta):
    r"""Applies layer normalization.
    Normalizes the last dimension of the tensor `x`.
    """
    mean, variance = tf.nn.moments(x, axes=[len(x.get_shape()) - 1], keep_dims=True)

    # apply layer normalization
    x = (x - mean) / tf.sqrt(variance + tf.sg_eps)

    # apply parameter
    return gamma * x + beta

class LSTMCell():
    def __init__(self, in_dim,dim, forget_bias=1.0, activation=tf.tanh,ln=True, bias=True,dtype=tf.float32,dev='/cpu:0',batch_size=3):
    
        self._in_dim = in_dim
        self._dim = dim
        self._forget_bias = forget_bias
        self._activation = activation
        self._ln = False
        self._bias = bias
        self._dev = dev
        self._size = self._in_dim*self._dim
        self._initializer = tf.contrib.layers.xavier_initializer() #tf.random_normal_initializer()
        self._dtype = dtype

        with tf.device(self._dev):        
            with tf.variable_scope("lstm") as scp:
                #self.rnn_state = tf.get_variable("rnn_c",(batch_size, self._dim), dtype=tf.sg_floatx,initializer=tf.constant_initializer(0.0),trainable=False)
                #self.rnn_h = tf.get_variable("rnn_h",(batch_size, self._dim), dtype=tf.sg_floatx,initializer=tf.constant_initializer(0.0),trainable=False)
                self.rnn_state, self.rnn_h = tf.zeros((batch_size, self._dim), dtype=tf.sg_floatx), tf.zeros((batch_size, self._dim), dtype=tf.sg_floatx)
                w_i2h = tf.get_variable('w_i2h', (self._in_dim, 4*self._dim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
                w_h2h = tf.get_variable('w_h2h', (self._dim, 4*self._dim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
                w_b = tf.get_variable('w_b', (1, 4*self._dim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),trainable=True) if self._bias == True else 0.0
                if self._ln:
                    with tf.variable_scope("ln_rnn"):
                      beta = tf.get_variable('beta', self._dim, dtype=tf.sg_floatx, initializer=tf.constant_initializer(0.0),trainable=True)
                      gamma = tf.get_variable('gamma', self._dim, dtype=tf.sg_floatx, initializer=tf.constant_initializer(1.0),trainable=True)

    @property
    def output_size(self):
        return self._size
    
    @property
    def state_size(self):
        return self._dim

    def _linear(self,arys):
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope, reuse=True):
            w_i2h = tf.get_variable("w_i2h")
            w_h2h = tf.get_variable("w_h2h")
            w_b = tf.get_variable("w_b") if self._bias == True else 0
        i2h = tf.matmul(arys[0],w_i2h)
        h2h = tf.matmul(arys[1],w_h2h)
        out = i2h + h2h + w_b
        return out
    
    def zero_state2(self, batch_size):
        dtype = tf.float32
        state_size = self.state_size
        zeros = [0]*2
        for i in range(2):
            zeros_size = _state_size_with_prefix(state_size, prefix=[batch_size])
            zeros[i] = array_ops.zeros(array_ops.stack(zeros_size), dtype=dtype)
            zeros[i].set_shape(_state_size_with_prefix(state_size, prefix=[None]))  
        return (zeros[0], zeros[1])

    def zero_state(self, batch_size):
        dtype = tf.float32
        state_size = self.state_size
        return (tf.zeros((batch_size, state_size), dtype=tf.sg_floatx), tf.zeros((batch_size, state_size), dtype=tf.sg_floatx))

    def __call__(self, tensor, state, scope=None):        
        (prev_c, prev_h) = state
        # i = input_gate, c = new cell value for update, f = forget_gate, o = output_gate
        lstm_matrix = self._linear([tensor, prev_h])
        i, c, f, o = tf.split(value=lstm_matrix, num_or_size_splits=4, axis=1)
        if self._ln:
          with tf.variable_scope("ln_rnn", reuse=True):
              beta = tf.get_variable('beta')
              gamma = tf.get_variable('gamma')

        ln = lambda v: _ln_rnn(v, gamma, beta) if self._ln else v

        # do rnn loop
        new_c =  prev_c * tf.sigmoid(ln(f)) + tf.sigmoid(ln(i)) * self._activation(ln(c))
        new_h =  self._activation(new_c) * tf.sigmoid(ln(o))

        return (new_c, new_h)




class ConvLSTMCell():
    def __init__(self,seqlen, in_dim,dim, forget_bias=1.0, activation=tf.tanh,ln=True, bias=True,dtype=tf.float32, dev='/cpu:0',batch_size=3):
    
        self._in_dim = in_dim
        self._dim = dim
        self._forget_bias = forget_bias
        self._activation = activation
        self._ln = ln
        self._dev = dev
        self._seqlen = seqlen
        self._bias = bias
        self._size = int(self._in_dim*self._dim)
        self._initializer=tf.contrib.layers.xavier_initializer()#tf.random_normal_initializer()
        self._dtype = dtype
        
        with tf.device(self._dev):
          with tf.variable_scope("clstm") as scp:
              #self.crnn_state = tf.get_variable("crnn_c",(batch_size, seqlen, self._dim), dtype=tf.sg_floatx,initializer=tf.constant_initializer(0.0),trainable=False)
              #self.crnn_h = tf.get_variable("crnn_h",(batch_size, seqlen, self._dim), dtype=tf.sg_floatx,initializer=tf.constant_initializer(0.0),trainable=False)
              
              w_ic = tf.get_variable('w_ic', (self._seqlen, self._dim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
              w_fc = tf.get_variable('w_fc', (self._seqlen, self._dim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
              w_oc = tf.get_variable('w_oc', (self._seqlen, self._dim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),trainable=True)

              self.make_states(batch_size)


    @property
    def output_size(self, batch_size):
        return self._size

    def make_states(self, batch_size):
      seqlen = self._seqlen 
      self.crnn_state = tf.get_variable("crnn_c",(batch_size, seqlen, self._dim), dtype=tf.sg_floatx,initializer=tf.constant_initializer(0.0),trainable=False)
      self.crnn_h = tf.get_variable("crnn_h",(batch_size, seqlen, self._dim), dtype=tf.sg_floatx,initializer=tf.constant_initializer(0.0),trainable=False)
        
    def zero_state2(self, batch_size):
        dtype = tf.float32
        state_size = self.state_size
        zeros = [0]*2
        for i in range(2):
            zeros_size = _state_size_with_prefix(state_size, prefix=[batch_size])
            temp = array_ops.zeros(array_ops.stack(zeros_size), dtype=dtype)
            temp.set_shape(_state_size_with_prefix(state_size, prefix=[None]))
            zeros[i] = tf.tile((temp.sg_expand_dims(axis=1)), [1, self._seqlen, 1])

    def zero_state(self, batch_size):
        dtype = tf.float32
        return (tf.zeros((batch_size, self._seqlen, self._dim), dtype=tf.sg_floatx), tf.zeros((batch_size, self._seqlen, self._dim), dtype=tf.sg_floatx)) 

    def __call__(self, x_t, state, size, scope=None, reuse_vars=False):  

        (prev_c, prev_h) = state
        scope = scope or tf.get_variable_scope()
        print("____reuse_______",reuse_vars)
        with tf.variable_scope(scope, reuse=True):
          w_ic = tf.get_variable("w_ic")
          w_fc = tf.get_variable("w_fc")
          w_oc = tf.get_variable("w_oc")

        with tf.sg_context(dev = self._dev,reuse=reuse_vars):
          i = x_t.sg_conv1d_gpus(name = "ix_",size=size)+\
          prev_h.sg_conv1d_gpus(name = "ih_",size=size)+\
          prev_c*w_ic

          f = x_t.sg_aconv1d_gpus(name = "fx_",size=size)+\
          prev_h.sg_aconv1d_gpus(name = "fh_",size=size)+\
          prev_c*w_fc

          c = x_t.sg_conv1d_gpus(name = "cx_",size=size)+\
          prev_h.sg_conv1d_gpus(name = "ch_",size=size)

          o = x_t.sg_conv1d_gpus(name = "ox_",size=size)+\
          prev_h.sg_conv1d_gpus(name = "oh_",size=size)+\
          prev_c*w_oc

        new_c = prev_c * tf.sigmoid(f) + tf.sigmoid(i) * self._activation(c)
        new_h =  self._activation(new_c) * tf.sigmoid(o)

        return (new_c, new_h)


























def tower_loss_manyparams(xx,scope,reu_vars=False):
  # make embedding matrix for source and target
  reu_vars = reu_vars
  with tf.variable_scope('embatch_size',reuse=reu_vars):
    # (vocab_size, latent_dim)
    emb_x = tf.sg_emb(name='emb_x', voca_size=Hp.vs, dim=Hp.hd,dev = self._dev)
    emb_y = tf.sg_emb(name='emb_y', voca_size=Hp.vs, dim=Hp.hd,dev = self._dev)

  xx = tf.cast(xx, tf.int32)

  time = tf.constant(0)
  losses_int = tf.constant(0.0)
  inputs = tf.transpose(xx, perm=[1, 0, 2])
  input_ta = tensor_array_ops.TensorArray(tf.int32, size=1, dynamic_size=True,clear_after_read = False)
  x_sent = input_ta.unstack(inputs) #each element is (batch, sentlen)

  n_steps = tf.shape(xx)[1] # number of sentences in paragraph

  # generate first an unconditioned sentence
  n_input = Hp.hd
  subrec1_init = subrec_zero_state(Hp.batch_size,Hp.hd)
  subrec2_init = subrec_zero_state(Hp.batch_size,Hp.hd)

  with tf.variable_scope("mem",reuse=reu_vars) as scp:
    rnn_cell = LSTMCell(in_dim=h,dim=Hp.hd)
    crnn_cell = ConvLSTMCell(seqlen = Hp.maxlen, in_dim=n_input//2,dim=Hp.hd//2)

  (rnn_state_init, rnn_h_init) = rnn_cell.zero_state(Hp.batch_size)

  #   (batch, sentlen, latentdim/2)
  (crnn_state_init, crnn_h_init) = crnn_cell.zero_state(Hp.batch_size)
  
  def rnn_cond(time,subrec1, subrec2, rnn_state, rnn_h,crnn_state, crnn_h, losses):
    return tf.less(time, n_steps-1)    
    
  def rnn_body(time,subrec1, subrec2, rnn_state, rnn_h,crnn_state, crnn_h, losses):
      x = x_sent.read(time)
      y = x_sent.read(time+1)      #   (batch, sentlen) = (16, 200) 
      
      # shift target by one step for training source
      y_src = tf.concat([tf.zeros((Hp.batch_size, 1), tf.int32), y[:, :-1]],1)
      reuse_vars = time == tf.constant(0) or reu_vars

# --------------------------   BYTENET ENCODER   --------------------------

      # embed table lookup
      enc = x.sg_lookup(emb=emb_x)  #(batch, sentlen, latentdim)
      # loop dilated conv block
      for i in range(num_blocks):
          enc = (enc
                 .sg_res_block(size=5, rate=1, name = "enc1_%d"%(i),reuse_vars=reuse_vars)
                 .sg_res_block(size=5, rate=2, name = "enc2_%d"%(i),reuse_vars=reuse_vars)
                 .sg_res_block(size=5, rate=4, name = "enc4_%d"%(i),reuse_vars=reuse_vars)
                 .sg_res_block(size=5, rate=8, name = "enc8_%d"%(i),reuse_vars=reuse_vars)
                 .sg_res_block(size=5, rate=16,name = "enc16_%d"%(i),reuse_vars=reuse_vars))

# --------------------------   QCNN + QPOOL ENCODER with attention #1  --------------------------

      #quasi cnn layer ZFO  [batch * 3, t, dim2 ]
      conv = enc.sg_quasi_conv1d(is_enc=True,size=3,name = "qconv_1",reuse_vars=reuse_vars)
      #attention layer
      # recurrent layer # 1 + final encoder hidden state
      subrec1 = tf.tile((subrec1.sg_expand_dims(axis=1)), [1, Hp.maxlen, 1])        
      concat = conv.sg_concat(target=subrec1,axis=0) # (batch*4, sentlen, latentdim)
      pool = concat.sg_quasi_rnn(is_enc=True,att=True,name="qrnn_1",reuse_vars=reuse_vars)
      subrec1 = pool[:Hp.batch_size,-1,:] # last character in sequence

# --------------------------   QCNN + QPOOL ENCODER with attention #2  --------------------------     

      # quazi cnn ZFO (batch*3, sentlen, latentdim)
      conv = pool.sg_quasi_conv1d(is_enc=True,size=2,name = "qconv_2",reuse_vars=reuse_vars)
      # (batch, sentlen-duplicated, latentdim)
      subrec2 = tf.tile((subrec2.sg_expand_dims(axis=1)), [1, Hp.maxlen, 1])
      # (batch*4, sentlen, latentdim)
      concat = conv.sg_concat(target=subrec2,axis=0)
      pool = concat.sg_quasi_rnn(is_enc=True,att=True,name="qrnn_2",reuse_vars=reuse_vars)
      subrec2 = pool[:Hp.batch_size,-1,:] # last character in sequence

# --------------------------   ConvLSTM with RESIDUAL connection and MULTIPLICATIVE block   --------------------------

      #residual block
      causal = False # for encoder
      crnn_input = (pool[:Hp.batch_size,:,:]
            .sg_bypass_gpus(name='relu_0',act='relu', bn=(not causal), ln=causal)
            .sg_conv1d_gpus(name = "dimred_0",size=1,dev="/cpu:0",reuse=reuse_vars, dim=Hp.hd/2, act='relu', bn=(not causal), ln=causal))

      # conv LSTM  
      with tf.variable_scope("mem/clstm") as scp: 
          (crnn_state, crnn_h) = crnn_cell(crnn_input,(crnn_state, crnn_h),size=5,reuse_vars=reuse_vars)
      # dimension recover and residual connection
      rnn_input0 = pool[:Hp.batch_size,:,:] + crnn_h\
                  .sg_conv1d_gpus(name = "diminc_0",size=1,dev="/cpu:0", dim=Hp.hd,reuse=reuse_vars, act='relu', bn=(not causal), ln=causal)
      
# --------------------------   QCNN + QPOOL ENCODER with attention #3  --------------------------

      # pooling for lstm input
      # quazi cnn ZFO (batch*3, sentlen, latentdim)
      conv = rnn_input0.sg_quasi_conv1d(is_enc=True,size=2,name = "qconv_3",reuse_vars=reuse_vars)
      pool = conv.sg_quasi_rnn(is_enc=True,att=False,name="qrnn_3",reuse_vars=reuse_vars)
      rnn_input = pool[:Hp.batch_size,-1,:] # last character in sequence

# --------------------------   LSTM with RESIDUAL connection and MULTIPLICATIVE block --------------------------

      # recurrent block
      with tf.variable_scope("mem/lstm") as scp: 
          (rnn_state, rnn_h) = rnn_cell(rnn_input,(rnn_state, rnn_h))

      rnn_h2 = tf.tile(((rnn_h + rnn_input).sg_expand_dims(axis=1)), [1, Hp.maxlen, 1]) 
      
# --------------------------   BYTENET DECODER   --------------------------

      # CNN decoder 
      dec = y_src.sg_lookup(emb=emb_y).sg_concat(target=rnn_h2, name = "dec")
      
      for i in range(num_blocks):
          dec = (dec
                 .sg_res_block(size=3, rate=1, causal=True,name = "dec1_%d"%(i),reuse_vars=reuse_vars)
                 .sg_res_block(size=3, rate=2, causal=True,name = "dec2_%d"%(i),reuse_vars=reuse_vars)
                 .sg_res_block(size=3, rate=4, causal=True,name = "dec4_%d"%(i),reuse_vars=reuse_vars)
                 .sg_res_block(size=3, rate=8, causal=True,name = "dec8_%d"%(i),reuse_vars=reuse_vars)
                 .sg_res_block(size=3, rate=16, causal=True,name = "dec16_%d"%(i),reuse_vars=reuse_vars))

      # final fully convolution layer for softmax
      dec = dec.sg_conv1d_gpus(size=1, dim=Hp.vs,name="out",summary=False, dev = self._dev,reuse=reuse_vars)

      ce_array = dec.sg_ce(target=y, mask=True, name = "cross_ent_example")        
      cross_entropy_mean = tf.reduce_mean(ce_array, name='cross_entropy')

      losses = tf.add_n([losses,cross_entropy_mean], name='total_loss')

      return (time+1,subrec1, subrec2, rnn_state, rnn_h,crnn_state, crnn_h, losses)


def tower_loss2_old(xx,scope,reuse_vars=False):

    # make embedding matrix for source and target
    with tf.variable_scope('embs',reuse=reuse_vars):
      emb_x = tf.sg_emb(name='emb_x', voca_size=Hp.vs, dim=Hp.hd,dev = self._dev)
      emb_y = tf.sg_emb(name='emb_y', voca_size=Hp.vs, dim=Hp.hd,dev = self._dev)

    x_sents = tf.unstack(xx,axis=1) #each element is (batch, sentlen)

    # generate first an unconditioned sentence
    n_input = Hp.hd

    subrec1 = subrec_zero_state(Hp.bs,Hp.hd)
    subrec2 = subrec_zero_state(Hp.bs,Hp.hd)

    rnn_cell = LSTMCell(in_dim=n_input,dim=Hp.hd)
    (rnn_state, rnn_h) = rnn_cell.zero_state(Hp.bs)

    crnn_cell = ConvLSTMCell(in_dim=n_input,dim=Hp.hd)
    (crnn_state, crnn_h) = crnn_cell.zero_state(n_input)

    for sent in range(len(x_sents)-1):
        y = x_sents[i+1]
        x = x_sents[i]      #   (batch, sentlen) = (16, 200) 
        # shift target by one step for training source
        y_src = tf.concat([tf.zeros((Hp.bs, 1), tf.sg_intx), y[:, :-1]],1)

        # embed table lookup
        enc = x.sg_lookup(emb=emb_x)  #(batch, sentlen, dim1)
        # loop dilated conv block
        for i in range(num_blocks):
            enc = (enc
                   .sg_res_block(size=5, rate=1, name = "enc1_%d"%(i),reuse_vars=reuse_vars)
                   .sg_res_block(size=5, rate=2, name = "enc2_%d"%(i),reuse_vars=reuse_vars)
                   .sg_res_block(size=5, rate=4, name = "enc4_%d"%(i),reuse_vars=reuse_vars)
                   .sg_res_block(size=5, rate=8, name = "enc8_%d"%(i),reuse_vars=reuse_vars)
                   .sg_res_block(size=5, rate=16,name = "enc16_%d"%(i),reuse_vars=reuse_vars))

        #quasi rnn layer  [batch * 3, t, dim2 ]
        conv = enc.sg_quasi_conv1d(is_enc=True,size=2,name = "conv1",reuse_vars=reuse_vars)
        #attention layer
        # recurrent layer # 1 + final encoder hidden state
        concat = subrec1.sg_concat(target=conv,dim=0)
        subrec1 = conv.sg_quasi_rnn(is_enc=True,att=True)

        conv = pool.sg_quasi_conv1d(is_enc=True,size=2,name = "conv2",reuse_vars=reuse_vars)
        concat = subrec2.sg_concat(target=conv,dim=0)
        subrec2 = conv.sg_quasi_rnn(is_enc=True,att=True)

        # conv LSTM  
        (crnn_state, crnn_h) = crnn_cell(subrec2,(crnn_state, crnn_h),5)

        # recurrent block
        (rnn_state, rnn_h) = rnn_cell(crnn_h,(rnn_state, rnn_h))
        

        # CNN decoder 
        dec = crnn_h.sg_concat(target=y_src.sg_lookup(emb=emb_y), name = "dec")
        
        for i in range(num_blocks):
            dec = (dec
                   .sg_res_block(size=3, rate=1, causal=True,name = "dec1_%d"%(i),reuse_vars=reuse_vars)
                   .sg_res_block(size=3, rate=2, causal=True,name = "dec2_%d"%(i),reuse_vars=reuse_vars)
                   .sg_res_block(size=3, rate=4, causal=True,name = "dec4_%d"%(i),reuse_vars=reuse_vars)
                   .sg_res_block(size=3, rate=8, causal=True,name = "dec8_%d"%(i),reuse_vars=reuse_vars)
                   .sg_res_block(size=3, rate=16, causal=True,name = "dec16_%d"%(i),reuse_vars=reuse_vars))

        # final fully convolution layer for softmax
        dec = dec.sg_conv1d_gpus(size=1, dim=Hp.vs,name="out",summary=False,\
          dev = self._dev,reuse=reuse_vars)

        ce_array = dec.sg_ce(target=y, mask=True, name = "cross_ent_example")
        cross_entropy_mean = tf.reduce_mean(ce_array, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)
    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    return total_loss

def load_vocab():
    #mean padding, BOS, EOS, and OOV
    vocab = u'''␀␂␃⁇ ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789().?!,:'-`;'''
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def convert_sent_toind(source_sent): 
    char2idx, idx2char = load_vocab()
    
    X, Sources = [], []
    
    x = [char2idx.get(char, 3) for char in source_sent] # 3: OOV
    if len(x) <= Hp.maxlen:
        x += [0] * (Hp.maxlen - len(x)) # zero postpadding
        
        X.append(x)
        Sources.append(source_sent)
    
    return X

# sents = parse_on_sentences_old(pars)
def parse_on_sentences_old(paragraph):
    print("############## ")
    print(type(paragraph))
    # cannot tokenize Tensor
    sents = sent_detector.tokenize(paragraph)
    X = []
    X.append([1]+[0]*Hp.maxlen) # BOS

    for sent in sents:
      X.append(convert_sent_toind(sent.strip()))

    X.append([2]+[0]*Hp.maxlen) # EOS

    X = np.array(X, np.int32)    
    print("X.shape =", X.shape) # (157014, 150)
    return X

#       pars = get_data_queue('csv_200symb')
def get_data_queue(mydir):
    files = [f for f in listdir(mydir) if isfile(join(mydir, f))]
    onlyfiles = sorted(files, key=lambda x: int(x.split('.')[0]))

    filename_queue = tf.train.string_input_producer(onlyfiles)

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [[" "]]
    x_p = tf.decode_csv(value, record_defaults=record_defaults)
    paragraphs = tf.train.shuffle_batch([x_p],
                                num_threads=32,
                                batch_size=Hp.bs, 
                                capacity=Hp.bs*64,
                                min_after_dequeue=Hp.bs*32, 
                                allow_smaller_final_batch=False)
    return paragraph