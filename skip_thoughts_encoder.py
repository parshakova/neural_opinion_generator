# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Class for encoding text using a trained SkipThoughtsModel.

Example usage:
  g = tf.Graph()
  with g.as_default():
    encoder = SkipThoughtsEncoder(embeddings)
    restore_fn = encoder.build_graph_from_config(model_config, checkpoint_path)

  with tf.Session(graph=g) as sess:
    restore_fn(sess)
    skip_thought_vectors = encoder.encode(sess, data)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path


import nltk
import nltk.tokenize
import numpy as np
import tensorflow as tf

from skip_thoughts import skip_thoughts_model
from skip_thoughts.data import special_words
import tensorflow.contrib.slim as slim


def make_mask(data):
  # mask for dynamic rnn to compute length of a sequence

  return (data!=0).astype(int)

def get_trainable_vars_fromchpt(checkpoint_path):
    """Loads the embedding matrix from a skip-thoughts model checkpoint.

    Args:
      checkpoint_path: Model checkpoint file or directory containing a checkpoint
          file.

    Returns:
      word_embedding: A numpy array of shape [vocab_size, embedding_dim].

    Raises:
      ValueError: If no checkpoint file matches checkpoint_path.
    """
    if tf.gfile.IsDirectory(checkpoint_path):
      checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
      if not checkpoint_file:
        raise ValueError("No checkpoint file found in %s" % checkpoint_path)
    else:
      checkpoint_file = checkpoint_path

    tf.logging.info("Loading skip-thoughts embedding matrix from %s",
                    checkpoint_file)
    reader = tf.train.NewCheckpointReader(checkpoint_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    restore_names = []
    for key in sorted(var_to_shape_map):
        if key != 'global_step':
          restore_names.append(key+':0')
          #print(key)


    return restore_names


class SkipThoughtsEncoder(object):
  """Skip-thoughts sentence encoder."""

  def __init__(self, embeddings):
    """Initializes the encoder.

    Args:
      embeddings: Dictionary of word index to embedding vector (1D numpy array).
    """
    self._embeddings = embeddings

  def _create_restore_fn(self, checkpoint_path, saver):
    """Creates a function that restores a model from checkpoint.

    Args:
      checkpoint_path: Checkpoint file or a directory containing a checkpoint
        file.
      saver: Saver for restoring variables from the checkpoint file.

    Returns:
      restore_fn: A function such that restore_fn(sess) loads model variables
        from the checkpoint file.

    Raises:
      ValueError: If checkpoint_path does not refer to a checkpoint file or a
        directory containing a checkpoint file.
    """
    if tf.gfile.IsDirectory(checkpoint_path):
      latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
      if not latest_checkpoint:
        raise ValueError("No checkpoint file found in: %s" % checkpoint_path)
      checkpoint_path = latest_checkpoint

    def _restore_fn(sess):
      tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
      saver.restore(sess, checkpoint_path)
      tf.logging.info("Successfully loaded checkpoint: %s",
                      os.path.basename(checkpoint_path))

    return _restore_fn

  def build_graph_from_config(self, model_config, checkpoint_path):
    """Builds the inference graph from a configuration object.

    Args:
      model_config: Object containing configuration for building the model.
      checkpoint_path: Checkpoint file or a directory containing a checkpoint
        file.

    Returns:
      restore_fn: A function such that restore_fn(sess) loads model variables
        from the checkpoint file.
    """
    tf.logging.info("Building model.")
    model = skip_thoughts_model.SkipThoughtsModel(model_config, mode="encode")
    model.build()
    variables = tf.global_variables()
    variables_to_restore =[]
    restore_names = get_trainable_vars_fromchpt(checkpoint_path)

    for v in variables:
      if v.name in restore_names:
        variables_to_restore += [ v ] 
        print(v.name, v.name in restore_names)


    saver = tf.train.Saver(variables_to_restore)

    return self._create_restore_fn(checkpoint_path, saver)

  def build_graph_from_proto(self, graph_def_file, saver_def_file,
                             checkpoint_path):
    """Builds the inference graph from serialized GraphDef and SaverDef protos.

    Args:
      graph_def_file: File containing a serialized GraphDef proto.
      saver_def_file: File containing a serialized SaverDef proto.
      checkpoint_path: Checkpoint file or a directory containing a checkpoint
        file.

    Returns:
      restore_fn: A function such that restore_fn(sess) loads model variables
        from the checkpoint file.
    """
    # Load the Graph.
    tf.logging.info("Loading GraphDef from file: %s", graph_def_file)
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(graph_def_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")

    # Load the Saver.
    tf.logging.info("Loading SaverDef from file: %s", saver_def_file)
    saver_def = tf.train.SaverDef()
    with tf.gfile.FastGFile(saver_def_file, "rb") as f:
      saver_def.ParseFromString(f.read())
    saver = tf.train.Saver(saver_def=saver_def)

    return self._create_restore_fn(checkpoint_path, saver)

  def _tokenize(self, item):
    """Tokenizes an input string into a list of words."""
    tokenized = []
    for s in self._sentence_detector.tokenize(item):
      tokenized.extend(nltk.tokenize.word_tokenize(s))

    return tokenized

  def _words_to_embedding(self, w):
    """Returns the embeddings for the words from their indices."""
    return self._embeddings[w]

  def encode(self,
             sess,
             data,
             use_norm=False,
             verbose=True,
             batch_size=128,
             use_eos=False):
    """Encodes a sequence of sentences as skip-thought vectors.

    Args:
      sess: TensorFlow Session.
      data: A list of input strings.
      use_norm: Whether to normalize skip-thought vectors to unit L2 norm.
      verbose: Whether to log every batch.
      batch_size: Batch size for the encoder.
      use_eos: Whether to append the end-of-sentence word to each input
        sentence.

    Returns:
      thought_vectors: A list of numpy arrays corresponding to the skip-thought
        encodings of sentences in 'data'.
    """
    # (batch x sent_len) -> (batch x sent_len x emb_size)
    embeddings = self._words_to_embedding(data)
    thought_vectors = []

    mask = make_mask(data)
    feed_dict = {
        "encode_emb:0": embeddings,
        "encode_mask:0": mask,
    }
    thought_vectors.extend(
        sess.run("encoder/thought_vectors:0", feed_dict=feed_dict))


    if use_norm:
        thought_vectorsF=[]
        for v in thought_vectors:
          if np.linalg.norm(v) != 0:
            thought_vectorsF += [v / np.linalg.norm(v)]
          else:
            thought_vectorsF += [v]
        thought_vectors = thought_vectorsF

    return thought_vectors
