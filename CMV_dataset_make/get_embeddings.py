import tensorflow as tf
import numpy as np


def main(unused_argv):

	checkpoint_path = "tnsrbrd/lec17d07m_1141g2/last_chpt"


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
	for key in var_to_shape_map:
		print("tensor_name: ", key)
	word_embedding = reader.get_tensor("embatch_size/emb_word")
	tf.logging.info("Loaded skip-thoughts embedding matrix of shape %s", word_embedding.shape)

	np.savetxt("glove300d_0722.txt", word_embedding)


if __name__ == "__main__":
	tf.app.run()