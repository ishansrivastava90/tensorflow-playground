import tensorflow as tf

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
	
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
	
def maxpool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	
def cnn_model(sess, x, y_):
	
	#### Construct 1st conv layer ###	
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	
	# reshape the image into 4D tensor
	x_image = tf.reshape(x, [-1, 28, 28, 1])
	
	# do convolution and apply ReLU
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = maxpool_2x2(h_conv1)
	
	#### Construct 2nd conv layer ###
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = maxpool_2x2(h_conv2)
	
	#### Deeply connected layer ###
	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])
	
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	
	#### Dropout ####
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	#### Readout layer ####
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	
	# Define loss
	loss_ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	
	return keep_prob, y_conv, loss_ce
	
def train_model(sess, data):
	
	# Declare x & y_ placeholders for input & output data
	x  = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	

	# Get model params & loss
	keep_prob, y_conv, loss = cnn_model(sess, x, y_)
		
	# define a train step
	train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
	
	# declare prediction and accuracy nodes in graph
	correct_prediction= tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	# initialize variables
	sess.run(tf.global_variables_initializer())
		
	
	# run training for 2000 epochs
	for i in range(2000):
		batch = data.train.next_batch(50)
		if i % 100 == 0:
			train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0})
			print('step %d, training accuracy %g' % (i, train_accuracy))
		
		train_step.run(feed_dict={x: batch[0], y_:batch[1], keep_prob: 0.5})
	
	return x, y_conv, y_, keep_prob
	
def eval_model(sess, ypred, y_):
	correct_predictions = tf.equal(tf.argmax(ypred, 1), tf.argmax(y_, 1))
	
	accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

	return accuracy