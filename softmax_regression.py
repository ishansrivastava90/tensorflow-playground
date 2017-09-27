import tensorflow as tf

def softmax_regmodel(sess, x, y_):
	
	# Declare the model params
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	
	# Initialize the model params
	sess.run(tf.global_variables_initializer())
	
	ypred = tf.matmul(x, W) + b
	
	# define cross-entropy loss after softmax
	loss_ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=ypred)) 
	
	return W, b, ypred, loss_ce
	
def deep_cnn(sess):
	raise NotImplementedError;
	
def train_model(sess, data, model="SoftmaxRegression"):
	
	# Declare x & y_ placeholders for input & output data
	x  = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	
	if model == "DeepCNN":
		W, b, ypred, loss = deep_cnn(sess, x, y_)
	else:
		W, b, ypred, loss = softmax_regmodel(sess, x, y_)

	# define a train step
	train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)
	
	# run training for 1000 epochs
	for _ in range(1000):
		batch = data.train.next_batch(100)
		train_step.run(feed_dict={x: batch[0], y_:batch[1]})
		
	
	return x, ypred, y_
	
def eval_model(sess, ypred, y_):
	correct_predictions = tf.equal(tf.argmax(ypred, 1), tf.argmax(y_, 1))
	
	accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

	return accuracy