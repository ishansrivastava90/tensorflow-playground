import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import softmax_regression as smreg
import deep_cnn as cnn

def load_data():
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	return mnist
	
	
if __name__=="__main__":

	# Initialize session
	sess = tf.InteractiveSession()
	
	# Load data
	data = load_data()
	
	# Train a softmax regression model
	print("Training a softmax model....\n")
	x, ypred, y_ = smreg.train_model(sess, data)
	
	
	# Evaluation
	accuracy = smreg.eval_model(sess, ypred, y_)
	print('Accuracy from softmax regression model = %g' % (sess.run(accuracy, {x: data.test.images, y_: data.test.labels})))
	
	
	# Train a deep CNN
	print("Training a deep CNN model....\n")
	x, ypred, y_, keep_prob = cnn.train_model(sess, data)
		
	# Evaluation
	accuracy = cnn.eval_model(sess, ypred, y_)
	print('Accuracy from depp CNN model = %g' % (sess.run(accuracy, {x: data.test.images, y_: data.test.labels, keep_prob: 1.0})))
	
	
	
	
	
	
	
	
	