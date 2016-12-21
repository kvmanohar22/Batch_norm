from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import os

class Model:
	
	def __init__(self):
		
		"""
		A simple model with two fully connected layers
		First layer : 100 units
		Second layer : 100 units 

		"""
		print 'Initializing hyper-parameters...'
		# Initialize hyper-parameters
		self.learning_rate=1e-4
		self.epochs = 40000
		self.batch_size = 32
		self.epsilon = 1e-3

		w1_initial = np.random.normal(size=(784,100)).astype(np.float32)
		w2_initial = np.random.normal(size=(100,100)).astype(np.float32)
		w3_initial = np.random.normal(size=(100,10)).astype(np.float32)		
		
		# Load data
		print 'Loading data...'
		self.mnist = input_data.read_data_sets('/home/kv/mnist/', one_hot=True)

		self.x = tf.placeholder(dtype=tf.float32, name='Input', 
						shape=[None, 784])
		self.y = tf.placeholder(dtype=tf.float32, name='Labels', 
						shape=[None, 10])		

		# reshape input image
		_input = tf.reshape(self.x, [-1, 784])
	
		print '\n\nStacking layers of the network...'
		# stack the layers
		# without batchnorm
		# W1 = self.weight_init(name='W1', shape=[784, 100])
		# b1 = self.bias_init(name='b1', shape=[100])
		W1 = tf.Variable(w1_initial)
		b1 = tf.Variable(tf.zeros([100]))
		
		full1 = tf.add(tf.matmul(_input, W1), b1)
		self.full1 = tf.nn.sigmoid(full1)
	
		# with batchnorm
		# W1 = self.weight_init(name='W1', shape=[784, 100])
		W1_BN = tf.Variable(w1_initial)
		
		_full1_BN = tf.matmul(_input, W1_BN)
		batch_mean1, batch_var1 = tf.nn.moments(_full1_BN, [0])
		scale1 = tf.Variable(tf.ones([100]))
		beta1 = tf.Variable(tf.zeros([100]))
		_BN1 = tf.nn.batch_normalization(_full1_BN, batch_mean1, batch_var1, 
				scale1, beta1, self.epsilon)
		self.full1_BN = tf.nn.sigmoid(_BN1) 

		# without batchnorm
		# W2 = self.weight_init(name='W2', shape=[100, 100])
		# b2 = self.bias_init(name='b2', shape=[100])
		W2 = tf.Variable(w2_initial)
		b2 = tf.Variable(tf.zeros([100]))

		full2 = tf.add(tf.matmul(self.full1, W2), b2)
		self.full2 = tf.nn.sigmoid(full2)
	
		# with batchnorm
		# W2 = self.weight_init(name='W2', shape=[100, 100])
		W2_BN = tf.Variable(w2_initial)
		
		_full2_BN = tf.matmul(self.full1_BN, W2_BN)
		batch_mean2, batch_var2 = tf.nn.moments(_full2_BN, [0])
		scale2 = tf.Variable(tf.ones([100]))
		beta2 = tf.Variable(tf.zeros([100]))
		_BN2 = tf.nn.batch_normalization(_full2_BN, batch_mean2, batch_var2, 
				 scale2, beta2, self.epsilon)
		self.full2_BN = tf.nn.sigmoid(_BN2)

		# final layer without batchnorm
		# W3 = self.weight_init(name='W3', shape=[100, 10])
		# b3 = self.bias_init(name='b3', shape=[10])
		W3 = tf.Variable(w3_initial)
		b3 = tf.Variable(tf.zeros([10]))		
	
		self.logits = tf.add(tf.matmul(self.full2, W3), b3) 
		self.output = tf.nn.softmax(self.logits)
		
		# final layer with batchnorm
		# W3 = self.weight_init(name='W3', shape=[100, 10])
		# b3 = self.bias_init(name='b3', shape=[10])
		W3_BN = tf.Variable(w3_initial)
		b3_BN = tf.Variable(tf.zeros([10]))		
	
		self.logits_BN = tf.add(tf.matmul(self.full2_BN, W3_BN), b3_BN)
		self.output_BN = tf.nn.softmax(self.logits_BN)

		print 'Creating loss function...'
		# create loss function
		self.loss = -tf.reduce_sum(self.y * tf.log(self.output))
		self.loss_BN = -tf.reduce_sum(self.y * tf.log(self.output_BN))
		
		# optimizer
		self.optimizer = tf.train.AdamOptimizer(learning_rate=
								self.learning_rate).minimize(self.loss)
		self.optimizer_BN = tf.train.AdamOptimizer(learning_rate=
								self.learning_rate).minimize(self.loss_BN)
		
		# predicted class
		self.predicted_class = tf.arg_max(self.output, dimension=1)
		self.predicted_class_BN = tf.arg_max(self.output_BN, dimension=1)

		# number of correct predictions
		self.correct_preds = tf.equal(self.predicted_class, tf.arg_max(self.y, 1))
		self.correct_preds_BN = tf.equal(self.predicted_class_BN, tf.arg_max(
										self.y, 1))

		# accuracy	
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_preds, tf.float32))
		self.accuracy_BN = tf.reduce_mean(tf.cast(self.correct_preds_BN, 
								tf.float32))

	# initialize weights
	def weight_init(self, shape, name=None):
		if name is None:
			name='W'

		W = tf.get_variable(name=name, shape=shape, dtype=tf.float32)
		print type(W)
		return W

	# initialize bias
	def bias_init(self, shape, name=None):
		if name is None:
			name='b'	
		
		b = tf.get_variable(name=name, shape=shape, 
				initializer=tf.constant_initializer(0.2))


	def train(self):
		
		print 'Starting session...'
		self.init_op = tf.initialize_all_variables()
		with tf.Session() as sess:
			sess.run(self.init_op)
			print 'Launching the graph...'
			acc, acc_BN = [], []
			for epoch in xrange(self.epochs):
				batch_x, batch_y = self.mnist.train.next_batch(self.batch_size)
				
				# without batchnorm
				_, _loss, _acc = sess.run([self.optimizer, self.loss, self.accuracy], feed_dict=
								{self.x: batch_x, self.y: batch_y})
				# with batchnorm
				_, _loss_BN, _acc_BN = sess.run([self.optimizer_BN, self.loss_BN, self.accuracy_BN], feed_dict=
								{self.x: batch_x, self.y: batch_y})
				
				print 'Epoch: %d\nWithout BN:\tLoss: %.3f\tAccuracy: %.3f\nWith BN: \tLoss: %.3f\tAccuracy: %.3f\n' % (epoch, _loss, _acc, _loss_BN, _acc_BN)
				if epoch%50 == 0:
					 _acc, _acc_BN, _full2, _full2_BN = sess.run([self.accuracy, self.accuracy_BN, 
																self.full2, self.full2_BN], feed_dict={self.x:
																 self.mnist.test.images, self.y: self.mnist.test.labels})

					 acc.append(_acc)
					 acc_BN.append(_acc_BN)

					 # plot and save histograms
					 plt.figure()
					 plt.subplot(121)
					 plt.hist(_full2.ravel(), 30, range=(-1, 1))
 					 plt.subplot(122)
  					 plt.hist(_full2_BN.ravel(), 30, range=(-1, 1))
					 plt.savefig('./plots/histograms/Epoch_{}'.format(epoch))
					 plt.cla()
					 plt.clf()
			
			print '\nTraining done!'
			print 'Final Test set Accuracy: %.3f\n' % (acc[-1]) 					
			# compare accuracy
			plt.figure()
			plt.plot(acc, 'g-', label='Without BN')
			plt.plot(acc_BN, 'r-', label='With BN')
			plt.title('Accuracy comparison with and without batchnorm')
			plt.xlabel('Epochs')
			plt.ylabel('Accuracy')
			plt.savefig('accuracy_comparison')


# Run the model
_model = Model()
_model.train()

