import os
import sys
import tensorflow as tf 
import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D,Conv2D
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.special import erf,erfc
import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils


def get_model(time_len=1):
	ch, row, col = 3, 160, 320  # camera format

	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.,
	        input_shape=(ch, row, col),
	        output_shape=(ch, row, col)))
	model.add(Conv2D(16, (8, 8), padding="same", strides=(4, 4)))
	model.add(ELU())
	model.add(Conv2D(32, (5, 5), padding="same", strides=(2, 2)))
	model.add(ELU())
	model.add(Conv2D(64, (5, 5), padding="same", strides=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(512))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1))

	model.compile(optimizer="adam", loss="mse")

	return model





# Parameters
learning_rate = 0.025
training_iters = 50
batch_size = 128
display_step = 1

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
keep_prob_ReLU = 0.5 # Dropout, probability to keep units
dropout_prob_SNN = 0.05 # Dropout, probability to dropout units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability for ReLU)
dropout_prob =  tf.placeholder(tf.float32) #dropout (dropout probability for SNN)
is_training = tf.placeholder(tf.bool)



weights2 = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32],stddev=np.sqrt(1/25)) ),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64],stddev=np.sqrt(1/(25*32)))),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024],stddev=np.sqrt(1/(7*7*64)))),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes],stddev=np.sqrt(1/(1024))))
}

biases2 = {
    'bc1': tf.Variable(tf.random_normal([32],stddev=0)),
    'bc2': tf.Variable(tf.random_normal([64],stddev=0)),
    'bd1': tf.Variable(tf.random_normal([1024],stddev=0)),
    'out': tf.Variable(tf.random_normal([n_classes],stddev=0))
}


def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0, 
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))

def conv2d_SNN(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return selu(x)

# Create model
def conv_net_SNN(x=tf.placeholder(tf.float32, [None, n_input]), weights=weights2, biases=biases2, dropout_prob=dropout_prob, is_training=is_training):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d_SNN(x, weights['wc1'], biases['bc1'],)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d_SNN(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = selu(fc1)
    
    # Apply Dropout
    fc1 = dropout_selu(fc1, dropout_prob,training=is_training)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# def get_modelSNN():
# 	pred_SNN = conv_net_SNN(x, weights2, biases2, dropout_prob,is_training)

# 	# Define loss and optimizer
# 	cost_SNN = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_SNN, labels=y))

# 	optimizer_SNN = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost_SNN)

# 	# Evaluate ReLU model
# 	accuracy_ReLU = tf.reduce_mean(tf.cast(correct_pred_ReLU, tf.float32))

# 	# Evaluate SNN model
# 	correct_pred_SNN = tf.equal(tf.argmax(pred_SNN, 1), tf.argmax(y, 1))
# 	accuracy_SNN = tf.reduce_mean(tf.cast(correct_pred_SNN, tf.float32))	