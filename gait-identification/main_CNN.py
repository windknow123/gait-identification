__author__ = 'jason'

from data_tool import load_training_validation_data
from model.models import RandomForestClassification
from feature.hog import flatten

import os
import tensorflow as tf 
import numpy as np
import tool

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.logging.set_verbosity(tf.logging.INFO)

CONV1_DEEP = 16
#CONV1_DEEP = 32
#CONV1_SIZE = 7
#CONV1_SIZE = 5
CONV1_SIZE = 3
CONV2_DEEP = 64
#CONV2_DEEP = 32
#CONV2_SIZE = 7
#CONV2_SIZE = 5
CONV2_SIZE = 3
CONV3_DEEP = 256
#CONV3_SIZE = 7
CONV3_SIZE = 3
POOL_SIZE = 2
POOL_STRIDE = 2
FC_SIZE = 1024
DROP_RATE = 0.4
OUTPUT_NODE = 20

BATCH_SIZE = 16

LEARNING_RATE_BASE = 0.00001
LEARNING_RATE_DECAY = 0.999

train_view = "090"
val_view = "090"
train_dir = ["nm-01", "nm-02", "nm-03", "nm-04"] 
val_dir = ["nm-05", "nm-06"] 

def cnn_model_fn(features, labels, mode):

	input_layer = tf.reshape(
		features["x"],
		[-1, tool.IMAGE_WIDTH, tool.IMAGE_LENGTH, 1])

	conv1 = tf.layers.conv2d(
		inputs = input_layer,
		filters = CONV1_DEEP,
		kernel_size = CONV1_SIZE,
		#padding = "valid",
		padding = "same",
		activation = tf.nn.relu)
	
	pool1 = tf.layers.max_pooling2d(
		inputs = conv1,
		pool_size = POOL_SIZE,
		strides = POOL_STRIDE)
	
	conv2 = tf.layers.conv2d(
		inputs = pool1,
		filters = CONV2_DEEP,
		kernel_size = CONV2_SIZE,
		#padding = "valid",
		padding = "same",
		activation = tf.nn.relu)
	
	pool2 = tf.layers.max_pooling2d(
		inputs = conv2,
		pool_size = POOL_SIZE,
		strides = POOL_STRIDE)
	
	conv3 = tf.layers.conv2d(
		inputs = pool2,
		filters = CONV3_DEEP,
		kernel_size = CONV3_SIZE,
		padding = "valid",
		activation = tf.nn.relu)
	
	#pool3 = tf.layers.max_pooling2d(
	#	inputs = conv3,
	#	pool_size = POOL_SIZE,
	#	strides = POOL_STRIDE)
	
	#flattened = tf.reshape(pool3, [-1, 11*21*256])

	flattened = tf.reshape(conv3, [-1, 10*34*256])
	
	dense = tf.layers.dense(
		inputs = flattened,
		units = FC_SIZE,
		activation = tf.nn.relu)
	
	dropout = tf.layers.dropout(
		inputs = dense,
		rate = DROP_RATE,
		training = mode==tf.estimator.ModeKeys.TRAIN)
	
	logits = tf.layers.dense(
		inputs = dropout,
		units = OUTPUT_NODE)

	predictions = {
		"classes": tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")}	

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)	
	
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=OUTPUT_NODE)
	loss = tf.losses.softmax_cross_entropy(
		onehot_labels=onehot_labels, logits=logits)
	
	if mode == tf.estimator.ModeKeys.TRAIN:
		learning_rate = tf.train.exponential_decay(
			LEARNING_RATE_BASE,
			tf.train.get_global_step(),
			4*OUTPUT_NODE/BATCH_SIZE,
			LEARNING_RATE_DECAY)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
		train_op = optimizer.minimize(
			loss = loss,
			global_step = tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
	
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
			labels=labels, predictions=predictions["classes"])}

	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
	# Load data
	train_x, train_y, val_x, val_y = load_training_validation_data(
		train_view = train_view,
		train_dir = train_dir,
		val_view = val_view,
		val_dir = val_dir)
	train_data = np.asarray([flatten(x) for x in train_x], dtype=np.float32)
	val_data = np.asarray([flatten(x) for x in val_x], dtype=np.float32)
	train_labels = np.asarray(train_y, dtype=np.int32)
	val_labels= np.asarray(val_y, dtype=np.int32)

	# Create the Estimator
	gei_classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn, model_dir="./chpts")

	# Set up logging for predictions
	#tensors_to_log = {"probabilities": "softmax_tensor"}
	#logging_hook = tf.train.LoggingTensorHook(
	#	tensors=tensors_to_log, every_n_iter=5000)

	# Train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x = {"x": train_data},
		y = train_labels,
		batch_size = BATCH_SIZE,
		num_epochs = None,
		shuffle = True)
	#gei_classifier.train(
	#	input_fn = train_input_fn,
	#	steps = 20000,
	#	hooks = [logging_hook])
	gei_classifier.train(
		input_fn = train_input_fn,
		steps = 20000)

	# Evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x = {"x": val_data},
		y = val_labels,
		num_epochs = 1,
		shuffle = False)
	eval_results = gei_classifier.evaluate(input_fn=eval_input_fn)		
	print(eval_results)

if __name__ == "__main__":
	tf.app.run()

#if __name__ == '__main__':
#	train_x, train_y, val_x, val_y = load_training_validation_data(
#		train_view=train_view,
#		train_dir=train_dir,
#		val_view=val_view,
#		val_dir=val_dir)	
#	train_feature_x = [flatten(x) for x in train_x]
#	val_feature_x = [flatten(x) for x in val_x]
#	
#	model = RandomForestClassification()
#	model.fit(x_train=train_feature_x, y_train=train_y)
#	predict_y = model.predict(val_feature_x)
#	
#	print "predict_y: "
#	print predict_y
#
#	correct_count = sum(predict_y==val_y)
#	accuracy = correct_count*1.0/len(val_y)
#
#	print "train view: %s, val view: %s, accuracy: %d/%d=%.3f" % \
#		(train_view, val_view, correct_count, len(val_y), accuracy)
