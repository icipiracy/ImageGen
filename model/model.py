"""

Based on LeCun deep convolutional network used for MNIST digits recognnition, to test generated data from Blender.

This can be improved  for better results, but for the memory use is just below the limit of my GPUs.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint
import argparse
import sys
import datetime
import input_data
import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

data_dir = sys.argv[1] # Full path to directory which contains the rendered images.
data_index = sys.argv[2] # Full path to the .csv file which contains the labels and rendered images filenames.
test = sys.argv[3] # Wether this app is run in 'test' mode or 'live'
amount = sys.argv[4]

saved = False
sfpath = False

# if len(sys.argv) > 4:
#
#     print(len(sys.argv))
#
#     saved = sys.argv[5] # Wether the preprocessed data is loaded from a previously saved file.
#     sfpath = sys.argv[6] # File path to saved image data.

def main():

    # Import png image data from 'ap_input_data' module:
    inputData = input_data.loadData(data_dir,data_index,test,amount,saved,sfpath)

    # ImgInfo provides array with width,height and depth of image data:
    ImgInfo = inputData['imginfo']
    inputData = inputData['dset']

    # Retrieve dataset information:
    classes = ImgInfo['classes'] # Amount of distinct objects
    width = ImgInfo['width'] # Size in pixels
    height = ImgInfo['height'] # Size in pixels
    total_size = width * height # Total amount of pixels
    depth = ImgInfo['depth'] # Amount of color channels.


    # config = tf.ConfigProto(log_device_placement=True,device_count={"CPU":2},inter_op_parallelism_threads=16,intra_op_parallelism_threads=1)
    config = tf.ConfigProto(log_device_placement=False)

    with tf.device('/cpu:0'):
        x = tf.placeholder(tf.float32, shape=[None, total_size])
        y_ = tf.placeholder(tf.float32, shape=[None, classes])

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        x_image = tf.reshape(x, [-1,width,height,depth])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # Hidden Layer 2:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        wfcwidth = int(width / 2 / 2)

        W_fc1 = weight_variable([wfcwidth * wfcwidth * 64, 1024]) # NOTE: Must be same shape as h_pool2
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, wfcwidth*wfcwidth*64]) # NOTE: Must be same shape as h_pool2
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout to reduce overfitting:
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Output layer:
        W_fc2 = weight_variable([1024, classes])
        b_fc2 = bias_variable([classes])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # Define train step with cross_entropy, reduce mean:
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # Define accuracy evaluation with the correct label and the estimated label:
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    run_metadata = tf.RunMetadata()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)

    # log_device_placement shows what devices are used by tensorflow:
    with tf.Session(config=config) as sess:

        init_op = tf.global_variables_initializer()

        sess.run(init_op,options=run_options,run_metadata=run_metadata)

        # Training steps:
        for i in range(1000):

            batch = inputData.train.next_batch(32)

            # Step size:
            if i%64 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, train: %g"%(i, train_accuracy))
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            if i%1000 == 0:
                print("test: %g"%accuracy.eval(feed_dict={x: inputData.test.images, y_: inputData.test.labels, keep_prob: 1.0}))

    # Log memory use to track down inefficiencies:
    logname = "TF_MemLog_" + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M") + "_.txt"

    with open(logname, "w") as out:
        out.write(str(run_metadata))

main()
