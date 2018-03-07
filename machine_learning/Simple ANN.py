import tensorflow as tf
import numpy as np
from random import shuffle
#import opencv

def mnist_data_loader():
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  return train_data, train_labels, eval_data, eval_labels

train_X, train_y, test_X, test_y = mnist_data_loader()

epoch_n = 40
batch_size = 100
nodes1 = 460
nodes2 = 460
nodes3 = 460
init = tf.global_variables_initializer()
o_nodes = 10
in_nodes = 784

input_l = tf.placeholder(tf.float32, [None, in_nodes])
labels = tf.placeholder(tf.float32)
hl1 = {"weights" : tf.Variable(tf.random_normal([in_nodes, nodes1])), "biases" : tf.Variable(tf.random_normal([nodes1]))}
hl2 = {"weights" : tf.Variable(tf.random_normal([nodes1, nodes2])), "biases" : tf.Variable(tf.random_normal([nodes2]))}
hl3 = {"weights" : tf.Variable(tf.random_normal([nodes2, nodes3])), "biases" : tf.Variable(tf.random_normal([nodes3]))}
output = {"weights" : tf.Variable(tf.random_normal([nodes3, o_nodes])), "biases" : tf.Variable(tf.random_normal([o_nodes]))}

l1 = tf.add(tf.matmul(input_l, hl1["weights"]), hl1["biases"])
l2 = tf.add(tf.matmul(l1, hl2["weights"]), hl2["biases"])
l3 = tf.add(tf.matmul(l2, hl3["weights"]), hl3["biases"])
output_l = tf.add(tf.matmul(l3, output["weights"]),output["biases"])


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = output_l, labels = labels))
optimize = tf.train.AdamOptimizer(0.00001).minimize(cost)
sess = tf.Session()
sess.run(init)
for epoch in range(epoch_n):
  shuffle(train_X)
  for data in train_X[::batch_size]:
    _, cost = sess.run([optimize, cost], feed_dict = {input_l : train_X[::batch_size], labels : train_y[::batch_size]})
    print(cost)