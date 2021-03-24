import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist


tf.compat.v1.disable_eager_execution()
tf = tf.compat.v1

learning_rate = 0.0001
epochs = 100
batch_size = 128
dropout = 0.8
filter_height = 3
filter_width = 3


(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

n_classes = len(np.unique(Y_train))



num_batches = Y_train.shape[0]/batch_size

labels_train = to_categorical(Y_train)
labels_test = to_categorical(Y_test)


X = tf.placeholder(tf.float32, shape=[None, 28, 28])
y = tf.placeholder(tf.float32, shape=[None, n_classes])



seed = 32
stddev = 0.0001
weights = {
    'wc1': tf.Variable(tf.random_normal(shape=[filter_height, filter_width, 1, 32], stddev=stddev, seed=seed)),
    'wc2': tf.Variable(tf.random_normal(shape=[filter_height, filter_width, 32, 32], stddev=stddev, seed=seed)),
    'wc3': tf.Variable(tf.random_normal(shape=[filter_height, filter_width, 32, 64], stddev=stddev, seed=seed)),
    'wc4': tf.Variable(tf.random_normal(shape=[filter_height, filter_width, 64, 64], stddev=stddev, seed=seed)),
    'wc5': tf.Variable(tf.random_normal(shape=[filter_height, filter_width, 64, 128], stddev=stddev, seed=seed)),
    'wc6': tf.Variable(tf.random_normal(shape=[filter_height, filter_width, 128, 128], stddev=stddev, seed=seed)),
    'wd1': tf.Variable(tf.random_normal(shape=[4*4*128, 128],stddev=stddev, seed=seed )),
    'wd2': tf.Variable(tf.random_normal(shape=[128, n_classes], stddev=stddev, seed=seed))
}

biases = {
    'bc1': tf.Variable(tf.random_normal(shape=[32], stddev=stddev, seed=seed)),
    'bc2': tf.Variable(tf.random_normal(shape=[32], stddev=stddev, seed=seed)),
    'bc3': tf.Variable(tf.random_normal(shape=[64], stddev=stddev, seed=seed)),
    'bc4': tf.Variable(tf.random_normal(shape=[64], stddev=stddev, seed=seed)),
    'bc5': tf.Variable(tf.random_normal(shape=[128], stddev=stddev, seed=seed)),
    'bc6': tf.Variable(tf.random_normal(shape=[128], stddev=stddev, seed=seed)),
    'bd1': tf.Variable(tf.random_normal([128], stddev=stddev, seed=seed)),
    'bd2': tf.Variable(tf.random_normal([n_classes], stddev=stddev, seed=seed))
}




def conv_layer(X, W, b, stride=1):
    X = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='SAME')
    X = tf.nn.bias_add(X, b)
    return tf.nn.elu(X)


def max_pooling_layer(X, stride=2):
    return tf.nn.max_pool(X, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')


def conv_net(X, weights, biases):
    X = tf.reshape(X, shape=[-1, 28, 28, 1])

    conv1 = conv_layer(X, weights['wc1'], biases['bc1'], stride=1)
    conv2 = conv_layer(conv1, weights['wc2'], biases['bc2'], stride=1)

    conv2 = max_pooling_layer(conv2, 2)
    conv2 = tf.nn.dropout(conv2, dropout)

    conv3 = conv_layer(conv2, weights['wc3'], biases['bc3'], stride=1)

    conv4 = conv_layer(conv3, weights['wc4'], biases['bc4'], stride=1)
    conv4 = max_pooling_layer(conv4, 2)
    conv4 = tf.nn.dropout(conv4, dropout)

    conv5 = conv_layer(conv4, weights['wc5'], biases['bc5'], stride=1)

    conv6 = conv_layer(conv5, weights['wc6'], biases['bc6'], stride=1)
    conv6 = max_pooling_layer(conv6, 2)
    conv6 = tf.nn.dropout(conv6, dropout)

    fc1 = tf.reshape(conv6, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.elu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])

    return fc2


y_pred = conv_net(X, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

prediction_accuracy = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(prediction_accuracy, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        print('Epoch ', epoch)
        for i in range(0, Y_train.shape[0], batch_size):
            x_batch = X_train[i:i+batch_size, :]
            y_batch = labels_train[i:i+batch_size, :]
            sess.run(optimizer, feed_dict={X: x_batch, y: y_batch})
            [loss, acc] = sess.run([cost, accuracy] ,  feed_dict={X: x_batch, y: y_batch})
            if i % 1000 == 0:
                print('i: ', i, 'acc: ', acc)

    print('Optimization Completed')
    test_acc = sess.run(accuracy, feed_dict={X: X_test, y: labels_test})
    print('Test Accuracy: ', test_acc)