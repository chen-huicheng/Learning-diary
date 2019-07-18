#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  
# one_hot 独热码的编码（encoding）形式
# 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 的十位数字
# 0 : 1000000000
# 1 : 0100000000
# 2 : 0010000000
# 3 : 0001000000
# 4 : 0000100000 ...


x_data = tf.placeholder("float", shape=[None, 784], name='x_data')  # 输入
y_data = tf.placeholder("float", shape=[None, 10], name='y_data')  # 实际值

# 初始化权重
def weight_variable(shape,names):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 产生正态分布 标准差0.1
    return tf.Variable(initial,name=names)
# 初始化偏置
def bias_variable(shape,names):
    initial = tf.constant(0.1, shape=shape)  # 定义常量
    return tf.Variable(initial,name=names)

# 卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
'''
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
input: 输入图像，张量[batch, in_height, in _width, in_channels]
filter: 卷积核， 张量[filter_height, filter_width, in_channels, out_channels]
strides: 步长，一维向量，长度4
padding：卷积方式，'SAME' 'VALID'
'''
# 池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 最大池化
'''
tf.nn.max_pool(value, ksize, strides, padding, name=None)
value: 输入，一般是卷积层的输出 feature map
ksize: 池化窗口大小，[1, height, width, 1]
strides: 窗口每个维度滑动步长 [1, strides, strides, 1]
padding：和卷积类似，'SAME' 'VALID'
'''
# 第一层卷积  卷积在每个5*5中算出32个特征
W_conv1 = weight_variable([5, 5, 1, 32],'W_conv1')
b_conv1 = bias_variable([32],'b_conv1')
tf.summary.histogram('W_conv1',W_conv1)
tf.summary.histogram('b_conv1',b_conv1)

x_image = tf.reshape(x_data, [-1, 28, 28, 1])
tf.summary.image('input', x_image, 10)

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64],'W_conv2')
b_conv2 = bias_variable([64],'b_conv2')
tf.summary.histogram('W_conv2',W_conv2)
tf.summary.histogram('b_conv2',b_conv2)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层  图片尺寸缩减到了7*7， 本层用1024个神经元处理
W_fc1 = weight_variable([7 * 7 * 64, 1024],'W_fc1')
b_fc1 = bias_variable([1024],'b_fc1')

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout 防止过拟合
keep_prob = tf.placeholder("float", name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层  最后添加一个Softmax层
W_fc2 = weight_variable([1024, 10],'W_fc2')
b_fc2 = bias_variable([10],'b_fc2')

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')

# 训练和评估模型
loss = -tf.reduce_sum(y_data * tf.log(y),name = 'loss')
tf.summary.scalar('loss',loss)
train = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_data, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"),name='accuracy')
tf.summary.scalar('accuracy',accuracy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
logdir = 'log_1'
merge = tf.summary.merge_all()
writer = tf.summary.FileWriter(logdir,sess.graph)

for i in range(10000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy,feed_dict={x_data: batch[0], y_data: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    _,summary = sess.run([train,merge],feed_dict={x_data: batch[0], y_data: batch[1], keep_prob: 0.5})
    writer.add_summary(summary, global_step=i)
print(sess.run(accuracy,feed_dict={x_data: mnist.test.images, y_data: mnist.test.labels, keep_prob: 1.0}))

# 保存模型
saver.save(sess, "Model/mnist_model.ckpt")

writer.close()
sess.close()
