---
title: first article by myself
date: 2017-12-19 18:59:38
tags: [article, chax]
---

# 使用TensorFlow实现LSTM
# 1. 预备知识
#### 比较经典的博客:[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
#### 中文版:[理解LSTM网络](http://www.jeyzhang.com/understanding-lstm-network.html)
# 2. TensorFlow相关函数
- unstack
```
unstack(value, num=Nonde, axis=0, name='unstack')
```
- - 将rank-R维的tensor分解为R-1维(会降维)的list。以一个常见的需要RNN处理的[batch_size, timesteps, n_input]为例，由于tensorflow.contrib.rnn.static_rnn的输入是[batch_size, n_input], 所以使用unstack可以轻松的将原始输入转化为需要的shape。 

- transpose

```
transpose(a, perm=None, name='transpose')
```
- - For example：
```
x = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.transpose(x)  # [[1, 4]
                 #  [2, 5]
                 #  [3, 6]]
#perm是需要转置的维度表示
tf.transpose(x, perm=[0, 2, 1])  # [[[1,  4],
             #   [2,  5],
             #   [3,  6]],
             #  [[7, 10],
             #   [8, 11],
             #   [9, 12]]]
```
- split

```
split(value, num_or_size_splits, axis=0, num=None, name='split')
```
将tensor按照axis的维度分解为子tensor的list(不会降维)。
# 3. static_rnn VS dynamic_rnn
###### TensorFlow中内置的rnn有两种：tf.contrib.rnn.static_rnn(之前版本是tf.nn.rnn)和tf.nn.dynamic_rnn

###### 
- static_rnn会创建一个展开的rnn，但是其长度是固定的，也就是说，如果你第一次传进去的shape是200 timesteps的，那么就会创建一个静态的含有200次循环的rnn cell。这会导致两个问题：1. 创建过程会比较耗时 2. 一旦创建好了之后，就不可以再传入比第一次更长的timesteps的sequence了。
- 而dynamic_rnn解决了这个问题，它内部实现的时候是动态的创建rnn的循环图的。
- 所以，比较推荐使用dunamic_rnn来创建rnn或者相关的网络
# 4. TensorFlow Code
#### mnist 数据集
- TensorFlow中内置了mnist数据集，本文使用该数据集作为LSTM用于分类的例子

```
from nnlayers.BasicLSTMLayer import BasicLSTMLayer
mnist = input_data.read_data_sets('tmp/data', one_hot = True)
```
- 设置数据集的training次数、mnist的label数量、batch_size、time_steps&num_input(mnist中数据为28*28的图片)、num_hidden等参数
- 
```
epochs = 10
num_classes = 10
batch_size = 128 
timesteps = 28
num_input = 28
num_hidden = 128
```
- 初始化网络rnn输入和输出之后全连接层的参数
```
x = tf.placeholder('float', [None, timesteps, num_input])
y = tf.placeholder('float', [None, num_classes])
layer = {'weights': tf.Variable(tf.random_normal([num_hidden, num_classes])),
         'biases': tf.Variable(tf.random_normal([num_classes]))}
lstm_layer = BasicLSTMLayer(None, 'test_lstm_layer', None, None,
                            **{'num_hidden': num_hidden, 'input': x, 'timesteps': timesteps})
```
- LSTM的两种实现
- - static_rnn
```
x = tf.unstack(x, timesteps, 1)
#equivalently:
#x = tf.transpose(x, [1,0,2])
#x = tf.reshape(x, [-1, chunk_size])
#x = tf.split(x, n_chunks, 0)
lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
```
- - dynamic_rnn
```
lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, time_major=False, dtype=tf.float32)
```
- 对LSTM输出结果的处理
```
# 如果使用static_rnn实现的话,这句话就不需要：
outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2])) #LSTM网络会将每个时刻的output append到outputs中,所以通过output[-1]取出最后一个时刻的输出
prediction = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases']) #全连接层
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))#计算cost
optimizer = tf.train.AdamOptimizer().minimize(cost)#优化网络中的参数
```
- 将以上步骤串起来通过Session进行计算
```
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        epoch_loss = 0
        for _ in range(int(mnist.train.num_examples / batch_size)):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            epoch_x = epoch_x.reshape((batch_size, timesteps, num_input))
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c

        print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)

    correct = tf.equal(tf.argmax(prediction, 1), tf.arg_max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print('Accuracy:', accuracy.eval({x: mnist.test.images.reshape((-1, timesteps, num_input)), y: mnist.test.labels}))
```
- 完成！最终的training Accuracy:
```
……
Epoch 7 completed out of 10 loss: 16.02936139
Epoch 8 completed out of 10 loss: 14.0518430057
Epoch 9 completed out of 10 loss: 12.9981804799
Accuracy: 0.9826
```










# Reference
1. [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
2. [理解LSTM网络](http://www.jeyzhang.com/understanding-lstm-network.html)
3. [RNNs in Tensorflow, a Practical Guide and Undocumented Features](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/)
4. [RNN w/ LSTM cell example in TensorFlow and Python](https://pythonprogramming.net/rnn-tensorflow-python-machine-learning-tutorial/?completed=/recurrent-neural-network-rnn-lstm-machine-learning-tutorial/)
5. 
6. [RNN LSTM 循环神经网络 (分类例子)](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-08-RNN2/)
7. [TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples)




