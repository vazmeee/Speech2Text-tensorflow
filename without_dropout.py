#  tensorflow 1.x version
# 20 epochs 0.729 speech18 dataset

import numpy as np
from tqdm import tqdm

import os
import librosa
import numpy as np
from scipy.io import wavfile

train_audio_path = 'C:/Users/ROG/Desktop/TDL/project/speech18/'
labels=os.listdir(train_audio_path)

all_wave = []
all_label = []
for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if(len(samples)== 8000) :
            all_wave.append(samples)
            all_label.append(label)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(all_label)
classes= list(le.classes_)

all_wave = np.array(all_wave).reshape(-1,8000,1)

from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)





batch_size = 20
n_epochs = 80
n_classes = 18
learning_rate = 1e-3

import tensorflow as tf


def conv1dl(x, W, b, strides=1):
    x = tf.nn.conv1d(x, W, stride=1, padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.leaky_relu(x,alpha=0.20)

def conv1dt(x, W, b, strides=1):
    x = tf.nn.conv1d(x, W, stride=1, padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.math.tanh(x)


weights = {
    # Convolution Layers
    'c1': tf.get_variable('W1', shape=(13,1,8), \
            initializer=tf.contrib.layers.xavier_initializer()),
    'c2': tf.get_variable('W2', shape=(11,8,16), \
            initializer=tf.contrib.layers.xavier_initializer()),
    'c3': tf.get_variable('W3', shape=(9,16,32), \
            initializer=tf.contrib.layers.xavier_initializer()),
    'c4': tf.get_variable('W4', shape=(7,32,64), \
            initializer=tf.contrib.layers.xavier_initializer()),

    # Dense Layers
    'd1': tf.get_variable('W5', shape=(6080,256),
            initializer=tf.contrib.layers.xavier_initializer()),
    'd2': tf.get_variable('W52', shape=(256,128),
            initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('W6', shape=(128,n_classes),
            initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    # Convolution Layers
    'c1': tf.get_variable('B1', shape=(8), initializer=tf.zeros_initializer()),
    'c2': tf.get_variable('B2', shape=(16), initializer=tf.zeros_initializer()),
    'c3': tf.get_variable('B3', shape=(32), initializer=tf.zeros_initializer()),
    'c4': tf.get_variable('B4', shape=(64), initializer=tf.zeros_initializer()),

    # Dense Layers
    'd1': tf.get_variable('B5', shape=(256), initializer=tf.zeros_initializer()),
    'd2': tf.get_variable('B52', shape=(128), initializer=tf.zeros_initializer()),
    'out': tf.get_variable('B6', shape=(n_classes), initializer=tf.zeros_initializer()),
}

def conv_net(data, weights, biases, training=False):
    # Convolution layers
    conv1 = conv1dt(data, weights['c1'], biases['c1'])
    print(conv1.get_shape().as_list())
    pool11= tf.nn.max_pool1d(conv1, ksize=3, padding='VALID', strides=3)
    print(pool11.get_shape().as_list())


    conv2 = conv1dl(pool11, weights['c2'], biases['c2'])
    print(conv2.get_shape().as_list())
    pool12= tf.nn.max_pool1d(conv2, ksize=3, padding='VALID', strides=3)
    print(pool12.get_shape().as_list())

    conv3 = conv1dt(pool12, weights['c3'], biases['c3'])
    print(conv3.get_shape().as_list())
    pool21= tf.nn.max_pool1d(conv3, ksize=3, padding='VALID', strides=3)
    print(pool21.get_shape().as_list())

    conv4 = conv1dl(pool21, weights['c4'], biases['c4'])
    print(conv4.get_shape().as_list())
    pool22= tf.nn.max_pool1d(conv4, ksize=3, padding='VALID', strides=3)
    print(pool22.get_shape().as_list())

    # Flatten
    shapenn=pool22.get_shape().as_list()
    print(shapenn)
    flat = tf.reshape(pool22, [-1, weights['d1'].get_shape().as_list()[0]])

    # Fully connected layer
    fc1 = tf.add(tf.matmul(flat, weights['d1']), biases['d1'])
    fc1 = tf.nn.leaky_relu(fc1,alpha=0.2)

    fc2 = tf.add(tf.matmul(fc1 , weights['d2']), biases['d2'])
    fc2 = tf.nn.leaky_relu(fc2,alpha=0.2)


    # Output
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out



# Dataflow Graph
dataset = tf.data.Dataset.from_tensor_slices((x_tr,y_tr)).repeat().batch(batch_size)
iterator = dataset.make_initializable_iterator()
batch_images, batch_labels = iterator.get_next()
logits = conv_net(batch_images, weights, biases, training=True)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=batch_labels))
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)
test_predictions = tf.nn.softmax(conv_net(x_val, weights, biases))
acc,acc_op = tf.metrics.accuracy(predictions=tf.argmax(test_predictions,1), labels=y_val)

# Run Session
with tf.Session() as sess:
    # Initialize Variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(iterator.initializer)

    # Train the Model
    for epoch in range(n_epochs):
        prog_bar = tqdm(range(int(len(x_tr)/batch_size)))
        for step in prog_bar:
            _,cost = sess.run([train_op,loss])
            prog_bar.set_description("cost: {:.3f}".format(cost))
        accuracy = sess.run(acc_op)

        print('\nEpoch {} Accuracy: {:.3f}'.format(epoch+1, accuracy))
    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, './model_no_drop.ckpt')
