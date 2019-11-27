#  tensorflow 1.x version
# 42 epochs 0.783 speech18 dataset

import numpy as np
#import tqdm as tdqm
from tqdm import tqdm
#import matplotlib.pyplot as plt

import os
import librosa
#import IPython.display as ipd
#import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
#import warnings

#warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

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


#from keras.utils import np_utils
#y=np_utils.to_categorical(y, num_classes=len(labels))



all_wave = np.array(all_wave).reshape(-1,8000,1)

from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)





batch_size = 20
n_epochs = 40
n_classes = 18
learning_rate = 1e-3

import tensorflow as tf


def conv1dl(x, W, b, strides=1):
#    x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')
    x = tf.nn.conv1d(x, W, stride=1, padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.leaky_relu(x,alpha=0.20)
    #return tf.math.tanh(x)

def conv1dt(x, W, b, strides=1):
#    x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')
    x = tf.nn.conv1d(x, W, stride=1, padding='VALID')
    x = tf.nn.bias_add(x, b)
    #return tf.nn.leaky_relu(x,alpha=0.20)
    return tf.math.tanh(x)



#relu -- 0.35
#leaky_relu alpha 0.2 -- 0.44
#leaky relu alpha 0.3 -- 0.43
#leaky_relu alpha 0.2 lr 1e-3 batch 20 -- 0.478
#leaky_relu alpha 0.2 lr 1e-3 batch 20 epoch 40 no_dropout -- 0.43

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
#    print(data.get_shape().as_list())
    conv1 = conv1dl(data, weights['c1'], biases['c1'])
#    pool11= tf.nn.max_pool(conv1, ksize=[1,1,3,1], padding='VALID',strides=[1,1,3,1])
    print(conv1.get_shape().as_list())
    pool11= tf.nn.max_pool1d(conv1, ksize=3, padding='VALID', strides=3)
    print(pool11.get_shape().as_list())

    if training:
        pool11 = tf.nn.dropout(pool11, rate=0.3)

    conv2 = conv1dl(pool11, weights['c2'], biases['c2'])
    print(conv2.get_shape().as_list())
#    pool12= tf.nn.max_pool(conv2, ksize=[1,1,3,1], padding='VALID',strides=[1,1,3,1])
    pool12= tf.nn.max_pool1d(conv2, ksize=3, padding='VALID', strides=3)
    print(pool12.get_shape().as_list())
    if training:
        pool12 = tf.nn.dropout(pool12, rate=0.3)

    conv3 = conv1dl(pool12, weights['c3'], biases['c3'])
    print(conv3.get_shape().as_list())
#    pool21= tf.nn.max_pool(conv3, ksize=[1,1,3,1], padding='VALID',strides=[1,1,3,1])
    pool21= tf.nn.max_pool1d(conv3, ksize=3, padding='VALID', strides=3)
    print(pool21.get_shape().as_list())
    if training:
        pool21 = tf.nn.dropout(pool21, rate=0.3)

    conv4 = conv1dl(pool21, weights['c4'], biases['c4'])
    print(conv4.get_shape().as_list())
#    pool22= tf.nn.max_pool(conv4, ksize=[1,1,3,1], padding='VALID',strides=[1,1,3,1])
    pool22= tf.nn.max_pool1d(conv4, ksize=3, padding='VALID', strides=3)
    print(pool22.get_shape().as_list())
    if training:
        pool22 = tf.nn.dropout(pool22, rate=0.3)

    # Flatten
    shapenn=pool22.get_shape().as_list()
    print(shapenn)
#    flat=tf.reshape(pool22, [-1, shapenn[1]*shapenn[2] ])
    flat = tf.reshape(pool22, [-1, weights['d1'].get_shape().as_list()[0]])
    #flat=tf.reshape(pool22, [-1,])
#    flat=tf.reshape(pool22, [-1, 6080] )
#    flat=tf.reshape(pool22,[int(pool22.shape[1]), int(pool22.shape[3]) ])
#    flat = tf.reshape(pool22, [-1, weights['d1'].shape[0]])
    # [7*7*32] = [1568]

    # Fully connected layer
    fc1 = tf.add(tf.matmul(flat, weights['d1']), biases['d1'])
    fc1 = tf.nn.leaky_relu(fc1,alpha=0.2)
    if training:
        fc1 = tf.nn.dropout(fc1, rate=0.3)

    fc2 = tf.add(tf.matmul(fc1 , weights['d2']), biases['d2'])
    fc2 = tf.nn.leaky_relu(fc2,alpha=0.2)



    # Dropout
    if training:
        #pool11 = tf.nn.dropout(pool11, rate=0.3)
        #pool12 = tf.nn.dropout(pool12, rate=0.3)
        #pool21 = tf.nn.dropout(pool21, rate=0.3)
        #pool22 = tf.nn.dropout(pool22, rate=0.3)

        #fc1 = tf.nn.dropout(fc1, rate=0.3)
        fc2 = tf.nn.dropout(fc2, rate=0.3)

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
    saver.save(sess, './modeltest1.h5')
