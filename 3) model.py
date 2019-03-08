import tensorflow as tf
import scipy

def weight_variable(shape):
    init = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    init = tf.constant(0.1,shape=shape)
    return tf.Variable(init)

def conv2d(x,W,stride):
    return tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='VALID')

x = tf.placeholder(tf.float32,shape=[None,66,200,3])
y_ = tf.placeholder(tf.float32,shape=[None,1])
keep_prob = tf.placeholder(tf.float32)

x_image = x

#1st Conv Layer
w_1 = weight_variable([5,5,3,24])
b_1 = bias_variable([24])
h_1 = tf.nn.relu(conv2d(x_image,w_1,2)+b_1)

#2nd Conv Layer
w_2 = weight_variable([5,5,24,36])
b_2 = bias_variable([36])
h_2 = tf.nn.relu(conv2d(h_1,w_2)+b_2)

#3rd Conv Layer
w_3 = weight_variable([5,5,36,48])
b_3 = bias_variable([48])
h_3 = tf.nn.relu(conv2d(h_2,w_3)+b_3)

#4th Conv Layerpt
w_4 = weight_variable([3,3,48,64])
b_4 = bias_variable([64])
h_4 = tf.nn.relu(conv2d(h_3,w_4)+b_4)

#5th Conv Layer
w_5 = weight_variable([3,3,64,64])
b_5 = bias_variable([64])
h_5 = tf.nn.relu(conv2d(h_4,w_5)+b_5)

#Fully Connected Layer1
w_fc1 = weight_variable([1152,1164]) #Flatten
b_fc1 = bias_variable([1164])
h_5_flat = tf.reshape(h_5,[-1,1152])
h_fc1 = tf.nn.relu(tf.matmul(h_5_flat,w_fc1) + b_fc1)
#DROPOUT
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Fully Connected Layer2
w_fc2 = weight_variable([1164,100]) #Flatten
b_fc2 = bias_variable([100])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop,w_fc2) + b_fc2)
#DROPOUT
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

#Fully Connected Layer3
w_fc3 = weight_variable([100,50]) #Flatten
b_fc3 = bias_variable([50])
h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop,w_fc3) + b_fc3)
#DROPOUT
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

#Fully Connected Layer4
w_fc4 = weight_variable([50,10]) #Flatten
b_fc4 = bias_variable([10])
h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop,w_fc4) + b_fc4)
#DROPOUT
h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

#output
w_fc5 = weight_variable([10,1])
b_fc5 = bias_variable([1])
y = tf.multiply(tf.atan(tf.matmul(h_fc4_drop,w_fc5)+b_fc5),2)