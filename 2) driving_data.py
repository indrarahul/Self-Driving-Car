import scipy.misc
import random

x = []
y = []

train_batch_pointer = 0
test_batch_pointer = 0

with open("driving_dataset/data.txt") as file:
    for line in file.readlines():
        x.append("driving_dataset/"+line.split()[0])
        y.append(float(line.split()[1]) * scipy.pi / 180)

num_of_images = len(x)

train_x = x[:int(len(x)*0.8)]
train_y = y[:int(len(x)*0.8)]

test_x = x[int(len(x)*0.8):]
test_y = y[int(len(x)*0.8):]

num_of_train_images = len(train_x)
num_of_test_images = len(test_x)

def LoadBatchFromTraining(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(train_x[(train_batch_pointer + i) % num_of_train_images])[-150:],[66,200])/255.0)
        y_out.append([train_y[(train_batch_pointer + i) % num_of_train_images]])
        train_batch_pointer += batch_size  
    return x_out, y_out

def LoadBatchFromTest(batch_size):
    global test_batch_pointer
    x_out = []
    y_out = []
    for i in range(batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(test_x[(test_batch_pointer + i ) % num_of_test_images])[-150:],[66,200])/255.0)
        y_out.append([test_y[(test_batch_pointer + i) % num_of_test_images]])
        test_batch_pointer += batch_size
    return x_out, y_out