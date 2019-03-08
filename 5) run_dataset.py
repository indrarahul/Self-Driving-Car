import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call
import math

sess= tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess,"save/model.ckpt")

img = cv2.imread('str.jpg',0)
rows,cols = img.shape

smoothAngle =0

x = []
y = []

with open("driving_dataset/data.txt") as file:
    for line in file.readlines():
        x.append("driving_dataset/"+ line.split()[0])
        y.append(float(line.split()[1])* scipy.pi / 180)

num_of_images = len(x)
i = math.ceil(num_of_images*0.8)
print("DATASET FROM " + str(i))

while(True):
    full_image = scipy.misc.imread("driving_dataset/" + str(i) + ".jpg", mode="RGB")
    image = scipy.misc.imresize(full_image[-150:],[66,200])/255.0
    degrees = model.y.eval(feed_dict={model.x:[image],model.keep_prob:1.0})[0][0] * 180.0 / scipy.pi
    print("Steering Angle : Prediction - " + str(degrees) + " Actual - " + str(y[i]*180/scipy.pi))
    cv2.imshow("frame",cv2.cvtColor(full_image,cv2.COLOR_RGB2RGB)
    
    #FOR SMOOTHING TO SMOOTH STEERING WHEEL EFFECT
    smoothAngle += 0.2 * pow(abs((degrees - smoothAngle)), 2.0 / 3.0) * (degrees - smoothAngle) / abs(degrees - smoothAngle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothAngle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    if(cv2.waitKey(10) == ord('q')):
        break