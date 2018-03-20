
import Dataset #in folder -> to 
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import cv2
from sklearn.utils import shuffle
import inspect
import os

path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"\\model"
path_name = path+"\\model_cat_dog"
img_size = 120

print(path_name)
## Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph(path_name+'.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint(path))
# Accessing the default graph which we have restored
graph = tf.get_default_graph()
# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")
## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, 2)) 

num_channels=3
imagesq = []
# Reading the image using OpenCV
image = cv2.imread("img/test/test.jpg")
# Resizing the image to our desired size and preprocessing will be done exactly as done during training
image = cv2.resize(image, (img_size, img_size),0,0, cv2.INTER_LINEAR)
imagesq.append(image)
imagesq = np.array(imagesq, dtype=np.uint8)
imagesq = imagesq.astype('float32')
imagesq = np.multiply(imagesq, 1.0/255.0) 
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = imagesq.reshape(1, img_size,img_size,num_channels)

feed_dict_testing = {x: x_batch, y_true: y_test_images}
with tf.Session() as session:
    saver.restore(session, path_name)
    result=session.run(y_pred, feed_dict=feed_dict_testing)
    print(result)

    if(result[0][0] > 0.5):
        print("E' un cane!")
    else:
        print("E' un gatto!")
    