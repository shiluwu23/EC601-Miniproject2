import glob

import tensorflow as tf
import numpy as np
import os, cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

image_size = 64
num_channels = 3
images = []

path = '003.jpg'
image = cv2.imread(path)
# Resizing the image to our desired size and preprocessing will be done exactly as done during training
image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0)
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(1, image_size,image_size,num_channels)

sess = tf.Session()


saver = tf.train.import_meta_graph('./sunflowertulip-model/sunflowertulip-model.ckpt-966.meta')


saver.restore(sess, './sunflowertulip-model/sunflowertulip-model.ckpt-966')


graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name("y_pred:0")

x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, 2))

feed_dict_testing = {x: x_batch, y_true: y_test_images}
result = sess.run(y_pred, feed_dict=feed_dict_testing)

res_label = ['sunflower', 'tulip']
print(res_label[result.argmax()])