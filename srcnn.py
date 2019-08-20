import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import scipy
import cv2
from skimage import measure

import pdb





def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)




def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def preprocess(path, scale=3):
  """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation
  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
  image = imread(path, is_grayscale=True)
  label_ = modcrop(image, scale)

  # Must be normalized
  image = image / 255.
  label_ = label_ / 255.

  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

  return input_, label_


##
path = 'C:/Users/Administrator/Desktop/deep_learning (copy)/CW1_for_students/CW1_Handout_Template_code/tf-SRCNN/image/butterfly_GT.bmp'
butterfly = imread(path,False)
print(butterfly.shape)
#pdb.set_trace()
##change image to gray scale
butterfly_gray = imread(path,True)

##shrink to 1/3
butterfly_shrink = scipy.misc.imresize(butterfly_gray,1/3)

#pdb.set_trace()
#butterfly_shrink =tf.image.resize_images(butterfly_gray,1/3,method = np.random.randint(4))
##large with 3 times
butterfly_final = scipy.ndimage.interpolation.zoom(butterfly_shrink, (3/1.), prefilter=False)
#butterfly_final = tf.shape(tf.expand_dims(butterfly_final,-1))
#print(butterfly_final.shape)
plt.imshow(butterfly_final, cmap='Greys_r')
plt.show()
#pdb.set_trace()


"""Set the image hyper parameters
"""
c_dim = 1
input_size = 255

"""Define the model weights and biases 
"""

# define the placeholders for inputs and outputs
inputs = tf.placeholder(tf.float32, [None, input_size, input_size, c_dim], name='inputs')

# conv1 layer with biases: 64 filters with size 9 x 9
# conv2 layer with biases and relu: 32 filters with size 1 x 1
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
weights = {
    'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
    'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
    'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
    }

biases = {
      'b1': tf.Variable(tf.zeros([64]), name='b1'),
      'b2': tf.Variable(tf.zeros([32]), name='b2'),
      'b3': tf.Variable(tf.zeros([1]), name='b3')
    }

"""Define the model layers with three convolutional layers
"""

# conv1 layer with biases and relu : 64 filters with size 9 x 9

conv1 = tf.nn.relu(tf.nn.conv2d(inputs,weights['w1'],strides=[1,1,1,1],padding='VALID')+biases['b1'])

# conv2 layer with biases and relu: 32 filters with size 1 x 1
conv2 = tf.nn.relu(tf.nn.conv2d(conv1,weights['w2'],strides=[1,1,1,1],padding='VALID')+biases['b2'])

# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
conv3 = tf.nn.conv2d(conv2,weights['w3'],strides=[1,1,1,1],padding='VALID')+biases['b3']


"""Load the pre-trained model file
"""
#model_path='./model/model.npy'
model_path = 'C:/Users/Administrator/Desktop/deep_learning (copy)/CW1_for_students/CW1_Handout_Template_code/tf-SRCNN/model/model.npy'
model = np.load(model_path, encoding='latin1').item()

# variabiles (w1, w2, w3)
w1 = model['w1']
w2 = model['w2']
w3 = model['w3']

b1 = model['b1']
b2 = model['b2']
b3 = model['b3']


##
print('the value of the 1st filter')
print(w1[:,:,:,0])

##
print('the bias of the 10th filter')
print(b1[9])

#
print('the value of the 5st filter ')
print(w2[:,:,:,4])
# To show the bias of the 6th filter
print('the bias of the 6th bias ')
print(b2[5])
## the channel number is 64 depends on previous layer

# To show the value of the 1st filter
print('the value of the 1st filter  ')
print(w3[:,:,:,0])
# To show the bias of the 1th filter
print('the bias of the 10th bias  ')
print(b3[0])


"""Initialize the model variabiles (w1, w2, w3, b1, b2, b3) with the pre-trained model file
"""
# launch a session
variable = tf.initialize_all_variables()
sess = tf.Session()
sess.run(variable)

for key in weights.keys():
  sess.run(weights[key].assign(model[key]))

for key in biases.keys():
  sess.run(biases[key].assign(model[key]))

"""Read the test image
"""
blurred_image, groudtruth_image =  preprocess('./image/butterfly_GT.bmp')

"""Run the model and get the SR image
"""
# transform the input to 4-D tensor
input_ = np.expand_dims(np.expand_dims(blurred_image, axis =0), axis=-1)

# run the session
# here you can also run to get feature map like 'conv1' and 'conv2'
output_ = sess.run(conv3, feed_dict={inputs: input_})
plt.figure(1)
plt.title('High-Resolution')
plt.imshow(output_[0,:,:,0],cmap='Greys_r') # show the high resolution
plt.show()

# hints: use the 'scipy.misc.imsave()'  and ' skimage.meause.compare_psnr()'

ground_t = scipy.misc.imsave('ground_truth.jpg',groudtruth_image)

blurred = scipy.misc.imsave('blurred_image.jpg',blurred_image)


#question3.6
groudtruth_image=cv2.resize(groudtruth_image,butterfly_final.shape)
baseline_result = measure.compare_psnr(butterfly_final,groudtruth_image)
print(baseline_result)

# question3.8
output = output_.squeeze()
groudtruth_image = cv2.resize(groudtruth_image,output.shape)
gt_out = measure.compare_psnr(output,groudtruth_image)
print(gt_out)

# show ground truth
plt.figure(2)
plt.title('ground_truth')
plt.imshow(groudtruth_image,cmap='Greys_r')
plt.show()
# show baseline result
plt.figure(3)
plt.title('baseline result')
plt.imshow(butterfly_final,cmap='Greys_r')
plt.show()

#show hr-srcnn
plt.figure(4)
plt.title('HR-SRCNN')
plt.imshow(output_[0,:,:,0],cmap='Greys_r')
plt.show()
