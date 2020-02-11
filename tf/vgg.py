# <Imports..

from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
import numpy as np

# ..stropmI>

"""
<VGG..
    Implementation of VGG16
..GGV>
"""
class vgg():
    def _init__(self, x, keep_prob, num_classes):
        
        
        """
        <Parameters..
            x :: input tensor placeholder
            keep_prob :: dropout prob
            num_classes :: number of differnt classifications
        ..sretemaraP>
        """
        
        
        self.X = x
        self.keep_prob = keep_prob
        self.num_classes = num_classes
        
        self.build_vgg()
    

#<Conv-Layer..
    def conv(self, x, num_filt, name, h_filt=3, w_filt=3, stride=1, pad="same"):    
        
        """
        <Parameters..
            x :: input tensor placeholder
            num_filt :: number of filters/channels
            name :: name of the layer
            h_filt, w_filt :: dimensions of filter
            stride :: stride..
            pad :: pad 
        ..sretemaraP>
        """

        incoming_channels = int(x.get_shape()[-1])
        with tf.variable_scope(name) as scope:
            # Weights and Bias variables
            W = tf.get_variable('wts', shape = [filter_height, filter_width, input_channels,
                            num_filters],initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01))
            b = tf.get_variable('biases', shape = [num_filters], initializer = tf.constant_initializer(0.0))
            #conv layer
            conv = tf.nn.bias_add(tf.rnn.conv2d(x, W, strides=[1,stride,stride,1], paddings=pad, name=name), b)
            a_relu = tf.nn.relu(conv)
        return a_relu
#..reyaL-Conv>

# <FullyConnectedLayer..
    def fc(self, x, input_size, output_size, name, relu=True):
        """
        <Parameters..
            x :: input tensor placeholder
            input_size :: number of neurons in prev layer
            output_size :: number of neurons in layer
            name :: name of the layer
            relu :: use ReLU as activation func else Sigmoid
        ..sretemaraP>
        """
        with tf.variable_scope(name) as scope:
            # Weights and Bias variables
            W = tf.get_variable('wts', shape = [input_size, output_size], initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01))
            b = tf.get_variable('biases', shape = [output_size], initializer = tf.constant_initializer(0.0))
            #conv layer
            fc = tf.nn.bias_add(tf.matmul(x, W), b)
            if relu:
                a = tf.nn.relu(fc)
            else:
                a = tf.nn.sigmoid(fc)
        return a
# ..reyaldetcennoCylluF>
        
# <MaxPool..
    def maxpool(self, x, name):
        """
        <Parameters..
            x :: input tensor
            name :: name of the layer
            Not needed.. all set to defaults 
            can be overridden later, I don't feel i 
            need any..
        ..sretemaraP>
        """
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "valid", name = name)

# ..looPxaM>
        
# <Dropout..
    def dropout(self, x, name):
        """
        <Parameters..
            x :: input tensor
            name :: name of the layer.
        ..sretemaraP>
        """
        return tf.nn.dropout(x, keep_prob = self.keep_prob)
# ..tuoporD>

# <Build_VGG..
    def build_vgg(self):
        # <Block1..
        conv1_1 = self.conv(self.X, 64, "conv1_1")
        conv1_2 = self.conv(conv1_1, 64, "conv1_2")
        max_pool1 = self.max_pool(conv1_2, "pool1")
        # ..1kcolB>
        # <Block2..
        conv2_1 = self.conv(max_pool1, 128, "conv2_1")
        conv2_2 = self.conv(conv2_1, 128, "conv2_2")
        max_pool2 = self.max_pool(conv2_2, "pool2")
        # ..2kcolB>
        # <Block3..
        conv3_1 = self.conv(max_pool2, 256, "conv3_1")
        conv3_2 = self.conv(conv3_1, 256, "conv3_2")
        conv3_3 = self.conv(conv3_2, 256, "conv3_3")
        max_pool3 = self.max_pool(conv3_3, "pool3")
        # ..3kcolB>
        # <Block4..
        conv4_1 = self.conv(max_pool3, 512, "conv4_1")
        conv4_2 = self.conv(conv4_1, 512, "conv4_2")
        conv4_3 = self.conv(conv4_2, 512, "conv4_3")
        max_pool4 = self.max_pool(conv4_3, "pool4")
        # ..4kcolB>
        # <Block5..
        conv5_1 = self.conv(max_pool4, 512, "conv5_1")
        conv5_2 = self.conv(conv5_1, 512, "conv5_2")
        conv5_3 = self.conv(conv5_2, 512, "conv5_3")
        max_pool5 = self.max_pool(conv5_3, "pool5")
        # ..5kcolB>
        flattened = tf.reshape(max_pool5, [self.X.shape[0], -1])
        # <Block6..
        
        # ..6kcolB>
        
# ..GGVdliuB>
        
""" 
<VGG16_load..
    This is already existing model loaded from
    tf.keras.applications.vgg16.VGG16
..doal_61GGV> 
"""

class Vgg16_load():
    def __init__(self):
        self.vgg = VGG16()
        super().__init__()
        
    def get_VGG(self):
        return self.vgg