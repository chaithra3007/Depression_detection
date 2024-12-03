import math
import tensorflow as tf
import numpy as np
from keras import initializers, layers
import tensorflow as tf
import sys
K=tf.keras.backend 
cluster_size=16
class NetVLAD(layers.Layer):
    """Creates a NetVLAD class.
    """
    def __init__(self, feature_size, max_samples, cluster_size, output_dim, **kwargs):
        
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.cluster_size = cluster_size
        super(NetVLAD, self).__init__(**kwargs)

    def build(self, input_shape):
    # Create a trainable weight variable for this layer.
        self.cluster_weights = self.add_weight(name='kernel_W1',
                                      shape=(self.feature_size, self.cluster_size),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)),
                                      trainable=True)
        self.cluster_biases = self.add_weight(name='kernel_B1',
                                      shape=(self.cluster_size,),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)),
                                      trainable=True)
        self.cluster_weights2 = self.add_weight(name='kernel_W2',
                                      shape=(1,self.feature_size, self.cluster_size),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)),
                                      trainable=True)
        self.hidden1_weights = self.add_weight(name='kernel_H1',
                                      shape=(self.cluster_size*self.feature_size, self.output_dim),
                                      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.cluster_size)),
                                      trainable=True)
        
        super(NetVLAD, self).build(input_shape)  # Be sure to call this at the end

    def call(self, reshaped_input):
        """Forward pass of a NetVLAD block.

        Args:
        reshaped_input: If your input is in that form:
        'batch_size' x 'max_samples' x 'feature_size'
        It should be reshaped in the following form:
        'batch_size*max_samples' x 'feature_size'
        by performing:
        reshaped_input = tf.reshape(input, [-1, features_size])

        Returns:
        vlad: the pooled vector of size: 'batch_size' x 'output_dim'
        """
        """
        In Keras, there are two way to do matrix multiplication (dot product)
        1) K.dot : AxB -> when A has batchsize and B doesn't, use K.dot
        2) tf.matmul: AxB -> when A and B both have batchsize, use tf.matmul
        
        Error example: Use tf.matmul when A has batchsize (3 dim) and B doesn't (2 dim)
        ValueError: Shape must be rank 2 but is rank 3 for 'net_vlad_1/MatMul' (op: 'MatMul') with input shapes: [?,21,64], [64,3]
        
        tf.matmul might still work when the dim of A is (?,64), but this is too confusing.
        Just follow the above rules.
        """
        activation = K.dot(reshaped_input, self.cluster_weights)
        
        activation += self.cluster_biases
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation,
                [-1, self.max_samples, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keepdims=True)
        
        a = tf.multiply(a_sum,self.cluster_weights2)
        
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,
            self.max_samples, self.feature_size])

        vlad = tf.matmul(activation,reshaped_input)
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.subtract(vlad,a)
        vlad = tf.nn.l2_normalize(vlad,1)
        vlad = tf.reshape(vlad,[-1, self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)
        vlad = K.dot(vlad, self.hidden1_weights)

        return vlad

    def compute_output_shape(self, input_shape):
        return tuple([None, self.output_dim])
