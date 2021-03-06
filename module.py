from __future__ import division
import tensorflow as tf
from ops import *

def u_net_model(image, options, reuse=False, name="U-Net", is_test=False):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False
        
        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(image, options.gf_dim, name='g_e1_conv')
        # e1 is (128 x 128 x self.gf_dim)
        e2 = instance_norm(conv2d(lrelu(e1), options.gf_dim*2, name='g_e2_conv'), 'g_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = instance_norm(conv2d(lrelu(e2), options.gf_dim*4, name='g_e3_conv'), 'g_bn_e3')
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = instance_norm(conv2d(lrelu(e3), options.gf_dim*8, name='g_e4_conv'), 'g_bn_e4')
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = instance_norm(conv2d(lrelu(e4), options.gf_dim*8, name='g_e5_conv'), 'g_bn_e5')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = instance_norm(conv2d(lrelu(e5), options.gf_dim*8, name='g_e6_conv'), 'g_bn_e6')
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = instance_norm(conv2d(lrelu(e6), options.gf_dim*8, name='g_e7_conv'), 'g_bn_e7')
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = instance_norm(conv2d(lrelu(e7), options.gf_dim*8, name='g_e8_conv'), 'g_bn_e8')
        # e8 is (1 x 1 x self.gf_dim*8)
        
        probkeep_dropout= 0.5
        if is_test:
            probkeep_dropout=1
        

        d1 = deconv2d(tf.nn.relu(e8), options.gf_dim*8, name='g_d1')
        d1 = tf.concat([tf.nn.dropout(instance_norm(d1, 'g_bn_d1'), probkeep_dropout), e7], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2 = deconv2d(tf.nn.relu(d1), options.gf_dim*8, name='g_d2')
        d2 = tf.concat([tf.nn.dropout(instance_norm(d2, 'g_bn_d2'), probkeep_dropout), e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3 = deconv2d(tf.nn.relu(d2), options.gf_dim*8, name='g_d3')
        d3 = tf.concat([tf.nn.dropout(instance_norm(d3, 'g_bn_d3'), probkeep_dropout), e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4 = deconv2d(tf.nn.relu(d3), options.gf_dim*8, name='g_d4')
        d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = deconv2d(tf.nn.relu(d4), options.gf_dim*4, name='g_d5')
        d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = deconv2d(tf.nn.relu(d5), options.gf_dim*2, name='g_d6')
        d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = deconv2d(tf.nn.relu(d6), options.gf_dim, name='g_d7')
        d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        d8 = deconv2d(tf.nn.relu(d7), options.num_classes, name='g_d8')
        # d8 is (256 x 256 x output_c_dim)

        return d8

def sem_criterion(logits,labels,num_labels):
    labels = tf.squeeze(labels, axis=3)
    mask = tf.cast(tf.where(tf.greater_equal(labels ,tf.ones_like(labels)* num_labels), tf.zeros_like(labels), tf.ones_like(labels)),tf.float32)
    labels = tf.where(tf.greater(labels ,tf.ones_like(labels)* num_labels), tf.zeros_like(labels), labels)
    labels = tf.cast(labels,tf.int32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.multiply(loss, mask)
    return tf.reduce_sum(loss)/(tf.reduce_sum(mask)+0.0000000001)

def accuracy_op(logits, labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,axis=-1),tf.cast(tf.squeeze(labels,axis=-1),tf.int64)),tf.float32))