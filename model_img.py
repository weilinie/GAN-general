__author__ = 'Weili Nie'

import tensorflow as tf
import tensorflow.contrib.layers as tcl
from utils import resBlock


def generator(net, z, hidden_num, output_dim, out_channels):
    if net == 'ResNet':
        return generatorResNet(z, hidden_num, output_dim, out_channels)
    elif net == 'DCGAN':
        return generatorDCGAN(z, hidden_num, output_dim, out_channels)
    elif net == 'MLP':
        return generatorMLP(z, output_dim, out_channels)
    else:
        raise Exception('[!] Caution! unknown generator type.')


def discriminator(net, x, hidden_num, reuse=True):
    if net == 'ResNet':
        return discriminatorResNet(x, hidden_num, reuse)
    elif net == 'DCGAN':
        return discriminatorDCGAN(x, hidden_num, reuse)
    elif net == 'MLP':
        return discriminatorMLP(x, reuse)
    else:
        raise Exception('[!] Caution! unknown discriminator type.')


# --------------------------------------------------
# +++++++++++++++++++++ ResNet +++++++++++++++++++++
# --------------------------------------------------
def generatorResNet(z, hidden_num, output_dim, out_channels, kern_size=3):
    '''
    Default values:
    :param z: 128
    :param hidden_num: 64
    :param output_dim: 64
    :param kern_size: 3
    :param out_channels: 3
    :return:
    '''
    with tf.variable_scope("G") as vs:

        fc = tcl.fully_connected(z, hidden_num*8*(output_dim/16)*(output_dim/16), activation_fn=None)
        output = tf.reshape(fc, [-1, output_dim/16, output_dim/16, hidden_num*8]) # data_format: 'NHWC'

        for i in range(6):
            output = resBlock(output, hidden_num*8, hidden_num*8, kern_size)
        output = resBlock(output, hidden_num*8, hidden_num*4, kern_size, resample='up')
        for i in range(6):
            output = resBlock(output, hidden_num*4, hidden_num*4, kern_size)
        output = resBlock(output, hidden_num*4, hidden_num*2, kern_size, resample='up')
        for i in range(6):
            output = resBlock(output, hidden_num*2, hidden_num*2, kern_size)
        output = resBlock(output, hidden_num*2, hidden_num, kern_size, resample='up')
        for i in range(6):
            output = resBlock(output, hidden_num, hidden_num, kern_size)
        output = resBlock(output, hidden_num, hidden_num/2, kern_size, resample='up')
        for i in range(5):
            output = resBlock(output, hidden_num/2, hidden_num/2, kern_size)

        gen_out = tcl.conv2d(output, out_channels, 1, activation_fn=tf.nn.tanh)

    g_vars = tf.contrib.framework.get_variables(vs)
    return gen_out, g_vars


def discriminatorResNet(x, hidden_num, reuse, kern_size=3):
    with tf.variable_scope("D") as vs:
        if reuse:
            vs.reuse_variables()

        output = tcl.conv2d(x, hidden_num/2, 1)
        for i in range(5):
            output = resBlock(output, hidden_num/2, hidden_num/2, kern_size)
        output = resBlock(output, hidden_num/2, hidden_num, kern_size, resample='down')
        for i in range(6):
            output = resBlock(output, hidden_num, hidden_num, kern_size)
        output = resBlock(output, hidden_num, hidden_num*2, kern_size, resample='down')
        for i in range(6):
            output = resBlock(output, hidden_num*2, hidden_num*2, kern_size)
        output = resBlock(output, hidden_num*2, hidden_num*4, kern_size, resample='down')
        for i in range(6):
            output = resBlock(output, hidden_num*4, hidden_num*4, kern_size)
        output = resBlock(output, hidden_num*4, hidden_num*8, kern_size, resample='down')
        for i in range(6):
            output = resBlock(output, hidden_num*8, hidden_num*8, kern_size)

        out_flt = tcl.flatten(output)  # data_format: 'NHWC'
        disc_out = tcl.fully_connected(out_flt, 1, activation_fn=None)

    d_vars = tf.contrib.framework.get_variables(vs)
    return disc_out, d_vars


# ---------------------------------------------------
# +++++++++++++++++++++ DCGAN +++++++++++++++++++++++
# ---------------------------------------------------
def generatorDCGAN(z, hidden_num, output_dim, out_channels, kern_size=5):
    '''
    Default values:
    :param z: 128
    :param hidden_num: 64
    :param output_dim: 64
    :param kern_size: 5
    :param out_channels: 3
    :return:
    '''
    with tf.variable_scope("G") as vs:

        fc = tcl.fully_connected(z, hidden_num*8*(output_dim/16)*(output_dim/16), activation_fn=None)
        output = tf.reshape(fc, [-1, output_dim/16, output_dim/16, hidden_num*8])  # data_format: 'NHWC'

        output = tcl.conv2d_transpose(output, hidden_num*4, kern_size, stride=2)
        output = tcl.conv2d_transpose(output, hidden_num*2, kern_size, stride=2)
        output = tcl.conv2d_transpose(output, hidden_num, kern_size, stride=2)
        gen_out = tcl.conv2d_transpose(output, out_channels, kern_size, stride=2, activation_fn=tf.nn.tanh)

    g_vars = tf.contrib.framework.get_variables(vs)
    return gen_out, g_vars


def discriminatorDCGAN(x, hidden_num, reuse, kern_size=5):
    with tf.variable_scope("D") as vs:
        if reuse:
            vs.reuse_variables()

        output = tcl.conv2d(x, hidden_num, kern_size, stride=2, activation_fn=tf.nn.elu)
        output = tcl.conv2d(output, hidden_num*2, kern_size, stride=2, activation_fn=tf.nn.elu)
        output = tcl.conv2d(output, hidden_num*4, kern_size, stride=2, activation_fn=tf.nn.elu)
        output = tcl.conv2d(output, hidden_num*8, kern_size, stride=2, activation_fn=tf.nn.elu)

        out_flt = tcl.flatten(output)  # data_format: 'NHWC'
        disc_out = tcl.fully_connected(out_flt, 1, activation_fn=None)

    d_vars = tf.contrib.framework.get_variables(vs)
    return disc_out, d_vars


# -------------------------------------------------
# +++++++++++++++++++++ MLP +++++++++++++++++++++++
# -------------------------------------------------
def generatorMLP(z, output_dim, out_channels, hidden_num=512, n_layers=3):
    with tf.variable_scope("G") as vs:

        output = tcl.fully_connected(z, hidden_num)
        for i in range(n_layers):
            output = tcl.fully_connected(output, hidden_num)
        fc = tcl.fully_connected(output, output_dim*output_dim*out_channels, activation_fn=tf.nn.tanh)
        gen_out = tf.reshape(fc, [-1, output_dim, output_dim, out_channels])

    g_vars = tf.contrib.framework.get_variables(vs)
    return gen_out, g_vars


def discriminatorMLP(x, reuse, hidden_num=512, n_layers=3):
    with tf.variable_scope("D") as vs:
        if reuse:
            vs.reuse_variables()

        # x_flt = tf.reshape(x, [x.get_shape().as_list()[0], -1])
        x_flt = tcl.flatten(x)
        output = tcl.fully_connected(x_flt, hidden_num)
        for i in range(n_layers):
            output = tcl.fully_connected(output, hidden_num)
        disc_out = tcl.fully_connected(output, 1, activation_fn=None)

    d_vars = tf.contrib.framework.get_variables(vs)
    return disc_out, d_vars