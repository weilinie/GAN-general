import tensorflow as tf
import tensorflow.contrib.layers as tcl
from utils import resBlock


def generator(net, z, hidden_num, output_dim, out_channels):
    if net == 'ResNet':
        return generatorResNet(z, hidden_num, output_dim, out_channels)
    elif net == 'DCGAN':
        return generatorDCGAN(z, hidden_num, output_dim, out_channels)
    elif net == 'MLP':
        return generatorMLP(z, hidden_num, output_dim, out_channels)
    else:
        raise Exception('[!] Caution! unknown generator type.')


def discriminator(net, x, hidden_num, output_dim, reuse=True):
    if net == 'ResNet':
        return discriminatorResNet(x, hidden_num, output_dim, reuse)
    elif net == 'DCGAN':
        return discriminatorDCGAN(x, hidden_num, output_dim, reuse)
    elif net == 'MLP':
        return discriminatorMLP(x, hidden_num, output_dim, reuse)
    else:
        raise Exception('[!] Caution! unknown discriminator type.')


# --------------------------------------------------
# ------------------ ResNet ------------------------
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


def discriminatorResNet(x, hidden_num, output_dim, kern_size=3, reuse=None):
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

        output = tf.reshape(output, [-1, hidden_num*8*(output_dim/16)*(output_dim/16)])  # data_format: 'NHWC'
        disc_out = tcl.fully_connected(output, 1, activation_fn=None)

    d_vars = tf.contrib.framework.get_variables(vs)
    return disc_out, d_vars


# ---------------------------------------------------
# -------------------- DCGAN ------------------------
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


def discriminatorDCGAN(x, hidden_num, output_dim, kern_size=5, reuse=None):
    with tf.variable_scope("D") as vs:
        if reuse:
            vs.reuse_variables()
        output = tcl.conv2d(x, hidden_num, kern_size, stride=2, activation_fn=tf.nn.elu)
        output = tcl.conv2d(output, hidden_num*2, kern_size, stride=2, activation_fn=tf.nn.elu)
        output = tcl.conv2d(output, hidden_num*4, kern_size, stride=2, activation_fn=tf.nn.elu)
        output = tcl.conv2d(output, hidden_num*8, kern_size, stride=2, activation_fn=tf.nn.elu)

        output = tf.reshape(output, [-1, hidden_num*8*(output_dim/16)*(output_dim/16)])  # data_format: 'NHWC'
        disc_out = tcl.fully_connected(output, 1, activation_fn=None)

    d_vars = tf.contrib.framework.get_variables(vs)
    return disc_out, d_vars


# -------------------------------------------------
# -------------------- MLP ------------------------
# -------------------------------------------------
def generatorMLP(z, hidden_num, output_dim, out_channels):
    with tf.variable_scope("G") as vs:
        pass

def discriminatorMLP(x, hidden_num, output_dim, reuse=None):
    with tf.variable_scope("D") as vs:
        if reuse:
            vs.reuse_variables()
            pass