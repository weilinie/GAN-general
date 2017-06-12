__author__ = 'Weili Nie'


import tensorflow as tf
import tensorflow.contrib.layers as tcl
from utils import resBlock, leaky_relu, layer_norm


def generator(net, z, hidden_num, output_dim, out_channels, normalize_g, is_train=True):
    if net == 'ResNet':
        return generatorResNet(z, hidden_num, output_dim, out_channels)
    elif net == 'DCGAN':
        return generatorDCGAN(z, hidden_num, output_dim, out_channels, normalize_g, is_train)
    elif net == 'MLP':
        return generatorMLP(z, output_dim, out_channels, normalize_g, is_train)
    else:
        raise Exception('[!] Caution! unknown generator type.')


def discriminator(net, x, hidden_num, normalize_d, is_train=True, reuse=True):
    if net == 'ResNet':
        return discriminatorResNet(x, hidden_num, reuse)
    elif net == 'DCGAN':
        return discriminatorDCGAN(x, hidden_num, normalize_d, is_train, reuse)
    elif net == 'MLP':
        return discriminatorMLP(x, normalize_d, is_train, reuse)
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
def generatorDCGAN(z, hidden_num, output_dim, out_channels, normalize_g, is_train, kern_size=5):
    '''
    Default values:
    :param is_train: True
    :param is_batchnorm: True
    :param z: 128
    :param hidden_num: 64
    :param output_dim: 64
    :param kern_size: 5
    :param out_channels: 3
    :return:
    '''
    with tf.variable_scope("G") as vs:

        if normalize_g == 'BN':
            normalizer_fn = tcl.batch_norm
            normalizer_params = {'scale': True, 'is_training': is_train}
        elif normalize_g == 'LN':
            normalizer_fn = layer_norm
            normalizer_params = None
        else:
            normalizer_fn = None
            normalizer_params = None

        fc = tcl.fully_connected(
            z, hidden_num*8*(output_dim/16)*(output_dim/16),
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=tf.nn.relu
        )
        output = tf.reshape(fc, [-1, output_dim/16, output_dim/16, hidden_num*8])  # data_format: 'NHWC'

        output = tcl.conv2d_transpose(
            output, hidden_num*4, kern_size, stride=2,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=tf.nn.relu
        )

        output = tcl.conv2d_transpose(
            output, hidden_num*2, kern_size, stride=2,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=tf.nn.relu
        )

        output = tcl.conv2d_transpose(
            output, hidden_num, kern_size, stride=2,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=tf.nn.relu
        )

        gen_out = tcl.conv2d_transpose(
            output, out_channels, kern_size, stride=2,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=tf.nn.tanh
        )

    g_vars = tf.contrib.framework.get_variables(vs)
    return gen_out, g_vars


def discriminatorDCGAN(x, hidden_num, normalize_d, is_train, reuse, kern_size=5):
    with tf.variable_scope("D") as vs:
        if reuse:
            vs.reuse_variables()

        if normalize_d == 'BN':
            normalizer_fn = tcl.batch_norm
            normalizer_params = {'scale': True, 'is_training': is_train}
        elif normalize_d == 'LN':
            normalizer_fn = layer_norm
            normalizer_params = None
        else:
            normalizer_fn = None
            normalizer_params = None

        output = tcl.conv2d(
            x, hidden_num, kern_size, stride=2,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=leaky_relu
        )

        output = tcl.conv2d(
            output, hidden_num*2, kern_size, stride=2,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=leaky_relu
        )

        output = tcl.conv2d(
            output, hidden_num*4, kern_size, stride=2,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=leaky_relu
        )

        output = tcl.conv2d(
            output, hidden_num*8, kern_size, stride=2,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            activation_fn=leaky_relu
        )

        out_flt = tcl.flatten(output)  # data_format: 'NHWC'
        disc_out = tcl.fully_connected(out_flt, 1, activation_fn=None)

    d_vars = tf.contrib.framework.get_variables(vs)
    return disc_out, d_vars


# -------------------------------------------------
# +++++++++++++++++++++ MLP +++++++++++++++++++++++
# -------------------------------------------------
def generatorMLP(z, output_dim, out_channels, normalize_g, is_train, hidden_num=512, n_layers=3):
    '''
    Default values:
    :param is_train: True
    :param is_batchnorm: True
    :param z: 128
    :param output_dim: 64
    :param out_channels: 3
    :param hidden_num: 512
    :param n_layers: 3
    :return:
    '''
    with tf.variable_scope("G") as vs:

        if normalize_g == 'BN':
            normalizer_fn = tcl.batch_norm
            normalizer_params = {'scale': True, 'is_training': is_train}
        elif normalize_g == 'LN':
            normalizer_fn = layer_norm
            normalizer_params = None
        else:
            normalizer_fn = None
            normalizer_params = None

        output = tcl.fully_connected(
            z, hidden_num,
            normalizer_fn=normalizer_fn,
            normalizer_params=normalizer_params
        )
        for i in range(n_layers):
            output = tcl.fully_connected(
                output, hidden_num,
                normalizer_fn = normalizer_fn,
                normalizer_params = normalizer_params
            )
        fc = tcl.fully_connected(output, output_dim*output_dim*out_channels, activation_fn=tf.nn.tanh)
        gen_out = tf.reshape(fc, [-1, output_dim, output_dim, out_channels])

    g_vars = tf.contrib.framework.get_variables(vs)
    return gen_out, g_vars


def discriminatorMLP(x, normalize_d, is_train, reuse, hidden_num=512, n_layers=3):
    with tf.variable_scope("D") as vs:
        if reuse:
            vs.reuse_variables()

        if normalize_d == 'BN':
            normalizer_fn = tcl.batch_norm
            normalizer_params = {'scale': True, 'is_training': is_train}
        elif normalize_d == 'LN':
            normalizer_fn = layer_norm
            normalizer_params = None
        else:
            normalizer_fn = None
            normalizer_params = None

        x_flt = tcl.flatten(x)
        output = tcl.fully_connected(x_flt, hidden_num)
        for i in range(n_layers):
            output = tcl.fully_connected(
                output, hidden_num,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params
            )
        disc_out = tcl.fully_connected(output, 1, activation_fn=None)

    d_vars = tf.contrib.framework.get_variables(vs)
    return disc_out, d_vars