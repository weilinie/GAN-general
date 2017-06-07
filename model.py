import tensorflow as tf
import tensorflow.contrib.layers as tcl


def generator(z, hidden_num, output_dim, kern_size, out_channels):
    with tf.variable_scope("G") as vs:
        fc = tcl.fully_connected(z, hidden_num*output_dim, activation_fn=None)
        fc = tf.reshape(fc, [-1, output_dim, hidden_num]) # data_format: 'NWC'
        res1 = resBlock(fc, hidden_num, kern_size)
        res2 = resBlock(res1, hidden_num, kern_size)
        res3 = resBlock(res2, hidden_num, kern_size)
        res4 = resBlock(res3, hidden_num, kern_size)
        res5 = resBlock(res4, hidden_num, kern_size)
        logits = tcl.conv2d(res5, out_channels, kernel_size=1)
        fake_data_softmax = tf.reshape(
            tf.nn.softmax(tf.reshape(logits, [-1, out_channels])),
            tf.shape(logits)
        )

    g_vars = tf.contrib.framework.get_variables(vs)
    return fake_data_softmax, g_vars


def discriminator(x, hidden_num, output_dim, kern_size, in_channels, reuse=True):
    with tf.variable_scope("D") as vs:
        if reuse:
            vs.reuse_variables()
        conv = tcl.conv2d(x, hidden_num, kernel_size=1)

        res1 = resBlock(conv, hidden_num, kern_size)
        res2 = resBlock(res1, hidden_num, kern_size)
        res3 = resBlock(res2, hidden_num, kern_size)
        res4 = resBlock(res3, hidden_num, kern_size)
        res5 = resBlock(res4, hidden_num, kern_size)

        res5 = tf.reshape(res5, [-1, output_dim*hidden_num])  # data_format: 'NWC'
        disc_out = tcl.fully_connected(res5, 1, activation_fn=None)

    d_vars = tf.contrib.framework.get_variables(vs)
    return disc_out, d_vars


def resBlock(inputs, hidden_num, kernerl_size):
    output = inputs
    output = tcl.conv2d(inputs=output, num_outputs=hidden_num, kernel_size=kernerl_size)
    output = tcl.conv2d(inputs=output, num_outputs=hidden_num, kernel_size=kernerl_size)
    return inputs + (0.3*output)

