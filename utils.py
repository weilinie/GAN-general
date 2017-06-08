from __future__ import print_function

import functools
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl


def prepare_dirs(config, dataset):

    config.model_name = "{}_{}".format(dataset, datetime.now().strftime("%m%d_%H%M%S"))

    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, config.model_name)

    if not hasattr(config, 'data_path'):
        config.data_path = os.path.join(config.data_dir, dataset)

    for dir in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)


# language dataset iterator
def inf_train_gen(lines, batch_size, charmap):
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines)-batch_size+1, batch_size):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i+batch_size]],
                dtype=np.int32
            )


def resBlock(inputs, input_num, output_num, kernel_size, resample=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_shortcut = functools.partial(tcl.conv2d, stride=2)
        conv_1 = functools.partial(tcl.conv2d, num_outputs=input_num/2)
        conv_1b = functools.partial(tcl.conv2d, num_outputs=output_num/2, stride=2)
        conv_2 = functools.partial(tcl.conv2d, num_outputs=output_num)
    elif resample == 'up':
        conv_shortcut = subpixelConv2D
        conv_1 = functools.partial(tcl.conv2d, output_dim=input_num/2)
        conv_1b = functools.partial(tcl.conv2d_transpose, num_outputs=output_num/2, stride=2)
        conv_2 = functools.partial(tcl.conv2d, num_outputs=output_num)
    elif resample == None:
        conv_shortcut = tcl.conv2d
        conv_1 = functools.partial(tcl.conv2d, output_dim=input_num/2)
        conv_1b = functools.partial(tcl.conv2d, num_outputs=output_num/2)
        conv_2 = functools.partial(tcl.conv2d, num_outputs=output_num)

    else:
        raise Exception('invalid resample value')

    if output_num==input_num and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(inputs=inputs, num_outputs=output_num, kernel_size=1) # Should kernel_size be larger?

    output = inputs
    output = conv_1(inputs=output, kernel_size=1)
    output = conv_1b(inputs=output, kernel_size=kernel_size)
    output = conv_2(inputs=output, kernel_size=1, biases_initializer=None) # Should skip bias here?
    # output = Batchnorm(name+'.BN', [0,2,3], output) # Should skip BN op here?

    return shortcut + (0.3*output)


# use depth-to-space for upsampling
def subpixelConv2D(*args, **kwargs):
    kwargs['num_outputs'] = 4*kwargs['num_outputs']
    output = tcl.conv2d(*args, **kwargs)
    output = tf.depth_to_space(output, 2)
    return output

