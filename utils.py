from __future__ import print_function

import functools
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from PIL import Image


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

# image residual block
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


# use depth-to-space for upsampling image
def subpixelConv2D(*args, **kwargs):
    kwargs['num_outputs'] = 4*kwargs['num_outputs']
    output = tcl.conv2d(*args, **kwargs)
    output = tf.depth_to_space(output, 2)
    return output


def f_congugate(t, option="KL", alpha=0):
    if option == "KL":
        return tf.exp(t-1)
    elif option == "RKL":
        return -1 - tf.log(-t)
    elif option == "JS":
        return -tf.log(2 - tf.exp(t))
    elif option == "Hellinger":
        return t / (1 - t)
    elif option == "TV":
        return t
    elif option == "Pearson":
        return t ** 2 / 4 + t
    elif option == "alpha" and alpha != 0:
        return 1 / alpha * ((1 - alpha) * t + 1) ** (alpha / (alpha - 1)) - 1 / alpha
    else:
        raise Exception("Not implemented divergence option")


def g_f(v, option="KL", alpha=0.5):
    if option == "KL":
        return v
    elif option == "RKL":
        return -tf.exp(v)
    elif option == "JS":
        return tf.log(2) - tf.log(1 + tf.exp(-v))
    elif option == "Hellinger":
        return 1 - tf.exp(-v)
    elif option == "TV":
        return tf.tanh(v) / 2
    elif option == "Pearson":
        return v
    elif option == "alpha" and alpha != 0:
        if alpha < 1:
            return 1 / (1 - alpha) - tf.log(1 + tf.exp(-v))
        else:
            return v
    else:
        raise Exception("Not implemented divergence option")



def make_grid(tensor, nrow=8, padding=2):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    batch_size = tensor.shape[0]
    xmaps = min(nrow, batch_size)
    ymaps = batch_size // xmaps
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= batch_size:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k += 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2):
    # Bathsize is 8*2
    ndarr = make_grid(tensor, nrow=nrow, padding=padding)
    im = Image.fromarray(ndarr)
    im.save(filename)