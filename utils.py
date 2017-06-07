from __future__ import print_function

import os
from datetime import datetime
import numpy as np


def prepare_dirs(config):

    config.model_name = "{}_{}".format(config.dataset, datetime.now().strftime("%m%d_%H%M%S"))

    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, config.model_name)

    if not hasattr(config, 'data_path'):
        config.data_path = os.path.join(config.data_dir, config.dataset)

    for dir in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)


# Dataset iterator
def inf_train_gen(lines, batch_size, charmap):
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines)-batch_size+1, batch_size):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i+batch_size]],
                dtype=np.int32
            )