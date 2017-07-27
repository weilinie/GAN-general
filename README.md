# GAN-general

Tensorflow implementation for training GANs with various objectives and gradient penalties, different network architectures, both image and word generations

## Requirements

- Python >=2.7
- Tensorflow 1.1.0

## Usage

First download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or other datasets with:

    $ python download.py --dataset CelebA --data_dir data
    
To train a model for image generation:
    
    $ python GAN_GP_Img.py
    
To train a model for word generation:
    
    $ python GAN_GP_Char.py

You might need to customize the training process by changing the default arguments
