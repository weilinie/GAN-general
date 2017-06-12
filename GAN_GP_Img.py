__author__ = 'Weili Nie'

import os
import tensorflow as tf
from tqdm import trange

from img_helpers import load_dataset
from config import get_config
from model_img import *
from utils import *


class GAN_GP_Img(object):
    def __init__(self, config):
        self.d_net = config.d_net
        self.g_net = config.g_net
        self.dataset = config.dataset_img
        self.data_path = config.data_path
        self.split = config.split

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.loss_type = config.loss_type

        self.z_dim = config.z_dim
        self.conv_hidden_num = config.conv_hidden_num
        self.img_dim = config.img_dim
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.normalize_d = config.normalize_d
        self.normalize_g = config.normalize_g

        self.model_dir = config.model_dir
        self.log_step = config.log_step
        self.save_step = config.save_step
        self.max_step = config.max_step

        self.lmd = config.lmd

        if self.loss_type in ['WGAN', 'WGAN-GP']:
            self.critic_iters = config.critic_iters
        else:
            self.critic_iters = 1

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.build_model()

        self.sv = tf.train.Supervisor(
            logdir=self.model_dir,
            summary_op=None,
            global_step=self.global_step,
            save_model_secs=300)

        # gpu_options = tf.GPUOptions(allow_growth=True)
        # self.sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    def build_model(self):
        self.x = load_dataset(
            data_path=self.data_path,
            batch_size=self.batch_size,
            scale_size=self.img_dim,
            split=self.split
        )
        x = self.x / 127.5 - 1. # Normalization
        img_chs = self.x.get_shape().as_list()[-1]
        self.z = tf.random_normal(shape=[self.batch_size, self.z_dim])

        fake_data, g_vars = generator(
            self.g_net, self.z, self.conv_hidden_num,
            self.img_dim, img_chs, self.normalize_g
        )

        self.fake_data = tf.clip_by_value((fake_data + 1)*127.5, 0, 255) # Denormalization

        d_out_real, d_vars = discriminator(
            self.d_net, x, self.conv_hidden_num,
            self.normalize_d, reuse=False
        )
        d_out_fake, _ = discriminator(
            self.d_net, fake_data, self.conv_hidden_num,
            self.normalize_d
        )

        self.d_loss, self.g_loss = self.cal_losses(x, fake_data, d_out_real, d_out_fake, self.loss_type)
        self.slope_real = tf.reduce_mean(tf.norm(tf.gradients(d_out_real, [x])[0], axis=1))

        if self.optimizer == 'adam':
            optim_op = tf.train.AdamOptimizer
        elif self.optimizer == 'rmsprop':
            optim_op = tf.train.RMSPropOptimizer
        else:
            raise Exception("[!] Caution! Other optimizers do not apply right now!")

        self.d_optim = optim_op(self.d_lr, self.beta1, self.beta2).minimize(
            self.d_loss, var_list=d_vars
        )
        self.g_optim = optim_op(self.g_lr, self.beta1, self.beta2).minimize(
            self.g_loss, global_step=self.global_step, var_list=g_vars
        )

        self.summary_op = tf.summary.merge([
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("grad/norm_gradient", self.slope_real)
        ])

    def train(self):
        print('start training...\n [{}] using d_net [{}] and g_net [{}] with loss type [{}]\n'
              'normalize_d: {}, normalize_g: {}'.format(
            self.dataset, self.d_net, self.g_net, self.loss_type, self.normalize_d, self.normalize_g
        ))
        z_fixed = np.random.normal(size=[self.batch_size, self.z_dim])

        with self.sv.managed_session() as sess:
            for _ in range(self.max_step):
                if self.sv.should_stop():
                    break

                step = sess.run(self.global_step)

                # Train generator
                sess.run(self.g_optim)

                # Train critic
                for _ in range(self.critic_iters):
                    sess.run(self.d_optim)

                if step % self.log_step == 0:
                    g_loss, d_loss, slope, summary_str = sess.run(
                        [self.g_loss, self.d_loss, self.slope_real, self.summary_op]
                    )
                    self.sv.summary_computed(sess, summary_str)
                    print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} Slope: {:.6f}".
                          format(step, self.max_step, d_loss, g_loss, slope))

                if step % self.save_step == 0:
                    self.generate_samples(sess, z_fixed, self.model_dir, idx=step)

    def generate_samples(self, sess, z_fixed, model_dir, idx):
        x = sess.run(self.fake_data, {self.z: z_fixed})
        sample_dir = os.path.join(model_dir, 'samples')

        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        path = os.path.join(sample_dir, 'sample_G_{:06d}.png'.format(idx))
        save_image(x, path)
        print("[*] Samples saved: {}".format(path))
        # save_image(sess.run(self.x), os.path.join(sample_dir, 'sample_x_{}.png'.format(idx)))

    def cal_grad_penalty(self, real_data, fake_data):
        # WGAN lipschitz-penalty
        epsilon = tf.random_uniform(shape=[self.batch_size, 1, 1], minval=0., maxval=1.)

        data_diff = fake_data - real_data
        interp_data = real_data + epsilon * data_diff
        disc_interp, _ = discriminator(
            self.d_net, interp_data, self.conv_hidden_num,
            self.normalize_d
        )
        grad_interp = tf.gradients(disc_interp, [interp_data])[0]
        print('The shape of grad_interp: {}'.format(grad_interp.get_shape().as_list()))

        slope = tf.norm(grad_interp, axis=1)
        grad_penalty = tf.reduce_mean((slope - 1.) ** 2)
        return grad_penalty

    def cal_losses(self, real_data, fake_data, d_out_real, d_out_fake, loss_type):
        f_div = ['KL', 'RKL', 'JS', 'Hellinger', 'TV', 'Pearson', 'alpha']
        f_div_gp = ['KL-GP', 'RKL-GP', 'JS-GP', 'Hellinger-GP', 'TV-GP', 'Pearson-GP', 'alpha-GP']

        if loss_type == 'WGAN-GP':
            d_loss = tf.reduce_mean(d_out_fake) - tf.reduce_mean(d_out_real)
            g_loss = -tf.reduce_mean(d_out_fake)

            grad_penalty = self.cal_grad_penalty(real_data, fake_data)
            d_loss += self.lmd * grad_penalty

            return d_loss, g_loss

        elif loss_type in ['GAN', 'GAN-GP']:
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_out_real, labels=tf.ones_like(d_out_real)
            ))
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
            ))
            d_loss = d_loss_real + d_loss_fake

            # use -logD trick
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_out_fake, labels=tf.ones_like(d_out_fake)
            ))

            if loss_type == 'GAN-GP':
                grad_penalty = self.cal_grad_penalty(real_data, fake_data)
                d_loss += self.lmd * grad_penalty

            return d_loss, g_loss

        elif loss_type in f_div + f_div_gp:
            loss_name = loss_type.split('-')[0]
            d_loss_real = -tf.reduce_mean(g_f(d_out_real, loss_name))
            d_loss_fake = tf.reduce_mean(f_congugate(g_f(d_out_fake, loss_name)))
            d_loss = d_loss_real + d_loss_fake
            g_loss = -d_loss_fake

            if loss_type in f_div_gp:
                grad_penalty = self.cal_grad_penalty(real_data, fake_data)
                d_loss += self.lmd * grad_penalty

            return d_loss, g_loss

        return None


if __name__ == '__main__':
    config, unparsed = get_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    prepare_dirs(config, config.dataset_img)

    gan_gp_img = GAN_GP_Img(config)
    gan_gp_img.train()
