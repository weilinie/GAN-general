import os
import tensorflow as tf
from tqdm import trange

from lang_helpers import load_dataset
from config import get_config
from model_lang import *
from utils import *

__author__= 'Weili Nie'


class WGAN_GP_Char(object):
    def __init__(self, config):
        self.d_net = config.d_net
        self.g_net = config.g_net
        self.dataset = config.dataset_lang
        self.data_path = config.data_path
        self.max_train_data = config.max_train_data

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.kernel_size = config.kernel_size
        self.z_dim = config.z_dim
        self.conv_hidden_num = config.conv_hidden_num
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.seq_len = config.seq_len

        self.model_dir = config.model_dir
        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step

        self.lmd = config.lmd
        self.critic_iters = config.critic_iters

        self.build_model()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=300,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

    def build_model(self):
        self.lines, self.charmap, self.inv_charmap = load_dataset(
            seq_len=self.seq_len,
            max_train_data=self.max_train_data,
            data_path=self.data_path
        )
        vocab_size = len(self.charmap)
        
        self.z = tf.random_normal(shape=[self.batch_size, self.z_dim])
        
        self.real_data = tf.placeholder(tf.int32, shape=[self.batch_size, self.seq_len])
        real_data_one_hot = tf.one_hot(self.real_data, vocab_size)
        fake_data_softmax, g_vars = generator(self.g_net, self.z, self.conv_hidden_num, self.seq_len,
                                              self.kernel_size, vocab_size)
        self.fake_data = tf.argmax(fake_data_softmax, fake_data_softmax.get_shape().ndims-1)

        d_out_real, d_vars = discriminator(self.d_net, real_data_one_hot, self.conv_hidden_num,
                                           self.seq_len, self.kernel_size, vocab_size, reuse=False)
        d_out_fake, _ = discriminator(self.d_net, fake_data_softmax, self.conv_hidden_num,
                                      self.seq_len, self.kernel_size, vocab_size)

        self.d_loss = tf.reduce_mean(d_out_fake) - tf.reduce_mean(d_out_real)
        self.g_loss = -tf.reduce_mean(d_out_fake)

        # WGAN lipschitz-penalty
        epsilon = tf.random_uniform(shape=[self.batch_size, 1, 1], minval=0., maxval=1.)

        data_diff = fake_data_softmax - real_data_one_hot
        interp_data = real_data_one_hot + epsilon * data_diff
        disc_interp, _ = discriminator(self.d_net, interp_data, self.conv_hidden_num,
                                       self.seq_len, self.kernel_size, vocab_size)
        grad_interp = tf.gradients(disc_interp, [interp_data])[0]
        print('The shape of grad_interp: {}'.format(grad_interp.get_shape().as_list()))

        self.slope = tf.norm(grad_interp)
        gradient_penalty = tf.reduce_mean((self.slope - 1.) ** 2)
        self.d_loss += self.lmd * gradient_penalty

        if self.optimizer == 'adam':
            optim = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Paper didn't use {} optim other than Adam".format(self.optimizer))

        self.d_optim = optim(self.d_lr, self.beta1, self.beta2).minimize(self.d_loss, var_list=d_vars)
        self.g_optim = optim(self.g_lr, self.beta1, self.beta2).minimize(self.g_loss, var_list=g_vars)

        self.summary_op = tf.summary.merge([
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("grad/norm_gradient", self.slope)
        ])

    def train(self):
        z_fixed = np.random.normal(size=[self.batch_size*10, self.z_dim]) # samples of 10 times batch size
        gen = inf_train_gen(self.lines, self.batch_size, self.charmap)

        for step in trange(self.max_step):
            # Train generator
            _data = gen.next()
            summary_str, _ = self.sess.run([self.summary_op, self.g_optim], feed_dict={self.real_data: _data})
            self.summary_writer.add_summary(summary_str, global_step=step)
            self.summary_writer.flush()

            # Train critic
            for i in range(self.critic_iters):
                _data = gen.next()
                self.sess.run(self.d_optim, feed_dict={self.real_data: _data})

            if step % 100 == 99:
                _data = gen.next()
                g_loss, d_loss, slope = self.sess.run([self.g_loss, self.d_loss, self.slope],
                                                      feed_dict={self.real_data: _data})
                print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} Slope: {:.6f}".
                      format(step+1, self.max_step, d_loss, g_loss, slope))
                self.generate_samples(z_fixed, idx=step+1)

    def generate_samples(self, z_fixed, idx):
        gen_samples = []
        for num in range(10):
            z_fixed_bs = z_fixed[self.batch_size*num: self.batch_size*(num+1)]
            samples = self.sess.run(self.fake_data, feed_dict={self.z: z_fixed_bs})
            decoded_samples = []
            for i in range(len(samples)):
                decoded = []
                for j in range(len(samples[i])):
                    decoded.append(self.inv_charmap[samples[i][j]])
                decoded_samples.append(tuple(decoded))
            gen_samples.extend(decoded_samples)

        with open(os.path.join(self.model_dir, 'samples_{}.txt'.format(idx)), 'w') as f:
            for s in gen_samples:
                s = "".join(s)
                f.write(s + "\n")



if __name__ == '__main__':
    config, unparsed = get_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    prepare_dirs(config, config.dataset_lang)

    wgan_gp_char = WGAN_GP_Char(config)
    wgan_gp_char.train()
