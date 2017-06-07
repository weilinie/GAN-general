import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_parser = add_argument_group('Network')
net_parser.add_argument('--conv_hidden_num', type=int, default=512, choices=[64, 128, 256, 512])
net_parser.add_argument('--batch_size', type=int, default=64)
net_parser.add_argument('--kernel_size', type=int, default=5)
net_parser.add_argument('--z_dim', type=int, default=128, choices=[64, 128])

# Data
data_parser = add_argument_group('Data')
data_parser.add_argument('--dataset', type=str, default='1-billion-words')
data_parser.add_argument('--seq_len', type=int, default=32, help='sequence length in characters')
data_parser.add_argument('--max_train_data', type=int, default=10000000)
data_parser.add_argument('--data_dir', type=str, default='data')

# Training
train_parser = add_argument_group('Training')
train_parser.add_argument('--optimizer', type=str, default='adam')
train_parser.add_argument('--max_step', type=int, default=100000)
train_parser.add_argument('--d_lr', type=float, default=1e-4)
train_parser.add_argument('--g_lr', type=float, default=1e-4)
train_parser.add_argument('--beta1', type=float, default=0.5)
train_parser.add_argument('--beta2', type=float, default=0.9)
train_parser.add_argument('--gpus', type=str, default='0')
train_parser.add_argument('--lmd', type=int, default=10, help='gradient penalty lambda hyperparameter')
train_parser.add_argument('--critic_iters', type=int, default=10)

# Summary and logs
summary_parser = add_argument_group('summary')
summary_parser.add_argument('--log_step', type=int, default=50)
summary_parser.add_argument('--save_step', type=int, default=5000)
summary_parser.add_argument('--log_dir', type=str, default='logs')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed