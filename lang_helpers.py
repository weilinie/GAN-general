import collections
import numpy as np
import os

def tokenize_string(sample):
    return tuple(sample.lower().split(' '))

def load_dataset(seq_len, max_train_data, tokenize=False, max_vocab_size=2048, data_path='data/1-billion-words'):
    dataset_name = os.path.basename(data_path)

    lines = []
    finished = False

    if dataset_name in ['1-billion-words']:
        train_data_path = os.path.join(data_path, 'training-monolingual.tokenized.shuffled')

        print("loading dataset: {}...".format(dataset_name))

        for i in range(99):

            filename = os.path.join(train_data_path, "news.en-{}-of-00100".format(str(i+1).zfill(5)))
            if i % 10 == 9:
                print('finished reading {} files...'.format(i+1))
            with open(filename, 'r') as f:
                for line in f:
                    line = line[:-1]
                    if tokenize:
                        line = tokenize_string(line)
                    else:
                        line = tuple(line)

                    if len(line) > seq_len:
                        line = line[:seq_len]

                    lines.append(line + ( ("`",)*(seq_len-len(line)) ) )

                    if len(lines) == max_train_data:
                        finished = True
                        break
            if finished:
                break

    else:
        raise Exception("[!] Caution! Paper didn't use other dataset")

    print('creating charmap...')
    np.random.shuffle(lines)

    counts = collections.Counter(char for line in lines for char in line)

    charmap = {'unk':0}
    inv_charmap = ['unk']

    for char, count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    print('length of charmap: {}'.format(len(charmap)))
    print('creating filtered lines...')
    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

    # for i in range(10):
    #     print(filtered_lines[i])

    print("loaded {} lines in dataset".format(len(lines)))
    return filtered_lines, charmap, inv_charmap