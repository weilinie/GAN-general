from __future__ import print_function

import os
import tarfile
from tqdm import tqdm
import requests

__author__ = 'Weili Nie'


def prepare_download_dir(dir = 'data'):
    if not os.path.exists(dir):
        os.mkdir(dir)


def download_1_billion_words(base_path):
    data_path = os.path.join(base_path, '1-billion-words')

    if os.path.exists(data_path):
        print('[!] Found 1-billion-worlds - skip')
        return

    save_path = os.path.join(base_path, '1-billion-word-language-modeling-benchmark-r13output.tar.gz')
    if os.path.exists(save_path):
        print('[*] {} already exists'.format(save_path))
    else:
        url = "http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz"
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(save_path, "wb") as f:
            for chunk in tqdm(response.iter_content(), total=total_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    # unzip the downloaded archive
    with tarfile.open(save_path, 'r:gz') as tarf:
        print('Extracting files from achives...')
        tarf.extractall(base_path)
        tarf.close()
    os.rename(os.path.join(base_path, "1-billion-word-language-modeling-benchmark-r13output"), data_path)
    # os.remove(save_path)


if __name__ == '__main__':
    base_path = 'data'
    prepare_download_dir()
    download_1_billion_words(base_path)