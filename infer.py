from __future__ import absolute_import, division, print_function

import argparse
import nltk
import os.path as osp
import numpy as np
import json
import h5py


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image-dir', dest='image_dir', type=str, required=True)
    parser.add_argument('--visdial-dir', dest='visdial_dir', type=str, default=None)
    parser.add_argument('--index', dest='index', type=int, default=0,
                        help='Which file used to generate dialog')
    parser.add_argument('--split', dest='split', type=str, default='train',
                        choices=['train', 'val', 'test'])

    parser.add_argument('--num-round', dest='num_round', type=int, default=10)
    parser.add_argument('--use-gpu', dest='use_gpu', action='store_true')
    parser.add_argument('--ques-len', dest='ques_len', type=int, default=20)
    parser.add_argument('--ans-len', dest='ans_len', type=int, default=20)
    parser.add_argument('--cap-len', dest='cap_len', type=int, default=40)

    args = parser.parse_args()
    return args


def process_downloaded_json(json_file='./data/visdial/chat_processed_params.json', split='train'):
    json_file = osp.expanduser(json_file)

    with open(json_file, 'r') as f:
        data = json.load(f)

    key = 'unique_img_{}'.format(split)
    img_names = data[key]

    word2ind = data['word2ind']
    ind2word = {int(k): v for k, v in data['ind2word'].items()}

    return img_names, word2ind, ind2word


def load_download_img_feats(img_feat_file='./data/visdial/data_img.h5', split='train'):
    img_feat_file = osp.expanduser(img_feat_file)

    data = h5py.File(img_feat_file, 'r')
    key = 'images_{}'.format(split)
    img_feats = np.array(data[key])  # Nx4096

    return img_feats


class Vocabulary(object):

    def __init__(self, word2ind, ind2word):
        for k, v in word2ind.items():
            assert v in ind2word, v
            assert k == ind2word[v]

        assert self.unk in word2ind, '{} not in vocabulary'.format(self.unk)
        assert self.start not in word2ind
        assert self.end not in word2ind

        assert self.pad_ind not in ind2word

        word_count = len(word2ind)
        word2ind[self.start] = word_count + 1
        word2ind[self.end] = word_count + 2

        self.word2ind = word2ind
        self.ind2word = ind2word

    @property
    def pad_ind(self):
        return 0

    @property
    def unk(self):
        return 'UNK'

    @property
    def start(self):
        return '<START>'

    @property
    def end(self):
        return '<END>'

    def encode(self, word):
        return self.word2ind.get(word, self.word2ind['UNK'])

    def decode(self, ind):
        return self.ind2word.get(ind, 'UNK')


def tokenize(sentence):
    return nltk.tokenize.word_tokenize(sentence.lower().strip())


# data in chat_processed_data.h5, where split can be train, val, or test:
# 'cap_<split>': Nx40
# 'ans_<split>': Nx10x20
# 'ques_<split>': Nx10x20


def main():
    args = parse_args()
    print(args)

    split = args.split
    img_names, word2ind, ind2word = process_downloaded_json(split=split)
    img_feats = load_download_img_feats(split=split)

    vocab = Vocabulary(word2ind, ind2word)
    print(tokenize('this is an example'))

    num_round = args.num_round
    in_seq = "This is an example"
    for i in range(num_round):
        print('Round {}'.format(i + 1))

        in_words = tokenize(in_seq)
        in_inds = [vocab.encode(w) for w in in_words]


if __name__ == '__main__':
    main()
