from __future__ import absolute_import, division, print_function

import torch
import argparse
import nltk
import os.path as osp
import numpy as np
import json
import h5py

import options
from utils import utilities as utils
from torch.autograd import Variable


class Vocabulary(object):

    def __init__(self, word2ind):
        assert self.unk in word2ind, '{} not in vocabulary'.format(self.unk)
        assert self.start not in word2ind
        assert self.end not in word2ind

        word_count = len(word2ind)
        word2ind[self.start] = word_count + 1
        word2ind[self.end] = word_count + 2

        self.word2ind = word2ind
        self.ind2word = {v: k for k, v in word2ind.items()}
        self.vocab_size = word_count + 3

    @property
    def size(self):
        return self.vocab_size

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


def load_downloaded_json(json_file, split='train', index=0):
    json_file = osp.expanduser(json_file)

    with open(json_file, 'r') as f:
        data = json.load(f)

    key = 'unique_img_{}'.format(split)
    img_name = data[key][index]
    print('image name: {}'.format(img_name))

    word2ind = data['word2ind']

    return img_name, word2ind


def load_download_img_feat(img_feat_file, split='train', index=0):
    img_feat_file = osp.expanduser(img_feat_file)

    data = h5py.File(img_feat_file, 'r')
    key = 'images_{}'.format(split)
    img_feat = np.array(data[key])[index]  # Nx4096

    return img_feat


def decode_sent(inds, length, vocab):
    words = [vocab.decode(i) for i in inds[:length]]
    return ' '.join(words)


def load_processed_data(processed_file, vocab, split='train', index=0, gt=False):
    processed_file = osp.expanduser(processed_file)
    data = h5py.File(processed_file, 'r')

    cap = np.array(data['cap_{}'.format(split)][index])
    cap_len = np.array(data['cap_length_{}'.format(split)][index]).item()

    if gt:
        ques = np.array(data['ques_{}'.format(split)][index])
        ques_len = np.array(data['ques_length_{}'.format(split)][index])

        ans = np.array(data['ans_{}'.format(split)][index])
        ans_len = np.array(data['ans_length_{}'.format(split)][index])
        assert ques.shape[0] == ans.shape[0]

        print('ground truth: ========================')
        print('caption: {}'.format(decode_sent(cap, cap_len, vocab)))

        num_round = ques.shape[0]
        for r in range(num_round):
            q, a = ques[r], ans[r]
            q_len, a_len = ques_len[r], ans_len[r]
            print('Q: {}'.format(decode_sent(q, q_len, vocab)))
            print('A: {}'.format(decode_sent(a, a_len, vocab)))
        print('===========================')

    return cap, cap_len


def tokenize_and_encode(sentence, vocab):
    tokens = nltk.tokenize.word_tokenize(sentence.lower().strip())
    return [vocab.encode(w) for w in tokens]


def process(sent, sent_len, vocab):
    sent = torch.from_numpy(np.asarray(sent, dtype=np.int64))

    res = torch.LongTensor(1, sent_len + 2).fill_(0)
    res[0, 0] = vocab.encode(vocab.start)
    res[0, 1: sent_len + 1] = sent[:sent_len]
    res[0, sent_len + 1] = vocab.encode(vocab.end)

    res_len = torch.LongTensor(1)
    res_len[0] = sent_len + 1
    return res, res_len


def main():
    params = options.readCommandLine()

    split = params['evalSplit']
    index = params['index']

    img_name, word2ind = load_downloaded_json(params['inputJson'], split=split, index=index)
    vocab = Vocabulary(word2ind)

    params['continue'] = True
    params['vocabSize'] = vocab.size
    aBot, loadedParams, _ = utils.loadModel(params, 'abot', overwrite=False)
    assert aBot.encoder.vocabSize == vocab.size
    aBot.eval()
    aBot.reset()

    img_feat = load_download_img_feat(params['inputImg'], split=split, index=index)
    img_feat = torch.from_numpy(img_feat)

    cap, cap_len = load_processed_data(
        params['inputQues'], vocab, split=split, index=index, gt=params['gt']
    )

    cap, cap_len = process(cap, cap_len, vocab)
    img_feat = Variable(img_feat, volatile=True)
    cap = Variable(cap, volatile=True)
    cap_len = Variable(cap_len, volatile=True)
    aBot.observe(-1, image=img_feat[None, :], caption=cap, captionLens=cap_len)

    num_rounds = params['numRounds']

    for r in range(num_rounds):
        q = input("Round {} question (will add ? automatically) >>> ".format(r + 1))
        q = tokenize_and_encode(q + '?', vocab)
        q_len = len(q)
        q, q_len = process(q, q_len, vocab)
        q = Variable(q, volatile=True)
        q_len = Variable(q_len, volatile=True)

        aBot.observe(r, ques=q, quesLens=q_len)
        answers, ansLens = aBot.forwardDecode(beamSize=5, inference='greedy')
        aBot.observe(r, ans=answers, ansLens=ansLens)

        print('A: {}'.format(
            decode_sent(answers.data[0].numpy(), ansLens.data[0], vocab)[8:]
        ))


if __name__ == '__main__':
    main()
