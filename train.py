#!/usr/bin/env python

import sys
import os
import logging
import pickle
import numpy as np
from collections import Counter
import random
from gensim.models import KeyedVectors
from keras.callbacks import ModelCheckpoint
import util

# available models
import bilstm_crf
import att_bilstm
import att_cnn
import att_bilstm_crf

CHUNK_TAGS = [
        'O',
        'B-Disease', 'I-Disease',
        'B-Reason', 'I-Reason',
        "B-Symptom", "I-Symptom",
        "B-Test", "I-Test",
        "B-Test_Value", "I-Test_Value",
        "B-Drug", "I-Drug",
        "B-Frequency", "I-Frequency",
        "B-Amount", "I-Amount",
        "B-Treatment", "I-Treatment",
        "B-Operation", "I-Operation",
        "B-Method", "I-Method",
        "B-SideEff","I-SideEff",
        "B-Anatomy", "I-Anatomy",
        "B-Level", "I-Level",
        "B-Duration", "I-Duration"]

ONE_HOT_CHUNK_TAGS = np.eye(len(CHUNK_TAGS))

def build_vocab(direc='./train', savedir='./data'):
    '''build vocab'''
    logging.info('building vocab ...')
    file_tags = [x for x in os.listdir(direc) if x.endswith('.txt.tag')]
    logging.debug(file_tags)
    liness = []
    for file_tag in file_tags:
        tagfp = os.path.join(direc, file_tag)
        with open(tagfp, 'r', encoding='utf8') as r:
            lines = [x.split('\t')[0] for x in r.readlines()]
            liness.extend(lines)

    word_counts = Counter([line for line in liness])
    # TODO: adjust f
    vocab = [w for w, f in iter(word_counts.items()) if f >= 1]
    word2idx = dict((w, i+2) for i, w in enumerate(vocab))
    word2idx['PAD'] = 0
    word2idx['UNK'] = 1

    idx2word = {}
    for word in word2idx:
        idx2word[word2idx[word]] = word
    print(len(idx2word))

    word2idx_pkl = os.path.join(savedir,'word2idx.pkl')
    with open(word2idx_pkl, 'wb') as w:
        pickle.dump(word2idx, w)
    logging.info('save word2idx pkl to: %s' % word2idx_pkl)

    idx2word_pkl = os.path.join(savedir,'idx2word.pkl')
    with open(idx2word_pkl, 'wb') as w:
        pickle.dump(idx2word, w)
    logging.info('save idx2word pkl to: %s' % idx2word_pkl)

    word2idx_fp = os.path.join(savedir,'word2idx.txt')
    with open(word2idx_fp, 'w') as w:
        for word in word2idx:
            w.write(word + '\t' + str(word2idx[word]) + '\n')
    logging.info('save word2idx txt to: %s' % word2idx_fp)

    return word2idx

def load_vocab(savedir):
    word2idx, idx2word = None, None
    with open(os.path.join(savedir, 'word2idx.pkl'), 'rb') as r:
        word2idx = pickle.load(r)

    with open(os.path.join(savedir, 'idx2word.pkl'), 'rb') as r:
        idx2word = pickle.load(r)

    return word2idx, idx2word


def load_tagfp_xy(tagfp, word2idx, sen_len):
    with open(tagfp, 'r', encoding='utf8') as r:
        lines = r.readlines()
        ori_senss, win_senss = util.make_senss(lines, sen_len, 'PAD\tO\n')
        senss = ori_senss + win_senss
        x = [ [word2idx.get(sen.split('\t')[0]) for sen in sens] for sens in senss]
        y = [ [CHUNK_TAGS.index(sen.split('\t')[1].strip()) for sen in sens] for sens in senss]
        return x, y

def load_train_data(direc, word2idx, sen_len, val_ratio=0.8):
    ''' load train data.
    sen_len: length of train sequence

    '''
    logging.info('loading train data ... ')
    file_tags = util.file_paths_with_postfix(direc, '.txt.tag')
    logging.info('load tags file: %s' % len(file_tags))
    random.shuffle(file_tags)    # shuffle
    # train data set
    num_train = int(len(file_tags) * val_ratio)
    train_x, train_y = [], []

    for file_tag in file_tags[0:num_train]:
        x, y = load_tagfp_xy(file_tag, word2idx, sen_len)
        train_x.extend(x)
        train_y.extend(y)

    # dev data set
    val_x, val_y = [], []

    for file_tag in file_tags[num_train: len(file_tags)]:
        x, y = load_tagfp_xy(file_tag, word2idx, sen_len)
        val_x.extend(x)
        val_y.extend(y)

    return np.array(train_x), \
           np.eye(len(CHUNK_TAGS), dtype='float32')[np.array(train_y)],\
           np.array(val_x),\
           np.eye(len(CHUNK_TAGS), dtype='float32')[np.array(val_y)]

def predict(model, emb_input_dim, args, test_x):
    raw = model.predict(np.array(test_x))
    result_tags = []
    for rawsen in raw:
        sen_tag = [CHUNK_TAGS[np.argmax(x)] for x in rawsen ]
        result_tags.append(sen_tag)
    return result_tags

def load_embedding_matrix(kvfp, emb_dim, word2idx):
    if not os.path.exists(kvfp):
        return None

    embedding_matrix = np.zeros((len(word2idx), emb_dim))
    logging.info('load c2v from: %s' % kvfp)
    wv = KeyedVectors.load(kvfp, mmap='r')
    for word, idx in word2idx.items():
        try:
            wordv = wv[word]
        except KeyError:
            wordv = wv[' ']
        embedding_matrix[idx] = wordv
    return embedding_matrix

class Arg():
    def __init__(self):
        self.vocab_dir = './data'
        self.model_dir = './model'
        # TODO: set the model to use
	self.model = att_cnn
        # TODO: save model fp
        self.model_fp = './model/att_bilstm.h5'

        self.sen_len = 80
        self.hiden_unit = 100
        self.emb_dim = 100

        # samples(train) / samples(dev)
        self.train_dev_ratio = 0.8

        self.batch = 10
        self.epoch = 20

def train():
    args = Arg()

    mode = sys.argv[1]

    if mode == 'build_vocab':
        loglevel = sys.argv[-1] if len(sys.argv) > 2 else "INFO"
        logging.basicConfig(level=logging.getLevelName(loglevel))

        build_vocab()

    elif mode == 'train':
        train_dir = sys.argv[2]
        loglevel = sys.argv[-1] if len(sys.argv) > 3 else "INFO"
        logging.basicConfig(level=logging.getLevelName(loglevel))

        word2idx, idx2word = load_vocab(args.vocab_dir)
        embedding_matrix = load_embedding_matrix('c2v.kv', args.emb_dim, word2idx)

        train_x, train_y, val_x, val_y = load_train_data(train_dir, word2idx, args.sen_len, args.train_dev_ratio)

        logging.info('load tx: %s,ty: %s,vx: %s,vy: %s' % (train_x.shape, train_y.shape, val_x.shape, val_y.shape))

        model = args.model.build_model(args.sen_len, len(word2idx), args.emb_dim, args.hiden_unit, len(CHUNK_TAGS), embedding_matrix)

        checkpoint = ModelCheckpoint(args.model_fp, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        model.fit(train_x, train_y, batch_size=args.batch, epochs=args.epoch, validation_data=[val_x, val_y],callbacks=[checkpoint])

    elif mode == 'validate':
        model_fp = sys.argv[2]
        test_dir_or_fp = sys.argv[3]
        loglevel = sys.argv[-1] if len(sys.argv) > 4 else "INFO"
        logging.basicConfig(level=logging.getLevelName(loglevel))

        word2idx, idx2word = load_vocab(args.vocab_dir)

        model = args.model.build_model(args.sen_len, len(word2idx), args.emb_dim, args.hiden_unit, len(CHUNK_TAGS))

        model.load_weights(model_fp)

        def _validate(test_fp):
            with open(test_fp, 'r', encoding='utf8') as r:
                lines = r.readlines()
                ori_senss, win_senss = util.make_senss(lines, args.sen_len, 'PAD\tO\n')

            # ori sens
            test_x = [ [word2idx.get(sen.split('\t')[0]) for sen in sens] for sens in ori_senss]
            logging.info('validate fp: %s, test_x shape: %s' % (test_fp, np.array(test_x).shape))
            result_tags= predict(model, len(word2idx), args, test_x)

            # win sens
            test_win_x = [ [word2idx.get(sen.split('\t')[0]) for sen in sens] for sens in win_senss]
            logging.info('validate fp: %s, test_win_x shape: %s' % (test_fp, np.array(test_win_x).shape))
            result_win_tags= predict(model, len(word2idx), args, test_win_x)

            # ori word and tag
            test_x_word = [ [sen.split('\t')[0] for sen in sens] for sens in ori_senss]
            test_x_tag = [ [sen.split('\t')[1].strip() for sen in sens] for sens in ori_senss]

            test_x_word = util.flatten(test_x_word)
            test_x_tag = util.flatten(test_x_tag)
            result_tags = util.flatten(result_tags)
            result_win_tags = util.flatten(result_win_tags)
            result_win_tags = ['O'] * (args.sen_len // 2) + result_win_tags + ['O'] * (args.sen_len // 2)    # padding back

            print(len(test_x_word), len(test_x_tag), len(result_tags), len(result_win_tags))

            merge_tags = util.merge_win_tags(result_tags, result_win_tags)

            val_fp = test_fp.replace('.tag','.val')
            val_fo = open(val_fp, 'w', encoding='utf8')
            for word, tag, res_tag, res_win_tag, merge_tag in zip(test_x_word, test_x_tag, result_tags, result_win_tags, merge_tags):
                val_fo.write(word + '\t' + tag + '\t' + merge_tag + '\t' + res_win_tag + '\t' + res_tag + '\n')

            val_fo.close()
            logging.info('save validate to: %s' % val_fp)

        if os.path.isdir(test_dir_or_fp):
            test_dir = test_dir_or_fp
            file_tags = [os.path.join(test_dir, x) for x in os.listdir(test_dir) if x.endswith('.txt.tag')]
            for file_tag in file_tags:
                _validate(file_tag)

        elif os.path.isfile(test_dir_or_fp):
            _validate(test_dir_or_fp)

    elif mode == 'test':
        model_fp = sys.argv[2]
        test_dir = sys.argv[3]
        submit_dir = sys.argv[4]
        loglevel = sys.argv[-1] if len(sys.argv) > 5 else "INFO"
        logging.basicConfig(level=logging.getLevelName(loglevel))

        word2idx, idx2word = load_vocab(args.vocab_dir)
        model = args.model.build_model(args.sen_len, len(word2idx), args.emb_dim, args.hiden_unit, len(CHUNK_TAGS))

        model.load_weights(model_fp)

        file_txts = [x for x in os.listdir(test_dir) if x.endswith('.txt')]
        for txt in file_txts:
            txtfp = os.path.join(test_dir, txt)
            logging.info('read %s' % txtfp)
            with open(txtfp, 'r', encoding='utf8') as r:
                lines = r.readlines()
                ori_senss, win_senss = util.make_senss(lines, args.sen_len, 'PAD')

            # ori sens
            test_x = [ [word2idx.get(sen.strip(), word2idx.get(' ')) for sen in sens] for sens in ori_senss]
            result_tags= predict(model, len(word2idx), args, test_x)

            # win sens
            test_win_x = [ [word2idx.get(sen.strip(), word2idx.get(' ')) for sen in sens] for sens in win_senss]
            result_win_tags= predict(model, len(word2idx), args, test_win_x)

            test_x_word = [ [sen.strip() for sen in sens] for sens in ori_senss]

            words = util.flatten(test_x_word)
            result_tags = util.flatten(result_tags)
            result_win_tags = util.flatten(result_win_tags)
            result_win_tags = ['O'] * (args.sen_len // 2) + result_win_tags + ['O'] * (args.sen_len // 2)    # padding back

            merge_tags = util.merge_win_tags(result_tags, result_win_tags)
            print(len(merge_tags))

            position_colls = util.collect_result(merge_tags)
            print(len(merge_tags))

            ann_fp = os.path.join(submit_dir, txt.replace('.txt','.ann'))
            util.make_submit(ann_fp, position_colls, merge_tags, words)

def main():
    train()

if __name__ == '__main__':
    main()
