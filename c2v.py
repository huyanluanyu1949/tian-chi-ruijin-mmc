#!/usr/bin/env python
'''
train character vector
'''

import sys
import logging
import os
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import util

def train(sens, modelfp):
    model = Word2Vec(size=100, window=15, sg=1, min_count=1, workers=4)
    model.build_vocab(sens)
    model.train(sens, total_examples=len(sens), epochs=10)
    model.wv.save(modelfp)

def load_sens(sen_dir, sen_len):
    sens = []
    file_txts = [x for x in os.listdir(sen_dir) if x.endswith('.txt')]
    for txt in file_txts:
        txtfp = os.path.join(sen_dir, txt)
        logging.info('read %s' % txtfp)
        with open(txtfp, 'r', encoding='utf8') as r:
            lines = r.readlines()
            ori_senss, _ = util.make_senss(lines, sen_len, ' ')
            sen =[[x.strip('\n') for x in ori_sens] for ori_sens in ori_senss]
            sens.extend(sen)
    return sens

def train_c2v():
    sen_dir = sys.argv[1]
    modelfp = sys.argv[2] if len(sys.argv) > 2 else 'c2v.kv'
    loglevel = sys.argv[-1] if len(sys.argv) > 3 else "INFO"
    logging.basicConfig(level=logging.getLevelName(loglevel))

    sens = load_sens(sen_dir, 80)
    logging.info(len(sens))
    logging.info('training ...')
    train(sens, modelfp)

def evaluate_c2v():
    char = sys.argv[1]
    modelfp = sys.argv[2] if len(sys.argv) > 2 else 'c2v.kv'
    wv = KeyedVectors.load(modelfp, mmap='r')
    print('similar', wv.most_similar([char]))
    #print(wv[char])

def main():
    train_c2v()
    #evaluate_c2v()

if __name__ == '__main__':
    main()
