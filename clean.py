#!/usr/bin/env python

'''
Clean data
'''
import sys
import os
import logging
import json
import util

DISEASE_TAG = {
        'Disease': 'A',
        'Reason':'R',
        'Symptom':'S',
        'Test': 'T',
        'Test_Value': 'V',
        'Drug':'D',
        'Frequency':'F',
        'Amount': 'N',
        'Method': 'M',
        'Treatment':'P',
        'Operation': 'O',
        'Anatomy':'Y',
        'Level':  'L',
        'Duration':'U',
        'SideEff': 'E'
        }

def clean_txt(direc, save_dir):
    file_txts = [x for x in os.listdir(direc) if x.endswith('txt')]
    logging.debug(file_txts)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # clean text file
    for txt in file_txts:
        txtfp = os.path.join(direc, txt)
        logging.info('read %s' % txtfp)
        clean_txts = []
        with open(txtfp, encoding='utf8') as r:
            lines = r.readlines()
            for line in lines:
                for c in line:
                    if c == '\n':
                        clean_txts.append('LSEG')
                    else:
                        clean_txts.append(c)

        logging.debug(clean_txts)

        txt_savefp = os.path.join(save_dir, txt)
        with open(txt_savefp, 'w', encoding='utf8') as w:
            for clean_txt in clean_txts:
                w.write(clean_txt+'\n')
            logging.info('save clean txt file to: %s' % txt_savefp)

def merge_anns(direc):
    '''
    merge all anns in train dataset
    '''
    file_anns = [x for x in os.listdir(direc) if x.endswith('ann')]
    logging.debug(file_anns)
    enities = set()
    enities_ori = set()
    for ann in file_anns:
        annfp = os.path.join(direc, ann)
        logging.info('read %s' % annfp)
        with open(annfp, encoding='utf8') as r:
            lines = r.readlines()
            for line in lines:
                enity = line.strip().split('\t')[-1]
                enities_ori.add(enity)
                enities.add(enity.replace(' ', ''))

    sorted_entinies = sorted(list(enities))
    logging.info('len of enities: %s' % len(enities))
    logging.debug('enities: %s' % sorted_entinies)

    with open('merge_anns.txt', 'w', encoding='utf8') as w:
        w.write('\n'.join(sorted_entinies))
        logging.info('save sorted_entinies to: %s' % sorted_entinies)

    sorted_entinies_ori = sorted(list(enities_ori))
    with open('merge_anns_ori.txt', 'w', encoding='utf8') as w:
        w.write('\n'.join(sorted_entinies_ori))
        logging.info('save sorted_entinies to: %s' % sorted_entinies_ori)

class ANN(object):
    def __init__(self, seq, typ, st, ed, name):
        self.seq = seq
        self.typ = typ
        self.st = st
        self.ed = ed
        self.name = name

    def todict(self):
        return {'seq': self.seq, 'typ': self.typ, 'st': self.st, 'ed': self.ed, 'name': self.name}

    def serial(self):
        return json.dumps(self.todict())

    def __str__(self):
        return '%s, %s, %s, %s, %s' % (self.seq, self.typ, self.st, self.ed, self.name)

def parse_ann(annfp):
    ret = []
    with open(annfp, encoding='utf8') as r:
        lines = r.readlines()
        for line in lines:
            seq, typ_and_pos, name = line.strip().split('\t')
            typ_and_pos_ = typ_and_pos.split(' ')
            typ = typ_and_pos_[0]
            pos_st = int(typ_and_pos_[1])
            pos_ed = int(typ_and_pos_[-1])
            ret.append(ANN(seq, typ, pos_st, pos_ed, name))
    return ret

def parse_anns(direc, save_dir):
    file_anns = [x for x in os.listdir(direc) if x.endswith('ann')]
    logging.debug(file_anns)
    for ann in file_anns:
        annfp = os.path.join(direc, ann)
        logging.info('read %s' % annfp)
        ann_objs = parse_ann(annfp)
        ann_savefp = os.path.join(save_dir, ann) + '.jsons'
        with open(ann_savefp, 'w', encoding='utf8') as w:
            for ann_obj in ann_objs:
                logging.debug(ann_obj.serial())
                w.write(ann_obj.serial() + '\n')
        logging.info('save ann .jsons to: %s' % ann_savefp)

def get_ann_tag(typ, st, ed):
    ''' B-DISEASE, I-DISEASE, ... etc'''
    return ['B-%s' % typ] + ['I-%s' % typ] * (ed - st - 1)

def tag_lines(direc):
    file_txts = [x for x in os.listdir(direc) if x.endswith('.txt')]
    for file_txt in file_txts:
        tag_line(os.path.join(direc, file_txt))

def tag_line(txt_clean_fp):
    anns = []
    ann_jsons_fp = txt_clean_fp.replace('.txt', '.ann.jsons')
    with open(ann_jsons_fp, 'r', encoding='utf8') as r:
        for line in r.readlines():
            js = json.loads(line)
            anns.append(ANN(js['seq'], js['typ'], js['st'], js['ed'], js['name']))

    with open(txt_clean_fp, 'r', encoding='utf8') as r:
        lines = [x.strip('\n') for x in r.readlines()]
        lines_tag = ['O'] * len(lines)
        for ann in anns:
            lines_tag[ann.st: ann.ed] = get_ann_tag(ann.typ, ann.st, ann.ed)

        txt_tag_fp = txt_clean_fp + '.tag'
        with open(txt_tag_fp, 'w', encoding='utf8') as w:
            for ori, tag in zip(lines, lines_tag):
                w.write(ori + '\t' + tag + '\n')

        logging.info('save tag file to: %s' % txt_tag_fp)

def tag_lines_with_no_space_lseg(direc):
    ''' remove space and LSEG from ./train to ./train2 '''
    file_tags = util.file_paths_with_postfix(direc, '.txt.tag')
    for file_tag in file_tags:
        nlines = []
        with open(file_tag, 'r', encoding='utf8') as r:
            lines = r.readlines()
            for line in lines:
                word, tag = line.split('\t')
                if word == ' ' or word == 'LSEG':
                    continue
                nlines.append(line)
        savefp = file_tag.replace(direc, './train2')
        with open(savefp, 'w', encoding='utf8') as w:
            for nline in nlines:
                w.write(nline)
        print('save to: %s' % savefp)

def cli_clean_txt():
    '''./clean.py ruijin_round1_train2_20181022 ./train'''
    direc = sys.argv[1]
    save_dir = sys.argv[2]
    loglevel = sys.argv[-1] if len(sys.argv) > 3 else "INFO"
    logging.basicConfig(level=logging.getLevelName(loglevel))
    clean_txt(direc, save_dir)

def cli_tag_lines():
    ''' tag ann lines on clean txt '''
    direct = sys.argv[1]
    loglevel = sys.argv[-1] if len(sys.argv) > 2 else "INFO"
    logging.basicConfig(level=logging.getLevelName(loglevel))
    tag_lines(direct)

def cli_tag_line():
    loglevel = sys.argv[-1] if len(sys.argv) > 2 else "INFO"
    logging.basicConfig(level=logging.getLevelName(loglevel))
    tag_line(sys.argv[1])

def cli_tag_lines_with_no_space_lseg():
    tag_lines_with_no_space_lseg('./train')

def cli_parse_anns():
    direc = sys.argv[1]
    save_dir = sys.argv[2]
    loglevel = sys.argv[-1] if len(sys.argv) > 3 else "INFO"
    logging.basicConfig(level=logging.getLevelName(loglevel))
    parse_anns(direc, save_dir)

def main():
    # clean the txt file
    cli_clean_txt()

    # parse anns file
    cli_parse_anns()

    # tag ann
    cli_tag_lines()
    #cli_tag_lines_with_no_space_lseg()

if __name__ == '__main__':
    main()
