#!/usr/bin/env python

'''
help script
'''
import sys
import logging
import util

def check_val(direct):
    print('checking: %s' % direct)
    file_vals = util.file_paths_with_postfix(direct, '.txt.val')
    for file_val in file_vals:
        with open(file_val, 'r', encoding='utf8') as r:
            lines = r.readlines()
            print(file_val)
            linum = 1
            for line in lines:
                word, ori_tag, merge_tag, res_tag, win_tag = line.strip('\n').split('\t')
                #if (ori_tag.startswith('O') \
                if (ori_tag.startswith('B-') \
                #if ((ori_tag.startswith('B-') or ori_tag == 'O') \
                        and (res_tag.startswith('B-') or (win_tag.startswith('B-')))):
                        #and (win_tag == ori_tag):
                        #and (res_tag == 'O') \
                    print(linum, ':', word, '\t',  ori_tag, '\t', res_tag, '\t', win_tag)
                linum += 1

def main():
    direct =sys.argv[-1] if len(sys.argv) > 1 else "INFO"

    loglevel =sys.argv[-1] if len(sys.argv) > 2 else "INFO"
    logging.basicConfig(level=logging.getLevelName(loglevel))

    check_val(direct)

if __name__ == '__main__':
    main()
