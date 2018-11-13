
import os
import logging

def file_paths_with_postfix(directory, postfix):
    return [os.path.join(directory, x) for x in os.listdir(directory) if x.endswith(postfix)]

def flatten(result_tags):
    return [tag for tags in result_tags for tag in tags]

def merge_win_tags(tags, win_tags):
    ''' merge win_tags to tags'''
    merge_tags = tags
    idx = 0
    for tag, wtag in zip(tags, win_tags):
        if (wtag.startswith('B-') or wtag.startswith('I-')) and tag == 'O':
            merge_tags[idx] = wtag
        idx += 1

    return merge_tags

def collect_result(tag_list):
    '''result_tags: [[O, O, B-Desease, ], ...]'''
    coll = []
    ppos = 0
    while ppos < len(tag_list):
        ppos_tag = tag_list[ppos]
        if ppos_tag.startswith('B-'):
            typ = ppos_tag.split('-')[1] # get type
            pos = ppos + 1
            while pos < len(tag_list) \
                and (not tag_list[pos].startswith('B-')) \
                and (tag_list[pos].startswith('I-') and tag_list[pos].split('-')[1] == typ):
                pos += 1

            #if pos > ppos + 1:    # 去除单字的
            coll.append((ppos, pos))

            ppos = pos

        else:
            ppos += 1

    return coll

def make_senss(lines, sen_len, pad_str):
    '''
    seperate article with sentenses. use override window
    '''
    ori_senss = []    # 源语句
    win_senss = []    # 加窗语句
    lines_num = len(lines)
    if not (lines_num // sen_len == lines_num / sen_len): # padding
        lines.extend([pad_str] * (sen_len - lines_num % sen_len))

    ed = sen_len
    half_sen_len = sen_len // 2    # override window size
    lines_num = len(lines)
    while ed <= lines_num:
        ori_senss.append(lines[ed - sen_len: ed])
        if ed + half_sen_len <= lines_num:
            win_senss.append(lines[ed - sen_len + half_sen_len : ed + half_sen_len])
        ed += sen_len
    return ori_senss, win_senss

def make_submit(save_fp, pos_collects, tags, words):
    i = 0
    with open(save_fp, 'w', encoding='utf8') as w:
        for pos in pos_collects:
            typ = tags[pos[0]].split('-')[1]
            typ_pos = typ + ' ' + str(pos[0]) + ' ' + str(pos[1])
            name = ''.join(words[pos[0]:pos[1]])
            w.write('T%s\t%s\t%s\n' % (i, typ_pos, name))
            i+=1
    logging.info('save ann file to: %s' % save_fp)


def main():
    result_tags = ['O','O','B-a','I-a','B-c','I-c','I-c','B-a','B-c','I-c','O']
    print(collect_result(result_tags))
    print('-'*20)

    result_tags = ['O','O','B-a','I-a','O', 'O', 'B-c','I-c','I-c','B-a','B-c','I-c','O']
    coll = collect_result(result_tags)
    for x in coll:
        print(result_tags[x[0]:x[1]])

    print('-'*20)
    win_tags = flatten([['B-a','I-a','B-a','I-a','O', 'O', 'B-c','I-c','I-c','B-a','B-c','I-c','O']])
    merge_tags = merge_win_tags(result_tags, win_tags)
    coll = collect_result(merge_tags)
    for x in coll:
        print(merge_tags[x[0]:x[1]])


if __name__ == '__main__':
    main()
