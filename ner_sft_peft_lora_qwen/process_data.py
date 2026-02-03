# !/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import json


def strQ2B(ustring):
    """把字符串全角转半角"""
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)


def process_entity_extraction_2_generation(filepath):
    '''解析命名实体识别文本，输出为字典，结果如：

    {
        'input':
'找出文本中的实体\n文本如下\n目的补气健脾升清法治疗糖尿病性功能性消化不良的临床
效果',
        'target': '甘草:中药, 糖尿病性功能性消化不良:西医诊断'
    }
    '''
    data = []
    input, entity, entity_name = '', '', ''
    entities = []
    with open(filepath, 'r', encoding='utf8') as f:
        for l in f:
            l = l.rstrip()
            l = strQ2B(l)
            if not l:
                if entity:
                    entities.append(f'{entity}:{entity_name}')
                    entity = ''
                target = ', '.join(entities) if entities else '未发现实体'
                data.append({'input': '找出文本中的实体\n文本如下\n'+input, 'target': target})
                input = ''
                entities = []
                continue
            word = l[:1]
            BIO_entity = l[1:].strip()
            input+=word
            if BIO_entity.startswith('O'):
                continue
            BIO, entity_name_ = BIO_entity.split('-')
            if BIO=='B':
                if entity:
                    entities.append(f'{entity}:{entity_name}')
                    entity = ''
                entity+=word
                entity_name = entity_name_
            elif BIO=='I':
                entity+=word
    return data


if __name__ == "__main__":
    data_dir = './dataset/raw/'
    process_dir = "./dataset/processed/"
    for part in ['train','test','dev']:
        data = process_entity_extraction_2_generation(os.path.join(data_dir, f'medical.{part}'))
        with open(os.path.join(process_dir, f'medical_{part}.jsonl'), 'w') as f:
           for d in data:
               f.write(json.dumps(d,ensure_ascii=False) + '\n')
