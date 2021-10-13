#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/11 6:02 下午
# @Author  : sunchao

import os
import jieba
import json

import logging

from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def open_process_file(file_dir, output_file):
    # read stopwords
    stopwords = set()
    file = open("/Users/sunchao/sources/stopwords-master/baidu_stopwords.txt", 'r', encoding='UTF-8')
    for line in file:
        stopwords.add(line.strip())
    file.close()

    # read file
    for root, _, files in os.walk(file_dir):
        for file in files:
            if not file.endswith('.json'):
                continue
            print(os.path.join(root, file))
            with open(os.path.join(root, file), 'r') as fp:
                # preprocess
                documents = [json.loads(line)['text'] for line in fp]

    # preprocess
    # 将分词、去停用词后的文本数据存储在list类型的texts中
    texts = []
    for line in tqdm(documents):
        words = jieba.lcut(line)

        # 过滤停用词，只保留不属于停用词的词语
        # text = [word for word in words if word not in stopwords]
        texts.append(" ".join(words))

    # save file
    with open(output_file, 'w', encoding='utf-8') as fp:
        for line in texts:
            fp.write('{}\n'.format(line))


def w2v_train(src_file, model_file, vec_file):
    model = Word2Vec(LineSentence(src_file), vector_size=128, window=3, min_count=5, workers=2)
    model.save(model_file)
    model.wv.save_word2vec_format(vec_file, binary=False)


def similar_word(w2v_model_file, word):
    w2v = Word2Vec.load(w2v_model_file)
    print(w2v.wv.most_similar(word))


if __name__ == '__main__':
    processed_file = '../data/processed_text.txt'
    w2v_model_file = '../result/test_w2v.model'
    w2v_vec_file = '../result/test_w2v.vector'
    # open_process_file('/Users/sunchao/sources/data/short_news', processed_file)
    # w2v_train(processed_file, w2v_model_file, w2v_vec_file)
    similar_word(w2v_model_file, '电视剧')
