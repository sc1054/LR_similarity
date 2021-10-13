#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/11 11:52 上午
# @Author  : sunchao


""""python计算TFIDF"""
import jieba
from collections import Counter

from gensim import corpora
from gensim.models import Word2Vec, TfidfModel


# 准备数据：现有8条文本数据，将8条文本数据放入到list中
documents = ["1)键盘是用于操作设备运行的一种指令和数据输入装置，也指经过系统安排操作一台机器或设备的一组功能键（如打字机、电脑键盘）",
             "2)鼠标称呼应该是“鼠标器”，英文名“Mouse”，鼠标的使用是为了使计算机的操作更加简便快捷，来代替键盘那繁琐的指令。",
             "3)中央处理器（CPU，Central Processing Unit）是一块超大规模的集成电路，是一台计算机的运算核心（Core）和控制核心（ Control Unit）。",
             "4)硬盘是电脑主要的存储媒介之一，由一个或者多个铝制或者玻璃制的碟片组成。碟片外覆盖有铁磁性材料。",
             "5)内存(Memory)也被称为内存储器，其作用是用于暂时存放CPU中的运算数据，以及与硬盘等外部存储器交换的数据。",
             "6)显示器（display）通常也被称为监视器。显示器是属于电脑的I/O设备，即输入输出设备。它是一种将一定的电子文件通过特定的传输设备显示到屏幕上再反射到人眼的显示工具。",
             "7)显卡（Video card，Graphics card）全称显示接口卡，又称显示适配器，是计算机最基本配置、最重要的配件之一。",
             "8)cache高速缓冲存储器一种特殊的存储器子系统，其中复制了频繁使用的数据以利于快速访问。"]
# 待比较的文档
new_doc = "内存又称主存，是CPU能直接寻址的存储空间，由半导体器件制成。"

print('预处理预料，筛掉低频词、语气词')
stopwords = set()
processed_doc = [[word for word in jieba.cut(item) if word not in stopwords] for item in documents]

# 1、记录语料库中词频大于等于2的词汇构成词典
counter = Counter((word for sample in processed_doc for word in sample))
counter = dict(filter(lambda xx: xx[1] >= 2, sorted(counter.items(), key=lambda x: x[1], reverse=True)))
processed_doc = [[word for word in item if word in counter] for item in processed_doc]
for doc in processed_doc:
    print(doc)

# 2.创建字典（单词与编号之间的映射）
print('创建字典（单词与编号之间的映射）')
dictionary = corpora.Dictionary(processed_doc)
print(dictionary)
# 打印字典，key为单词，value为单词的编号
print(dictionary.token2id)
# dictionary.save('./src/dict4tfidf.dict')
# dictionary = corpora.Dictionary.load('./src/dict4tfidf.dict')

print('建立语料库')
# 将每一篇文档转换为向量
corpus = [dictionary.doc2bow(text) for text in processed_doc]
for item in corpus:
    print(item)

print('初始化模型')
# 初始化一个tfidf模型,可以用它来转换向量（词袋整数计数），表示方法为新的表示方法（Tfidf 实数权重）
tfidf = TfidfModel(corpus, normalize=False)
tfidf.save("./src/my_model.tfidf")

# 将整个语料库转为tfidf表示方法
# 载入模型
tfidf = TfidfModel.load("./src/my_model.tfidf")
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)


