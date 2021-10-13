#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/8 1:54 下午
# @Author  : sunchao

import nltk
# from nltk import ngrams

import numpy as np
from collections import Counter


def ngram_distance(s1, s2, ngram=1):
    ngram_1 = Counter(ngrams(s1, ngram))
    ngram_2 = Counter(ngrams(s2, ngram))
    common = float(len(set(ngram_1) & set(ngram_2)))
    total = float(len(set(ngram_1) | set(ngram_2)))
    if total == 0:
        return 0.
    return common / total


def cos_distance(s1_w2v, s2_w2v):
    """
    计算两向量之间的cos距离（余弦距离=1-余弦相似度）
    :param s1_w2v: 词向量矩阵 shape [batch, dim]
    :param s2_w2v: [batch, dim]
    :return: [-1, 1]
    """
    dot_sum = np.sum(s1_w2v * s2_w2v, axis=-1)
    l2_s1 = np.linalg.norm(s1_w2v, axis=-1)
    l2_s2 = np.linalg.norm(s2_w2v, axis=-1)
    return 1 - dot_sum / (l2_s1 * l2_s2)


def lcs_distance(s1, s2):
    """"
    计算s1和s2的最长公共子序列
    s1为query，s2为库中数据
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # return dp[m][n] / min(m, n)  不合理
    return dp[m][n] / (m + n)


def edit_distance(s1, s2):
    """计算s1与s2的编辑距离"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i][j - 1] + 1, dp[i - 1][j] + 1, dp[i - 1][j - 1] + 1)

    return dp[m][n] / (m + n)


def ngrams(seq, n):
    """
    :param seq: 待计算的句子 type:str or list
    :param n: ngram
    :return: list
    """
    if len(seq) <= n:
        return seq if isinstance(seq, list) else [seq]
    return [''.join(item) for item in zip(*[seq[i:] for i in range(n)])]


if __name__ == '__main__':
    import jieba
    print(ngram_distance(jieba.lcut('我在高速公路上开车呢'),
                         jieba.lcut('在高速公路上开车呢啊'),
                         2))
    # print(edit_distance('horse', 'ros'))
    print(ngrams('在高速公路上呢'))

