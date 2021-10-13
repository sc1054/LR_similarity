#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/24 14:46
# @Author  : sunchao

import os
import jieba
import pandas as pd
from tqdm import tqdm

from multiprocessing import Pool

from distance import ngram_distance, cos_distance, lcs_distance, edit_distance

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


# 读取数据，获取特征，训练LR

def get_text(file_dir, output_file):
    data = []
    for root, _, files in os.walk(file_dir):
        for file in files:
            if not file.endswith('train.data') or file.startswith('STS-B'):
                continue
            print(os.path.join(root, file))
            with open(os.path.join(root, file), 'r') as fp:
                for line in fp:
                    line = line.strip().split()
                    if len(line) != 3:
                        continue
                    data.append(line)
    df = pd.DataFrame(data, columns=['text1', 'text2', 'label'])
    df.to_csv(output_file, index=None)


def get_feature(items):
    # item = ['text1', 'text2', 'label']

    ans = []
    for item in tqdm(items, total=len(items)):
        str1, str2, label = item[0], item[1], item[-1]
        # str1, str2 = jieba.lcut(str1), jieba.lcut(str2)

        uni_gram_dis = ngram_distance(str1, str2, 1)
        bi_gram_dis = ngram_distance(str1, str2, 2)
        tri_gram_dis = ngram_distance(str1, str2, 3)

        lcs_dis = lcs_distance(str1, str2)
        edit_dis = edit_distance(str1, str2)

        # s1_vec, s2_vec = get_w2v(cut_str1), get_w2v(cut_str2)
        # cos_dis = cos_distance(s1_vec, s2_vec)

        ans.append([item[0], item[1], uni_gram_dis, bi_gram_dis, tri_gram_dis, lcs_dis, edit_dis, label])
    return ans


def cal_feature_file(input_file, output_file, num_works=4):
    df = pd.read_csv(input_file)

    # 计算特征 n-gram、LCS、Edit、length、cos
    data = df.values.tolist()  # [['text1', 'text2', 'label'], ...]

    batch_size = 5000
    batch_data = [data[i:(i + batch_size)] for i in range(0, len(data), batch_size)]

    result = []

    p = Pool(processes=num_works)
    for sample in tqdm(batch_data, total=len(batch_data)):
        # print(sample)
        result.append(p.apply_async(func=get_feature, args=(sample,)))
    p.close()
    p.join()

    ans = []
    for res in result:
        ans.extend(res.get())
    # result = [res.get() for res in result]
    print('length of data: ', len(ans))

    df = pd.DataFrame(ans, columns=['text1', 'text2', 'uni-gram', 'bi-gram', 'tri-gram', 'lcs', 'edit', 'label'])
    df.to_csv(output_file, index=None)


def get_w2v(text):
    pass


def train(input_file, saved_model):
    df = pd.read_csv(input_file)
    print(df.columns)
    print(df['label'].value_counts())
    print(len(df[df['label'] == 0]) / len(df))

    x, y = df[['uni-gram', 'bi-gram', 'tri-gram', 'lcs', 'edit']], df['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # if os.path.exists(saved_model):
    if False:
        clf = joblib.load(saved_model)
    else:
        clf = LogisticRegression(penalty='l2', C=100, random_state=100)
        clf.fit(x_train, y_train)
        # 保存模型
        joblib.dump(clf, saved_model)

    print(clf.coef_)
    print(clf.get_params())
    print(clf.score(x_test, y_test))


def eval_file(saved_model, input_file, save_test_file):
    df = pd.read_csv(input_file)
    print(df.columns)
    print(df['label'].value_counts())

    X, y = df[['text1', 'text2', 'uni-gram', 'bi-gram', 'tri-gram', 'lcs', 'edit']], df['label']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    text1, text2 = x_test['text1'], x_test['text2']
    x_train = x_train[['uni-gram', 'bi-gram', 'tri-gram', 'lcs', 'edit']]
    x_test = x_test[['uni-gram', 'bi-gram', 'tri-gram', 'lcs', 'edit']]
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    clf = joblib.load(saved_model)
    pred = clf.predict(x_test)
    prob = clf.predict_proba(x_test)
    # print(pred)
    # print(prob[:, 1])
    prob, pred = prob[:, 1].tolist(), pred.tolist()
    flag = [1 if y_test.tolist()[i] == pred[i] else 0 for i in range(len(pred))]
    # print(sum(flag) / len(flag))
    df_test = pd.DataFrame(
        {'text1': text1, 'text2': text2, 'label': y_test, 'pred': pred, 'prob': prob, 'flag': flag})
    df_test.to_csv(save_test_file, index=None)

    print(clf.coef_)
    print(clf.score(x_test, y_test))


if __name__ == '__main__':
    # get_text('/Users/sunchao/sources/data/senteval_cn', './data/text2df_atec.csv')
    # # jieba.cut('aaaa')
    # cal_feature_file('./data/text2df_atec.csv', './data/text2df_atec_feature.csv')
    # train('./data/text2df_atec_feature.csv', 'result/atec_lr.model')
    eval_file('result/atec_lr.model', './data/text2df_atec_feature.csv', 'result/text2df_atec_res.csv')
