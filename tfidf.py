import math
import copy
import collections

"""idf需要从所有文档中计算出，tf只与当前文档有关"""


corpus = [
    'this is the first document',
    'this is the second second document',
    'and the third one',
    'is this the first document'
]

word_list = []
for i in range(len(corpus)):
    word_list.append(corpus[i].split(' '))
print(word_list)

countlist = []  # 存储每个文档的情况
for i in range(len(word_list)):
    count = collections.Counter(word_list[i])
    countlist.append(count)
print(countlist)


# word可以通过count得到，count可以通过countlist得到
# count[word]可以得到每个单词的词频， sum(count.values())得到整个句子的单词总数
def tf(word, count):
    return count[word] / sum(count.values())


# 统计的是含有该单词的句子数
def n_containing(word, count_list):
    return sum(1 for count in count_list if word in count)


# len(count_list)是指句子的总数，n_containing(word, count_list)是指含有该单词的句子的总数，加1是为了防止分母为0
def idf(word, count_list):
    return math.log(len(count_list) / (n_containing(word, count_list)))


for i, count in enumerate(countlist):
    print("Top words in document {}".format(i + 1))
    print('idf', {word: idf(word, countlist) for word in count})
    scores = {word: tf(word, count) * idf(word, countlist) for word in count}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))


def get_data_stats(examples):
    """Compute the IDF score for each word. Then compute the TF-IDF score."""
    """在整份数据集上计算TF"""
    word_doc_freq = collections.defaultdict(int)
    # Compute IDF
    for i in range(len(examples)):  # 目的是得到某个单词在所有文档中出现过几次
        cur_word_dict = {}
        cur_sent = copy.deepcopy(examples[i].word_list_a)
        if examples[i].text_b:
            cur_sent += examples[i].word_list_b
        for word in cur_sent:  # 出现几次都算一次
            cur_word_dict[word] = 1
        for word in cur_word_dict:
            word_doc_freq[word] += 1
    idf = {}
    for word in word_doc_freq:
        idf[word] = math.log(len(examples) * 1. / word_doc_freq[word])
    # Compute TF-IDF
    tf_idf = {}
    for i in range(len(examples)):
        cur_word_dict = {}
        cur_sent = copy.deepcopy(examples[i].word_list_a)
        if examples[i].text_b:
            cur_sent += examples[i].word_list_b
        for word in cur_sent:
            if word not in tf_idf:
                tf_idf[word] = 0
            tf_idf[word] += 1. / len(cur_sent) * idf[word]
    return {
        "idf": idf,
        "tf_idf": tf_idf,
    }


class InputExample(object):
    def __init__(self, word_list_a, text_b=None):
        self.word_list_a = word_list_a
        self.text_b = text_b


ins = [InputExample(item) for item in word_list]
print(get_data_stats(ins))