import numpy as np
import torch
import torch.utils.data as data
import os
from tool.SeqSampler import SeqBatchSampler

"""Dataset 和 Dataloader是torch中的一套工具，
继承并改造Dataset将数据进行必要的格式化，则Dataloader
才能用于从Dataset中load数据

Dataset and Dataloader are essential part of pytorch.
Inherited from them, modified for own needs. Then you can
get data tools fitting your project.
"""


class PuncDataset(data.Dataset):
    """Representing a Dataset

    superclass
    ----------
    data.Dataset :
        Dataset is a abstract class, representing the real data.
    """
    def __init__(self, train_path, vocab_path, punc_path, seq_len=100):
        # 检查文件是否存在
        print(train_path)
        assert os.path.exists(train_path), "train文件不存在"
        assert os.path.exists(vocab_path), "词典文件不存在"
        assert os.path.exists(punc_path), "标点文件不存在"
        self.seq_len = seq_len

        self.word2id = load_vocab(
            vocab_path,
            extra_word_list=['<UNK>', '<END>']
        )
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.punc2id = load_vocab(
            punc_path,
            extra_word_list=[" "]
        )
        self.id2punc = {k: v for (v, k) in self.punc2id.items()}

        tmp_seqs = open(train_path, encoding='utf-8').readlines()
        self.txt_seqs = [i for seq in tmp_seqs for i in seq.split()]
        # print(self.txt_seqs[:10])
        # with open('./txt_seq', 'w', encoding='utf-8') as w:
        #     print(self.txt_seqs, file=w)
        self.preprocess(self.txt_seqs)

    def __len__(self):
        """return the sentence nums in .txt
        """
        return self.in_len

    def __getitem__(self, index):
        """返回指定索引的张量对 (输入文本id的序列 , 其对应的标点id序列)

        Parameters
        ----------
        index : int
            索引
        """
        return self.input_data[index], self.label[index]

    def preprocess(self, txt_seqs: list):
        """将文本转为单词和应预测标点的id pair
        Parameters
        ----------
        txt : 文本
            文本每个单词跟随一个空格，符号也跟一个空格
        """
        input_data = []
        label = []
        input_r = []
        label_r = []
        # txt_seqs is a list like: ['char', 'char', 'char', '*，*', 'char', ......]
        count = 0
        length = len(txt_seqs)
        for token in txt_seqs:
            count += 1
            if count == length:
                break
            if token in self.punc2id:
                continue
            punc = txt_seqs[count]
            if punc not in self.punc2id:
                # print('标点{}：'.format(count), self.punc2id[" "])
                label.append(self.punc2id[" "])
                input_data.append(self.word2id.get(token, self.word2id["<UNK>"]))
                input_r.append(token)
                label_r.append(' ')
            else:
                # print('标点{}：'.format(count), self.punc2id[punc])
                label.append(self.punc2id[punc])
                input_data.append(self.word2id.get(token, self.word2id["<UNK>"]))
                input_r.append(token)
                label_r.append(punc)
        with open('./inp_lbl', 'w', encoding='utf-8') as w:
            print('输入数据是：', input_r, file=w)
            print('输出标签是：', label_r, file=w)
        with open('./inp_lbl_id', 'w', encoding='utf-8') as w:
            print('输入数据是：', input_data, file=w)
            print('输出标签是：', label, file=w)
        if len(input_data) != len(label):
            assert 'error: length input_data != label'
        # code below is for using 100 as a hidden size
        print(len(input_data))
        self.in_len = len(input_data) // 100
        len_tmp = self.in_len * 100
        input_data = input_data[:len_tmp]
        label = label[:len_tmp]

        self.input_data = torch.tensor(np.array(input_data, dtype='i8').reshape(-1, 100))
        self.label = torch.tensor(np.array(label, dtype='i8').reshape(-1, 100))


# class PuncDataset(data.Dataset):
#     """Representing a Dataset

#     superclass
#     ----------
#     data.Dataset :
#         Dataset is a abstract class, representing the real data.
#     """
#     def __init__(self, train_path, vocab_path, punc_path):
#         # 检查文件是否存在
#         print(train_path)
#         assert os.path.exists(train_path), "train文件不存在"
#         assert os.path.exists(vocab_path), "词典文件不存在"
#         assert os.path.exists(punc_path), "标点文件不存在"

#         self.word2id = load_vocab(
#             vocab_path,
#             extra_word_list=['<UNK>', '<END>']
#         )
#         self.id2word = {v: k for k, v in self.word2id.items()}
#         self.punc2id = load_vocab(
#             punc_path,
#             extra_word_list=[" "]
#         )
#         self.id2punc = {k: v for (v, k) in self.punc2id.items()}

#         tmp_seqs = open(train_path, encoding='utf-8').readlines()
#         self.txt_seqs = [i for seq in tmp_seqs for i in seq.split()]
#         # print(self.txt_seqs[:10])
#         with open('./txt_seq', 'w', encoding='utf-8') as w:
#             print(self.txt_seqs, file=w)
#         self.preprocess(self.txt_seqs)

#     def __len__(self):
#         """return the sentence nums in .txt
#         """
#         return self.in_len

#     def __getitem__(self, index):
#         """返回指定索引的张量对 (输入文本id的序列 , 其对应的标点id序列)

#         Parameters
#         ----------
#         index : int
#             索引
#         """
#         return self.input_data[index], self.label[index]

#     def preprocess(self, txt_seqs: list):
#         """将文本转为单词和应预测标点的id pair
#         Parameters
#         ----------
#         txt : 文本
#             文本每个单词跟随一个空格，符号也跟一个空格
#         """
#         input_data = []
#         input_r = []
#         label = []
#         label_r = []
#         punc = " "
#         for token in txt_seqs:
#             if token in self.punc2id:
#                 punc = token
#             else:
#                 input_data.append(self.word2id.get(token, self.word2id["<UNK>"]))
#                 label.append(self.punc2id[punc])
#                 input_r.append(token)
#                 label_r.append(punc)
#                 # 这个设计使得标点符号的下一个单词的label是标点符号，将符号两侧的知识加入到了网络中
#                 punc = " "
#         # with open('./inp_lbl', 'w', encoding='utf-8') as w:
#         #     print('输入数据是：', input_r, file=w)
#         #     print('输出标签是：', label_r, file=w)

#         # code below is for using 100 as a hidden size
#         self.in_len = len(input_data) // 100
#         len_tmp = self.in_len * 100 - 1
#         input_data = input_data[:len_tmp]
#         print(len(input_data))
#         label = label[:len_tmp]
#         input_data.append(self.word2id['<END>'])
#         label.append(self.punc2id[punc])
#         # dt = np.dtype('i8')
#         self.input_data = torch.tensor(np.array(input_data, dtype='i8').reshape(-1, 100))
#         self.label = torch.tensor(np.array(label, dtype='i8').reshape(-1, 100))
#         # print('last: ', self.input_data[-1])
#         # print('len: ', self.in_len)


class NoPuncTextDataset(object):
    """parse text without punctuation
       - used by Inference.py
    Parameters
    ----------
    object : Inherited
        basic object.
    """

    def __init__(self, txt_path, vocab_path, punc_path):
        """[summary]

        Parameters
        ----------
        txt_path : str
            文本路径
        vocab_path : str
            字典路径
        punc_path : str
            标点文件路径
        """
        # 检查文件是否存在
        print(txt_path)
        assert os.path.exists(txt_path), "train文件不存在"
        assert os.path.exists(vocab_path), "词典文件不存在"
        assert os.path.exists(punc_path), "标点文件不存在"

        self.word2id = load_vocab(
            vocab_path,
            extra_word_list=['<UNK>', '<END>']
        )
        # for validation of dataset, can be annotated for speedup
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.punc2id = load_vocab(
            punc_path,
            extra_word_list=[" "]
        )
        self.id2punc = {k: v for (v, k) in self.punc2id.items()}

        tmp_seqs = open(txt_path, encoding='utf-8').readlines()
        self.txt_seqs = [i for seq in tmp_seqs for i in seq.split()]
        # print(self.txt_seqs[:10])
        self.preprocess(self.txt_seqs)

    def __len__(self):
        """return txt文件中的句子数目
        """
        return self.in_len

    def __getitem__(self, index):
        """返回指定索引的张量对 (输入文本id的序列 , 其对应的标点id序列)

        Parameters
        ----------
        index : int
            索引
        """
        return self.input_data[index], self.txt_seqs[index]
        # return self.input_data, self.txt_seqs

    def preprocess(self, txt_seqs: list):
        """Transform the word in .txt to be word_dict_id.

        Parameters
        ----------
        txt_seqs : list
            The word in txt, one by one splited with a whitespace.
        """
        input_data = []
        for token in txt_seqs:
            input_data.append(self.word2id.get(token, self.word2id["<UNK>"]))
        # code below is for using 100 as a hidden size
        length = len(input_data)
        self.in_len = length // 100 if length % 100 == 0 else length//100+1

        # ****************************************************************************
        # At inference phrase, seq lenth don't need be exactly 100
        self.input_data = [input_data[(i)*100:(i+1)*100] for i in range(self.in_len)]
        self.txt_seqs = [txt_seqs[(i)*100:(i+1)*100] for i in range(self.in_len)]

        # self.in_len = 1
        # self.input_data = input_data
        # self.txt_seqs = txt_seqs

        # input_data = input_data[:len_tmp]
        # txt_seqs = txt_seqs[:len_tmp]


def collate_fn(data):
    input_seqs, label_seqs = zip(*data)


def load_vocab(vocab_path, extra_word_list=[], encoding='utf-8'):
    n = len(extra_word_list)
    with open(vocab_path, encoding='utf-8') as vf:
        vocab = {word.strip(): i+n for i, word in enumerate(vf)}
    for i, word in enumerate(extra_word_list):
        vocab[word] = i
    return vocab


def get_loader(train_path, vocab_path, punc_path, batch_size=1):
    """return the dataloader for loading data from own data file

    Parameters
    ----------
    train_path : str
        training data
    vocab_path : str
        vocab for all data
    punc_path : str
        punctuation set in data
    batch_size : int, optional
        batch_size for training, by default 1
    """
    print(train_path)
    dataset = PuncDataset(train_path, vocab_path, punc_path)
    SeqSampler = SeqBatchSampler(
        dataset.in_len,
        batch_size=batch_size
        )
    data_loader = data.DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_sampler=SeqSampler
        )
    return data_loader
