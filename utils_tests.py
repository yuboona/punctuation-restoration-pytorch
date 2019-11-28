import torch
import numpy as np
import torch.utils.data as data
import data as data_class
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler, BatchSampler
from tool.SeqSampler import SeqBatchSampler


class TxtDataset(data.Dataset):  # 这是一个Dataset子类
    def __init__(self):
        self.Data = np.asarray([[1, 2], [3, 4], [2, 1], [6, 4],
                                [4, 5]])  # 特征向量集合,特征是2维表示一段文本
        self.Label = np.asarray([1, 2, 0, 1, 2])  # 标签是1维,表示文本类别

    def __getitem__(self, index):
        txt = torch.LongTensor(self.Data[index])
        label = torch.LongTensor(self.Label[index])
        return txt, label  # 返回标签

    def __len__(self):
        return len(self.Data)


if __name__ == "__main__":
    # ****测试在列表生成式中使用两个for**********************************************
    # tmp_seqs = ['asd aa', 'as11 qq', 'aaaas 223233']
    # txt_seqs = [i for seq in tmp_seqs for i in seq.split()]
    # print(txt_seqs)
    # foo = [1, 2, 4, 5,3, 4, 5]
    # print(foo[2:4])
    # **************************************************************************

    # *****测试使用dataloader取batchsize的具体情况*********************************
    # test = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    # inputing = torch.tensor(np.array([test[i:i + 3] for i in range(10)]))
    # target = torch.tensor(np.array([test[i:i + 1] for i in range(10)]))
    # torch_dataset = data.TensorDataset(inputing, target)
    # # print(torch_dataset[1])
    # batch = 3
    # loader = data.DataLoader(
    #     dataset=torch_dataset,
    #     batch_size=batch,  # 批大小
    #     # 若dataset中的样本数不能被batch_size整除的话，最后剩余多少就使用多少
    #     # collate_fn=lambda x: (
    #     #     torch.cat(
    #     #         [x[i][j].unsqueeze(0) for i in range(len(x))], 0
    #     #         ).unsqueeze(0) for j in range(len(x[0]))
    #     # )
    # )
    # **************************************************************************

    # 测试自己的PuncDataset对象***************************************************
    """ torch_dataset = data_class.PuncDataset(
                                            './data/train',
                                            './data/vocab',
                                            './data/punc'
                                            )
    print(torch_dataset[:4])
    randSampler = RandomSampler(
        torch_dataset,
        replacement=True,
        num_samples=100
        )  # num_samples是指取总数中的多少个数据作为样本集合，不指定则默认取整个数据集
    SeqSampler = SeqBatchSampler(
        torch_dataset.in_len,
        batch_size=4
        )
    sampler = BatchSampler(
        SequentialSampler(torch_dataset),
        batch_size=4,
        drop_last=True
        )
    loader = data.DataLoader(
        dataset=torch_dataset,
        # batch_size=4,
        shuffle=False,
        batch_sampler=SeqSampler
    )
    count = 0
    # print(torch_dataset.id2index)
    with open('./out/out.dat', 'w', encoding='utf-8') as w:
        for (i, j) in loader:
            # print("first:", i)
            # print("second:", j)
            count += 1
            if count > 3:
                break
            x = i.numpy().tolist()
            for foo in x:
                for foo2 in foo:
                    w.write(torch_dataset.id2word[foo2])
                    w.write(' ')
                    print(torch_dataset.id2word[foo2], end=' ')
                w.write('\n')
                print('\n')
                w.write('*********************************')
                print('*********************************') """
    # ***********************************************************************

    # 测试loader的迭代值***************************************************
    # torch_dataset = data_class.PuncDataset(
    #                                         './data/train',
    #                                         './data/vocab',
    #                                         './data/punc'
    #                                         )
    # print(torch_dataset[:4])
    # randSampler = RandomSampler(
    #     torch_dataset,
    #     replacement=True,
    #     num_samples=100
    #     )  # num_samples是指取总数中的多少个数据作为样本集合，不指定则默认取整个数据集
    # loader = data.DataLoader(
    #     dataset=torch_dataset,
    #     # batch_size=4,
    #     shuffle=False,
    #     sampler=randSampler
    # )
    # for m, (i, j) in enumerate(loader):
    #     print()

a = [1, 2, 3, 4]
a = a[:5]

print(a)