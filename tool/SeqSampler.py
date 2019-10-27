from torch.utils.data.sampler import Sampler


class SeqBatchSampler(Sampler):
    """NLP使用LSTM时，数据会需要考虑数据的连续性对结果的影响
    当batch间前后断开时，LSTM能获得的知识可能产生变化。

    **************************************************
    为了使用dataloader时，避免由于自动规划batch导致的数据断开，
    SeqBatchSampler，继承Sampler类，实现了batch间数据连续的取样方法
    """
    def __init__(self, data_len, batch_size=1, drop_last=True):
        # 异常处理
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.data_len = data_len
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        batch_num = self.data_len // self.batch_size
        total = batch_num * self.batch_size
        for i in range(batch_num):
            for j in range(self.batch_size):
                batch.append(i + batch_num*j)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        if not self.drop_last:
            if total != self.data_len:
                batch = [total+j for j in range((self.data_len-total))]
                yield batch

    def __len__(self):
        if self.drop_last:
            return self.data_len // self.batch_size
        else:
            return (self.data_len + self.batch_size - 1) // self.batch_size
