import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMPR(nn.Module):
    """LSTM for Punctuation Restoration
    """

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 num_class):
        super(LSTMPR, self).__init__()
        # hyper parameters
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_class = num_class

        # 网络中的层
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # print(hidden_size)
        # print(embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        # Here is a one direction LSTM. If bidirection LSTM, (hidden_size*2(,))
        self.fc = nn.Linear(hidden_size*2, num_class)
        # self.fc = nn.Linear(hidden_size, num_class)
        self.init_weights()

    def init_weights(self, init_range=0.1):
        """Init weights

        Parameters
        ----------
        init_range : float
            set range of weight, by default 0.1
        """
        for p in self.parameters():
            if p.dim() > 1:
                p.data.uniform_(-init_range, init_range)
            else:
                p.data.fill_(0)

    def init_hidden(self, batch_size):
        # h = Variable(torch.zeros(self.num_layers*1, batch_size, self.hidden_size))
        # c = Variable(torch.zeros(self.num_layers*1, batch_size, self.hidden_size))

        # when bidirection
        h = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_size))
        c = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_size))
        # h for storing hidden layer weight，c for storing cell states
        return (h, c)

    def reset_hidden(self, hidden):
        h = Variable(hidden[0].data)
        c = Variable(hidden[1].data)
        hidden = (h, c)
        return hidden

    def forward(self, inputs, hidden, train=False):
        """The forward process of Net

        Parameters
        ----------
        inputs : tensor
            Training data, batch first
        """
        # Inherit the knowledge of context
        if train:
            hidden = self.reset_hidden(hidden)
        # hidden = self.init_hidden(inputs.size(0))
        # print('input_size',inputs.size())
        embedding = self.embedding(inputs)
        # print('embedding_size', embedding.size())
        # packed = pack_sequence(embedding, inputs_lengths, batch_first=True)
        # embedding本身是同样长度的，用这个函数主要是为了用pack
        # *****************************************************************************
        outputs, hidden = self.lstm(embedding, hidden)      # 输入pack，lstm默认输出pack
        outputs = outputs.contiguous()
        # print(outputs.size())
        score = self.fc(outputs.view(outputs.size(0)*outputs.size(1), outputs.size(2)))
        return score.view(outputs.size(0), outputs.size(1), score.size(1)), hidden

    @classmethod
    def load_model(cls, path, cuda=True):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(vocab_size=package['vocab_size'],
                    embedding_size=package['embedding_size'],
                    hidden_size=package['hidden_size'],
                    num_layers=package['num_layers'],
                    num_class=package['num_class'])
        model.load_state_dict(package['state_dict'])
        if cuda:
            model.cuda()
        return model


    @staticmethod
    def serialize(model, optimizer, epoch):
        """存储模型的信息

        Parameters
        ----------
        model : nn.module
            模型
        optimizer : nn.optimizer
            优化器
        epoch : 轮
            迭代轮数
        """
        package = {
            'vocab_size': model.vocab_size,
            'embedding_size': model.embedding_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'num_class': model.num_class,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch}
        return package
