import io
import sys

import numpy as np
import torch
from torch.autograd import Variable

from tool.dataset import NoPuncTextDataset
from model.Net import LSTMPR
from tool.utils import add_punc_to_txt


# Change the standard out encoding to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def main(args):
    dataset = NoPuncTextDataset(args.txt_data, args.vocab, args.punc_vocab)
    model = LSTMPR.load_model(args.model_path, cuda=args.cuda)

    # output result
    if args.output == '-':
        output, endline = print, ''
    else:
        ofile = open(args.output, 'w', encoding='utf-8')
        output, endline = ofile.write, '\n'
        # print(output==print)
    print(len(dataset))
    for i, (id_seq, txt_seq) in enumerate(dataset):
        # print(id_seq.size())
        # Note: As a Input, id_seq must be shape of (batch_size, seq_len)
        id_seq = np.reshape(id_seq, (1, -1))
        if args.cuda:
            inputs = Variable(torch.LongTensor(id_seq).cuda(), volatile=True)
        else:
            inputs = Variable(torch.LongTensor(id_seq), volatile=True)

        hidden = model.init_hidden(batch_size=1)
        # print(model)
        scores, hidden = model(inputs, hidden, train=False)

        scores = scores.view(-1, model.num_class)
        _, predict = torch.max(scores, 1)
        predict = predict.data.cpu().numpy().tolist()

        # add punc to text
        result = add_punc_to_txt(txt_seq, predict, dataset.id2punc)
        # print(result[:30])
        output(result + endline)

    if output != print:
        ofile.close()
        print('file closed!')