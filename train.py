import argparse
import io
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from data import get_loader, load_vocab
from Net import LSTMPR

import utils


parser = argparse.ArgumentParser(description='lstm punctuation prediction training.')
# training data path
parser.add_argument(
                    '--train_data',
                    '-t',
                    default='',
                    help='training text data path'
                    )
# check&validation data path
parser.add_argument(
                    '--cv_data',
                    '-c',
                    default='',
                    help='cross validation text data path'
                    )
# vocab data path
parser.add_argument(
                    '--vocab',
                    '-v',
                    default='',
                    help='training text data path'
                    )
# punc_vocab data path
parser.add_argument(
                    '--punc_vocab',
                    '-p',
                    default='',
                    help='training text data path'
                    )
# batch_size
parser.add_argument(
                    '--batch_size',
                    '-b',
                    default='',
                    help='training text data path'
                    )
# L2
parser.add_argument(
                    '--L2',
                    '-l',
                    default=0,
                    type=float,
                    help='L2 regularization'
                    )
# continue from
parser.add_argument(
    '--continue_from',
    '-cf',
    default='',
    help='continue from checkpoint model'
)
# save folder
parser.add_argument(
    '--save_folder',
    '-s',
    default='./tmp',
    help='location to save epoch model'
)
# tell me epoch size
parser.add_argument(
    '--epochs',
    '-e',
    default=32,
    type=int,
    help='Number of training epochs'
)
# set the max_norm for clip the gradient, prevent gradient explosion
parser.add_argument(
    '--max-norm',
    default=250,
    type=int,
    help='Norm cutoff to prevent explosion of gradients'
)



# def the func to train a epoch's data
def run_one_epoch(
    data_loader,
    model,
    criterion,
    optimizer,
    epoch,
    args,
    cross_valid=False
):
    total_loss = 0.0
    total_acc = 0.0
    total_words = 0
    start = time.time()
    for i, (inputs, labels) in enumerate(data_loader):
        # 1. mini_batch data******************************************************
        inputs = Variable(inputs, requires_grad=False)
        labels = Variable(labels, requires_grad=False)

        # 2. forward and compute loss**********************************************
        optimizer.zero_grad()
        # forward compute, use *model()* call the forward()
        scores = model(inputs, lengths)
        scores = scores.view(-1, args.num_class)
        # criterion() receive a 1.train softmax out and a 2.labels to compute the CrossEntropy
        loss = criterion(scores, labels.view(-1))
        if not cross_valid:
            # 3. backward()*********************************************************
            loss.backward()
            # Clip gradient
            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)
            # 4. update************************************************************
            optimizer.step()

        total_loss += loss.data[0]

    


def main(args):
    # IO load data from file. loader is a iter, format: batch of (inputs, labels)*****
    tr_data_loader = get_loader(
        args.train_data,
        args.vocab,
        args.punc_vocab,
        batch_size=args.batch_size
    )
    cv_data_loader = get_loader(
        args.cv_data,
        args.vocab,
        args.punc_vocab,
        batch_size=args.batch_size
    )

    # load vocab and punc***************************************************************
    vocab = load_vocab(
            args.vocab,
            extra_word_list=['<UNK>', '<END>']
    )
    vocab_len = len(vocab)

    punc = load_vocab(
        args.punc_vocab,
        extra_word_list=[" "]
    )
    num_class = len(punc)

    # Model*****************************************************************************
    model = LSTMPR(
        vocab_size=vocab_len,
        embedding_size=100,
        hidden_size=100,
        num_layers=1,
        num_class=num_class
    )

    # Loss****************************************************************************
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # optimizer***********************************************************************
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.L2
    )

    # log****************************************************************************
    print(model)
    print('Number of parameters:{}'.format(utils.num_param(model)))

    # restore model information from specified model file ***************************
    if args.continue_from:
        print('Loading checkpoint model {}', args.continue_from)
        package = torch.load(args.continue_from)
        model.load_state_dict(package['static_dict'])
        optimizer.load_state_dict(package['optim_dict'])
        start_epoch = int(package.get('epoch', 1))
    else:
        start_epoch = 0

    # Create save folder**************************************************************
    save_folder = args.save_folder
    utils.mkdir(save_folder)

    prev_val_loss = float('inf')
    best_val_loss = float('inf')
    halving = False

    # Train model multi-epochs********************************************************
    for epoch in range(start_epoch, args.epochs):
        print('Training...')
        # set train mode to make train and test be different
        model.train()
        start = time.time()
        avg_loss, avg_acc = run_one_epoch()








if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
