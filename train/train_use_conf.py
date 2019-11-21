# -*- coding:utf-8 -*-
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.Net import LSTMPR

from tool.dataset import get_loader, load_vocab
import tool.utils as utils


parser = argparse.ArgumentParser(
    description="Lstm Punctuation Prediction training.",
    usage="In windows CMD or linux bash: execute `python train_use_conf.py > ./log/blstm &`  at root dir of this projct."
    )


def run_one_epoch(
                    data_loader,
                    model,
                    criterion,
                    optimizer,
                    epoch,
                    args,
                    cross_valid=False
                    ):
    """Def the Func to train a epoch's data

    Parameters
    ----------
    data_loader : data.DataLoader
        a dataloader for loading data in a unified way
    model : nn.Module
        self designed nn model
    criterion : CrossEntropyLoss
        Loss function giving the criterion of stop training
    optimizer : Adam

    epoch : [type]
        [description]
    cross_valid : bool, optional
        [description], by default False
    """
    total_loss = 0.0
    total_acc = 0.0
    total_acc_2 = 0.0
    total_words = 0
    hidden = model.init_hidden(args.batch_size)
    # print("hidden_size:", hidden[0].size())
    start = time.time()
    for i, (inputs, labels) in enumerate(data_loader):
        # 1. mini_batch data******************************************************
        inputs = Variable(inputs, requires_grad=False)
        labels = Variable(labels, requires_grad=False)

        # 2. forward and compute loss**********************************************
        optimizer.zero_grad()
        # forward compute, use *model()* call the forward()
        print('', hidden[0].size())
        scores, hidden = model(inputs, hidden, train=True)
        scores = scores.view(-1, args.num_class)
        # criterion() receive a 1.train out and a 2.labels to compute the CrossEntropy
        # it has combined nn.LogSoftmax to compute probability
        loss = criterion(scores, labels.view(-1))
        # 区分是否是训练过程
        if not cross_valid:
            # 3. backward()*********************************************************
            loss.backward()
            # Clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            # 4. update************************************************************
            optimizer.step()

        total_loss += loss.data
        # argmax是保持大小关系的到百分比数值的映射，不用argmax也有max
        _, predict = torch.max(scores, 1)  # score is `[(0,1,2), (), ...]`
        labels = labels.view(-1)
        tmp = (predict == labels)
        res = tmp.sum().data
        correct = [0. for i in range(args.num_class)]
        total = [0. for i in range(args.num_class)]
        for label_idx in range(len(labels)):
            label_single = labels[label_idx]
            correct[label_single] += int(tmp[label_idx].data)
            total[label_single] += 1
        # words in total
        words = 1
        for s in inputs.size():
            words *= s
        acc = 100.0 * res / words
        acc_2 = 100.0 * sum(correct[1:]) / sum(total[1:])
        # acc_2 = 100.0 * correct[5] / total[5]
        total_acc += acc
        total_acc_2 += acc_2
        total_words += words

        if args.verbose and i % args.print_freq == 0:
            print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                  'Perplexity {3:.3f} | Acc {4:.3f} | Acc_2 {7:.3f} | {5} words/s | '
                  '{6:.1f} ms/batch'.format(
                      epoch + 1, i, total_loss / (i + 1),
                      np.exp(total_loss / (i + 1)),
                      total_acc / (i + 1),
                      int(total_words / (time.time()-start)),
                      1000*(time.time()-start)/(i+1),
                      total_acc_2 / (i+1)),
                  flush=True)
        del loss
        del scores
    return total_loss / (i + 1), total_acc / (i+1), total_acc_2 / (i+1)


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
        embedding_size=200,
        hidden_size=100,
        num_layers=args.num_layers,
        num_class=num_class
    )

    # Loss****************************************************************************
    # criterion = nn.CrossEntropyLoss(ignore_index=-1)
    criterion = nn.CrossEntropyLoss(
        ignore_index=-1,
        weight=torch.from_numpy(
            np.array([1, 1, 1, 1, 1, 1])
            # np.array([2, 3, 30, 15, 30, 0.1])
            # np.array([2, 1, 15, 10, 15, 0.1])
            # np.array([8, 2, 15, 10, 15, 0.5])
            # np.array([15, 24, 243, 117, 269, 1])
            # np.array([1.5, 2.4, 24.3, 11.7, 26.9, 0.1])


            ).float()
        )

    # optimizer***********************************************************************
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.L2
    )

    # log*************************************************************************
    print(model)
    print('Number of parameters:{}'.format(utils.num_param(model)))

    # restore model information from specified model file **********************
    if args.continue_from:
        print('Loading checkpoint model {}', args.continue_from)
        package = torch.load(args.continue_from)
        model.load_state_dict(package['static_dict'])
        optimizer.load_state_dict(package['optim_dict'])
        start_epoch = int(package.get('epoch', 1))
    else:
        start_epoch = 0

    # Create save folder********************************************************
    save_folder = args.save_folder
    utils.mkdir(save_folder)

    prev_val_loss = float('inf')
    best_val_loss = float('inf')
    # 二分学习率，以小步前进寻找最优
    halving = False

    # Train model multi-epochs**************************************************
    for epoch in range(start_epoch, args.epochs):
        print('Training...')
        # set train mode to make train and test be different
        model.train()
        start = time.time()
        avg_loss, avg_acc, avg_acc_2 = run_one_epoch(tr_data_loader, model, criterion,                                         optimizer, epoch, args)
        print('-'*85)
        print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
              'Train Loss {2:.3f} | Train Acc {3:.2f} | Train Acc_2 {4:.2f} '.format(
                  epoch + 1,
                  time.time() - start,
                  avg_loss,
                  avg_acc,
                  avg_acc_2))
        print('-'*85)

        # ######## Save model at each epoch
        if args.checkpoint:
            file_path = os.path.join(
                save_folder,
                'epoch{}.pth.tar'.format(epoch+1)
                )
            torch.save(LSTMPR.serialize(model, optimizer, epoch+1), file_path)
            print('Saving checkpoint model to {}'.format(file_path))

        # ######## Cross validation
        print('Cross validation...')
        # 进入验证状态
        model.eval()
        start = time.time()
        val_loss, val_acc, val_acc_2 = run_one_epoch(
                                            cv_data_loader, model,
                                            criterion, optimizer,
                                            epoch, args, cross_valid=True
                                            )
        print('-'*85)
        print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
              'Valid Loss {2:.3f} | Valid Acc {3:.2f} | Valid Acc_2 {4:.2f}'.format(
                epoch + 1, time.time() - start, val_loss, val_acc, val_acc_2))
        print('-'*85)

        # ######## Adjust learning rate, halving
        if val_loss >= prev_val_loss:
            if args.early_stop and halving:
                print("Already start halving learing rate, it still gets too "
                      "small imporvement, stop training early.")
                break
            halving = True
        if halving:
            optim_state = optimizer.state_dict()
            optim_state['param_groups'][0]['lr'] = (
                optim_state['param_groups'][0]['lr'] / 2.0
                )
            optimizer.load_state_dict(optim_state)
            print('Learning rate adjusted to: {lr:.6f}'.format(
                  lr=optim_state['param_groups'][0]['lr']))
        prev_val_loss = val_loss

        # ######## Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            file_path = os.path.join(save_folder, args.model_path)
            torch.save(
                LSTMPR.serialize(model, optimizer, epoch+1),
                file_path)
            print("Find better validated model, saving to %s" % file_path)
