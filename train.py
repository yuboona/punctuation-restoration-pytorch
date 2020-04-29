import argparse
import io
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tool.dataset import get_loader, load_vocab
from model.Net import LSTMPR

import tool.utils as utils


parser = argparse.ArgumentParser(description='lstm punctuation prediction training.')

# training data path
parser.add_argument(
    '--train_data', '-t', default='',
    help='training text data path'
)
# check&validation data path
parser.add_argument(
    '--cv_data', '-c', default='',
    help='cross validation text data path'
)
# vocab data path
parser.add_argument(
    '--vocab', '-v', default='',
    help='training text data path'
)
# punc_vocab data path
parser.add_argument(
    '--punc_vocab', '-p', default='',
    help='training text data path'
)
# continue from
parser.add_argument(
    '--continue_from', '-cf', default='',
    help='continue from checkpoint model'
)
# ######## save and load model
parser.add_argument(
    '--save_folder', '-s', default='./tmp',
    help='location to save epoch model'
)
parser.add_argument(
    '--checkpoint', dest='checkpoint', action='store_true',
    help='Enables checkpoint saving of model'
)
# ######## training hyper param
# batch_size
parser.add_argument(
    '--batch_size', '-b', default=10, type=int,
    help='training text data path'
)
# L2
parser.add_argument(
    '--L2', '-l', default=0, type=float,
    help='L2 regularization'
)
# tell me epoch size
parser.add_argument(
    '--epochs', '-e', default=32, type=int,
    help='Number of training epochs'
)
# set the max_norm for clip the gradient, prevent gradient explosion
parser.add_argument(
    '--max_norm', '-m', default=250, type=int,
    help='Norm cutoff to prevent explosion of gradients'
)
# eraly stop
parser.add_argument(
    '--early_stop', '-es', dest='early_stop', action='store_true',
    help='Early stop training when get small improvement'
)
# learning rate
parser.add_argument(
    '--lr', '--learning-rate', default=1e-2, type=float,
    help='Initial learning rate (now only support Adam)'
)

# ******************************************************************************

# logging
parser.add_argument(
    '--verbose', '-vb', dest='verbose', action='store_true',
    help='Watching training process'
)
parser.add_argument(
    '--print_freq', '-pf', default=1000, type=int,
    help='Frequency of printing training infomation'
)

# ######## model hyper parameter
parser.add_argument(
    '--num_class', default=3, type=int,
    help='Number of output classes. (Include blank space " ")'
)
# ******************************************************************************
# ######## save and load model
parser.add_argument(
    '--model_path', default='final.pth.tar',
    help='Location to save best validation model'
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
    args : [type]
        [description]
    cross_valid : bool, optional
        [description], by default False
    """
    total_loss = 0.0
    total_acc = 0.0
    total_words = 0
    hidden = model.init_hidden(args.batch_size)
    start = time.time()
    for i, (inputs, labels) in enumerate(data_loader):
        # 1. mini_batch data******************************************************
        inputs = Variable(inputs, requires_grad=False)
        labels = Variable(labels, requires_grad=False)

        # 2. forward and compute loss**********************************************
        optimizer.zero_grad()
        # forward compute, use *model()* call the forward()
        # scores = model(inputs)
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
            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)
            # 4. update************************************************************
            optimizer.step()

        total_loss += loss.data
        # argmax是保持大小关系的到百分比数值的映射，不用argmax也有max
        _, predict = torch.max(scores, 1)  # score is `[(0,1,2), (), ...]`
        correct = (predict == labels.view(-1)).sum().data
        # words in total
        words = 1
        for s in inputs.size():
            words *= s
        acc = 100.0 * correct / words
        total_acc += acc
        total_words += words

        if args.verbose and i % args.print_freq == 0:
            print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                  'Perplexity {3:.3f} | Acc {4:.3f} | {5} words/s | '
                  '{6:.1f} ms/batch'.format(
                      epoch + 1, i, total_loss / (i + 1),
                      np.exp(total_loss / (i + 1)),
                      total_acc / (i + 1),
                      int(total_words / (time.time()-start)),
                      1000*(time.time()-start)/(i+1)),
                  flush=True)
        del loss
        del scores
    return total_loss / (i + 1), total_acc / (i+1)


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
        # hidden_size is highly related to dataset class split
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
    if not os.path.exists(save_folder):
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
        avg_loss, avg_acc = run_one_epoch(tr_data_loader, model, criterion,                                         optimizer, epoch, args)
        print('-'*85)
        print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
              'Train Loss {2:.3f} | Train Acc {3:.2f} '.format(
                  epoch + 1,
                  time.time() - start,
                  avg_loss,
                  avg_acc))
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
        val_loss, val_acc = run_one_epoch(
                                            cv_data_loader, model,
                                            criterion, optimizer,
                                            epoch, args, cross_valid=True
                                            )
        print('-'*85)
        print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
              'Valid Loss {2:.3f} | Valid Acc {3:.2f} '.format(
                epoch + 1, time.time() - start, val_loss, val_acc))
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


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
