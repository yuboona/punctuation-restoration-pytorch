import argparse
import io
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='lstm punctuation prediction training.')
parser.add_argument(
                    '--train_data',
                    '-t',
                    default='',
                    help='training text data path'
                    )
parser.add_argument(
                    '--cv_data',
                    '-c',
                    default='',
                    help='cross validation text data path'
                    )
parser.add_argument(
                    '--vocab',
                    '-v',
                    default='',
                    help='training text data path'
                    )
parser.add_argument(
                    '--punc_vocab',
                    '-p',
                    default='',
                    help='training text data path'
                    )
parser.add_argument(
                    '--batch_size',
                    '-b',
                    default='',
                    help='training text data path'
                    )



def main(args):
    tr_data_loader = get_loader(args.train_data, )



if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)