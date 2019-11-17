import os
import errno


def num_param(model):
    """return the param num of models

    Parameters
    ----------
    model : torch.nn.Module
        a model obj of Module
    """
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params


def mkdir(path):
    """create path

    Parameters
    ----------
    path : str
        path string should be created
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise


def add_punc_to_txt(txt_seq: list, predict: list, id2punc: dict):
    """add punc to text

    Parameters
    ----------
    txt_seq : list
        word sequence without punc
    predict : list
        punc_id prediction for every word
    id2punc : dict
        punc_id to punctuation
    """
    txt_predict = ''
    for i, word in enumerate(txt_seq):
        punc = id2punc[predict[i]]
        txt_predict += word if punc == ' ' else punc + word
    # punc = id2punc[predict[i+1]]
    txt_predict += punc
    return txt_predict
