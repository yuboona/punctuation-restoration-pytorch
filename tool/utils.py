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

