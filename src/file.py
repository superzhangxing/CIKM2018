from configs import cfg
from src.record_log import _logger
import json
import numpy as np
import pickle
import os

def save_file(data, filePath, dataName='data', mode='pickle'):
    _logger.add()
    _logger.add('Saving %s to %s' %(dataName, filePath))

    if mode == 'pickle':
        with open(filePath, 'wb') as f:
            pickle.dump(obj=data, file=f)
    elif mode == 'json':
        with open(filePath, 'w', encoding='utf-8') as f:
            json.dump(obj=data, fp=f)
    elif mode == 'txt':
        pass
    else:
        raise(ValueError, 'Fuction save_file does not have mode %s' % (mode))
    _logger.add('Done')

def load_file(filePath, dataName='data', mode='pickle'):
    _logger.add()
    _logger.add('Trying to load %s from %s' % (dataName, filePath))
    data = None
    ifLoad = False

    if os.path.isfile(filePath):
        _logger.add('Have found the file, loading...')

        if mode == 'pickle':
            with open(filePath,'rb') as f:
                data = pickle.load(f)
                ifLoad = True
        elif mode == 'json':
            with open(filePath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                ifLoad = True
        elif mode == 'txt':
            pass
        else:
            raise(ValueError, 'Function save_file does not have mode % s' % (mode))
    else:
        _logger.add('Have not found the file')
    _logger.add('Done')
    return (ifLoad, data)
