import random
import os
import numpy as np
import torch
import pickle


def random_seed(SEED):
    
      random.seed(SEED)
      os.environ['PYTHONHASHSEED'] = str(SEED)
      np.random.seed(SEED)
      torch.manual_seed(SEED)
      torch.cuda.manual_seed(SEED)
      torch.cuda.manual_seed_all(SEED)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False



def pickle_dump(obj, path):
  with open(path, mode='wb') as f:
    pickle.dump(obj,f)


def pickle_load(path):
  with open(path, mode='rb') as f:
    data = pickle.load(f)
    return data


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def softmax(x):

    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def get_ranking(array):
     org_shape = array.shape
     array = array.reshape(-1)
     array = np.arange(len(array))[array.argsort().argsort()]
     array = array / array.max()
     array = array.reshape(org_shape)
     return array

# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    # pixels = (pixels >= thr).astype(int)
    
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    
    runs[::2] = np.where(runs[::2]>0,runs[::2]-1,runs[::2]) # offset 1 pixel
    
    
    return ' '.join(str(x) for x in runs)