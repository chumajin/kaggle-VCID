
import timm
from timm import create_model
from pprint import pprint
from timm.data.mixup import Mixup
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import os
import gc

import matplotlib.pyplot as plt
from tqdm import tqdm


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset,Dataset
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2


import gc

import random
import time

import transformers

from transformers import  AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
import warnings
warnings.simplefilter('ignore')

import cv2

import segmentation_models_pytorch as smp
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

from sklearn.metrics import fbeta_score


from sklearn.metrics import roc_auc_score
from transformers import SegformerForSemanticSegmentation
from google.colab import runtime


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # cpuがgpuかを自動判断

