import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from  torchvision import models


class YoloModel(object):
	"""docstring for YoloModel"""
	def __init__(self, arg):
		super(YoloModel, self).__init__()
