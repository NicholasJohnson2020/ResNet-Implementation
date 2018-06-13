import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

def conv3x3(in_planes, out_planes, stride = 1):
  # 3x3 convultion with padding (no dimension reduction)
  return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                   padding = 1, bias = False)

class BasicBlock(nn.Module):
  expansion = 1
  
  def __init__(self, inplanes, planes, stride = 1, downsample = None):
    super(BasicBlock, self).__init__()
    self.
