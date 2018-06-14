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
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace = True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2=nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride
 
  def forward(self, x):
    residual = x
    
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    
    out = self.conv2(out)
    out = self.bn2(out)
    
    if self.downsample is not None:
      residual = self.downsample(x)
    
    out += residual
    out = self.relu(out)
    
    return out
  
    
class Bottleneck(nn.Module):
  expansion = 4
  
  def __init__(self, inplanes, planes, stride = 1, downsample = None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 1, bias = False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride,
                           padding = 1, bias = False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size = 1,
                           bias = False)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion)
    self.relu = nn.ReLU(inplace = True)
    self.downsample = downsample
    self.stride = stride
  
  def forward(self, x):
    residual = x
    
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)
    
    out = self.conv3(out)
    out = self.bn3(out)
    
    if self.downsample is not None:
      residual = self.downsample(x)
      
    out += residual
    out = self.relu(out)
    
    return out

class PreactBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride = 1, downsample = None):
    super(PreactBlock, self).__init__()
    self.bn1 = nn.BatchNorm2d(inplanes)
    self.relu = nn.ReLU(inplace = True)
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv2 = conv3x3(planes, planes, stride)
    self.downsample = downsample
    self.stride = stride
    
  def forward(self, x):
    residual = x
    
    out = self.bn1(x)
    out = self.relu(out)
    out = self.conv1(out)
    
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv2(out)
    
    if self.downsample is not None:
      residual = self.downsample(x)
      
    out += residual
    
    return out
      
class ResNet(nn.Module):
  
  def __init__(self, block, layers, num_classes = 10):
    self.inplanes = 16
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1,
                           padding = 1, bias = False)
    self.bn1 = nn.BatchNorm2d(16)
    self.relu = nn.ReLU(inplane = True)
    self.layer1 = self._make_layer(block, 16, layers[0])
    self.layer2 = self._make_layer(block, 32, layers[1])
    self.layer3 = self._make_layer(block, 64, layers[2])
    self.avgpool = nn.AvgPool2d(8, stride = 1)
    self.fc = nn.Linear(64, num_classes)
