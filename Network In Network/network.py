import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class NiN(nn.Module):
  def __init__(self, num_classes):
    super(NiN, self).__init__()
    self.num_classes = num_classes
    self.first_conv = Conv2D(input, 192, 5, 2)
    self.b3_first_conv = Conv2D(input, 192, 3, 1)
    self.cross_channel = Conv2D(input, output, 1)
  
  def forward(self, x):
    x = F.relu(self.first_conv(x))
    x = F.relu(self.cross_channel(192, 160))  
    x = F.relu(self.cross_channel(160, 96))
    x = F.dropout(F.max_pool2d(x, 3, 2, 1))

    x = F.relu(self.first_conv(x))
    x = F.relu(self.cross_channel(192, 192))
    x = F.relu(self.cross_channel(192, 192))
    x = F.dropout(F.avg_pool2d(x, 3, 2, 1)) 

    x = F.relu(self.b3_first_conv(x))
    x = F.relu(self.cross_channel(192, 192))
    x = F.relu(self.cross_channel(192, 10))
    x = F.avg_pool2d(x, 8, 1)
    

    logits = x.view(x.size(0), self.num_classes)
    probs = torch.softmax(x, dims=1)
    return logits, probs
