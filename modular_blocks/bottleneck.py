import pdb
import torch
import numpy as np
from PIL import Image

sample_image_like = torch.rand((256, 64, 64)).unsqueeze(0)


class bottleneck(torch.nn.Module):
    def __init__(self):
        super(bottleneck, self).__init__()
        self.down_conv = torch.nn.Conv2d(256, 64, 1, stride=1)
        self.conv3x3 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.up_conv = torch.nn.Conv2d(64, 256, 1, stride=1)
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        print(f'Input shape before bottleneck block is {x.shape}')
        x = self.relu(self.down_conv(x))
        print(f'conv1 shape is {x.shape}')
        x = self.relu(self.conv3x3(x))
        print(f'conv2 shape is {x.shape}')
        x = self.relu(self.up_conv(x))
        print(f'Output shape after bottleneck block is {x.shape}')
        

if __name__ == '__main__':
    bn = bottleneck()
    bn.forward(sample_image_like)