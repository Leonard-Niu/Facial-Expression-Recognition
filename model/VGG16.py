#-----------------------------------------
# written by Leonard Niu
# HIT
#-----------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# image_size = 240 * 240
# input_size = 224 * 224
# output_size = 7

cfg = {'VGG16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature = self.make_layer(cfg['VGG16'])
        self.fc1 = nn.Linear(7 * 7 * 512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 7)
        

    def make_layer(self, para):
        layer = []
        inch = 3
        for x in para:
            if x == 'M':
                layer += [nn.MaxPool2d(2, 2)]
            else:
                layer += [
                        nn.Conv2d(inch, x, 3, padding=1, stride=1),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True)
                ]
                inch = x
        #layer += [nn.MaxPool2d(2, 2)]
        return nn.Sequential(*layer)
    
    def forward(self, x):
        out = self.feature(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = F.dropout(F.relu(self.fc1(out)))
        out = F.dropout(F.relu(self.fc2(out)))
        out = self.fc3(out)
        return out

#net = Net()
#print (net)