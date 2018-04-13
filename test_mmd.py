from __future__ import print_function
import os
import torch 
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as torchvar 
import torch.optim as optim

import numpy as np
import pickle
import PIL.Image
from multiprocessing import Pool

#Inference code for Image Tampering Detection

cuda = torch.cuda.is_available()
def get_patches(img, size = 64):
        ##First get indices of rows
        # Assumes input image is of shape C X H X W
        w, h = img.shape[2], img.shape[1]
        w_flag, h_flag = w - size, h - size
        col_idx, row_idx = [], []
        col_c, row_c = 0, 0 
        while col_c < w_flag:
                col_idx.append(col_c)
                col_c += size
        col_idx.append(w_flag)
        while row_c < h_flag:
                row_idx.append(row_c)
                row_c += size
        row_idx.append(h_flag)
        patches = np.zeros((len(col_idx)*len(row_idx), 3, size, size), dtype ='float32')
        count = 0 
        for i in row_idx:
                for j in col_idx:
                        patch = img[:, i:i+size, j:j+size]
                        patches[count] = patch
                        count += 1
        return patches, col_idx, row_idx

#Define the network. Please check the pdf for more details of the architecture.
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, stride = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, stride = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),128*3*3)
        x = self.classifier(x)
        return x


#This class contains all the things required to deploy the network. Just call the constructor and then the __call__. If the image is tampered, it will return 1 else will return 0. 
class perform_test(object):
        def __init__(self, transform, outpath=None):
                self.transform= transform
                self.outpath = outpath
                self.MMD_bneck_layer_t = nn.Sequential(nn.Dropout(),nn.Linear(512, 256), nn.ReLU(inplace=True))
                self.classifier_source = nn.Sequential(nn.Dropout(), nn.Linear(256, 2))
                self.classifier_target = nn.Sequential(nn.Dropout(), nn.Linear(256, 2))

                self.Net = net()
                a = torch.load('MMD_Net.pth', map_location=lambda storage, loc: storage) 
                b = torch.load('MMD_bneck_layer_t.pth', map_location=lambda storage, loc: storage)
                c = torch.load('MMD_ct.pth', map_location=lambda storage, loc: storage)
                self.Net.load_state_dict(a)
                self.MMD_bneck_layer_t.load_state_dict(b)
                self.classifier_target.load_state_dict(c)

        def __call__(self, img_name):
                self.Net.eval()
                self.classifier_target.eval()
                self.MMD_bneck_layer_t.eval()
                
                img_t = PIL.Image.open(img_name).convert('RGB')
                if self.transform is not None:
                      img_t_new = self.transform(img_t)   
     
                img_np = img_t_new.numpy().squeeze()
                patches, col_idx, row_idx = get_patches(img_np)
                ip = torch.from_numpy(patches).float()
                ip = torchvar(ip)              
                out = self.classifier_target(self.MMD_bneck_layer_t(self.Net(ip)))
                out = out.data.cpu().numpy() #Figure out multiplying by classificatioprobability
                prediction = np.argmax(out, 1).astype(np.int32)
                
                if sum(prediction) > 50:
                        return 1
                else:
                        return 0
                       



#Use the below lines to initialize the above class and then call.

#code added by Kirti to make it importable

if __name__ == "__main__": #execute this code only when the code is executed, not when imported.
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5,0.5,0.5], std = [0.225, 0.225, 0.225])])
    test = perform_test(transform)
    if (test == 1):
        print ("This image is tampered")
    else:
        print ("This image is not tampered")

#print test(img)

