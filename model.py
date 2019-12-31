import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Block(nn.Module):
    def __init__(self, in_planes,out_planes,stride=1):
        super(Block,self).__init__()
        self.conv1=conv3x3(in_planes,in_planes)
        self.bn1=nn.BatchNorm2d(in_planes)  
        self.conv2=conv3x3(in_planes,in_planes)
        self.bn2=nn.BatchNorm2d(in_planes)  

        self.conv3=conv3x3(in_planes,in_planes)
        self.bn3=nn.BatchNorm2d(in_planes)  
        self.conv4=conv3x3(in_planes,in_planes)
        self.bn4=nn.BatchNorm2d(in_planes)  

        self.nconv=conv1x1(in_planes,out_planes)
        self.nbn=nn.BatchNorm2d(out_planes)

    def forward(self,x):
        out1=F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
        out1=out1+x
        out2=F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(out1))))))
        out2=out2+out1
        outn=F.relu(self.nbn(self.nconv(out2)))
        return out2,outn

class featurenet(nn.Module):
    def __init__(self,Block,planes=[64,128,256]):
        super(featurenet, self).__init__()
        self.inconv=conv1x1(3, planes[0])
        self.inbn=nn.BatchNorm2d(planes[0])  
        self.layer1=Block(planes[0],planes[1])
        self.pool1=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer2=Block(planes[1],planes[2])
        self.pool2=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer3=Block(planes[2],planes[2])

        fin=planes[0]+planes[1]+planes[2]
        self.fconv=nn.Conv2d(fin, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fbn=nn.BatchNorm2d(128)

    def forward(self,x):
        N,C,H,W=x.shape
        out1=F.relu(self.inbn(self.inconv(x)))
        out1,out1n=self.layer1(out1)
        out2,out2n=self.layer2(self.pool1(out1n))
        out3,out3n=self.layer3(self.pool1(out2n))

        out2u=F.upsample(out2,size=(H,W),mode='bilinear',align_corners=True)
        out3u=F.upsample(out3,size=(H,W),mode='bilinear',align_corners=True)
        conFx=torch.cat((out1,out2u,out3u),dim=1)
        out=self.fbn(self.fconv(conFx))
        return out



