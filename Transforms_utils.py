import config
import pdb
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
from skimage import io,transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import  utils
from torchvision.transforms import functional as F
from skimage.color import rgb2lab
import random
from PIL import Image
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import color
from skimage.color import rgb2lab


class RandomHorizontalFlip(object):
    
    def __init__(self, p=0.4):
        self.p = p
    def __call__(self, sample):
        if random.random() < self.p:
        	sample['image']=F.hflip(sample['image'])
        	sample['gt'] = F.hflip(sample['gt'])
        return sample

class RandomVerticalFlip(object):    
    def __init__(self, p=0.4):
        self.p = p
    def __call__(self, sample):
        if random.random() < self.p:
            sample['image']=F.vflip(sample['image'])
            sample['gt']= F.vflip(sample['gt'])
        return sample

class PILImageToNumpyArray(object):
    def __init__(self):
        pass
    def __call__(self,sample):
        sample['image']=np.array(sample['image'])
        sample['gt']= np.array(sample['gt'])
        return sample
        
class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']
        h, w = image.size[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        sample['image']=image.resize((new_h,new_w),Image.BILINEAR)
        sample['gt']=gt.resize((new_h,new_w),Image.BILINEAR)
        return sample

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']

        h, w = image.size[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        sample['image']=image.crop((top,left,top+new_h,left+new_w))
        sample['gt']=gt.crop((top,left,top+new_h,left+new_w))
        return sample
        
class imagespixels(object):
    def __init__(self,n_segments=100,sigma=2):
        self.n_segments=n_segments
        self.sigma=sigma
    def __call__(self,sample):
        inImage=sample['image']
        segments=slic(inImage,n_segments=self.n_segments,sigma=self.sigma)
        sample['segments']=segments
        return sample


class ToTensor(object):
    def __init__(self):
        pass
        
    def __call__(self, sample):
        sample['image']=F.to_tensor(np.array(sample['image']))
        sample['gt']=torch.from_numpy(np.array(sample['gt']))
        sample['segments']=torch.from_numpy(np.array(sample['segments']))

        return sample

class transformImage(object):
    def __init__(self,sample):
        self.sample=sample
    
    def __call__(self,transformList):
        for tsfrm in transformList:
            self.sample=tsfrm(self.sample) 
        return self.sample


if __name__== '__main__':
    
    image_name=config.data_ade.trainData+'/ADE_train_00000002.jpg'
    gt_pix_map= config.data_ade.train_pixel_map+'/ADE_train_00000002.png'

    image=Image.open(image_name).convert('RGB')
    gt_pix_Labels=Image.open(gt_pix_map)
    assert(gt_pix_Labels.mode == "L")
    assert(image.size[0] == gt_pix_Labels.size[0])
    assert(image.size[1] == gt_pix_Labels.size[1])


    sample={'image':image, 'gt':gt_pix_Labels, 'segments': None}
    trans= transformImage(sample)

    transformList=[
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                Rescale(256),
                RandomCrop(224),
                imagespixels(n_segments=500,sigma=2),
                ToTensor()
                ]

    sample=trans(transformList)
    x=sample['image']
    g=sample['gt']
    segments=sample['segments']
    # plt.imshow(mark_boundaries(x.numpy().transpose(1,2,0),segments.numpy()))
    # plt.show()
    pdb.set_trace()
    
    