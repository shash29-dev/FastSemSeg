from torch.utils.data import Dataset, DataLoader
import Transforms_utils
import config

import warnings
warnings.filterwarnings('ignore')
import os
import scipy
from scipy import io
import skimage
import matplotlib.pyplot as plt
import pdb
from skimage.transform import resize
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import color
import pdb
import PIL
from sklearn import cluster
import pdb
import glob

class DataProvider(Dataset):
    def __init__(self,transformList=None,data='val'):
        
        self.data=data
        if data=='train':
            self.image_paths=config.data_ade.trainData
            self.gt_map_paths=config.data_ade.train_pixel_map
        if data=='val':
            self.image_paths=config.data_ade.valData
            self.gt_map_paths=config.data_ade.val_pixel_map

        self.image_name_list=glob.glob(self.image_paths+'/*.jpg')
        self.transformList=transformList

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,idx):
        im_name=self.image_name_list[idx]
        gt_name=self.gt_map_paths+'/'+im_name.split(os.sep)[-1][:-3]+'png'        
        image=PIL.Image.open(im_name).convert('RGB')
        gt=PIL.Image.open(gt_name)
        sample={'image':image,'gt':gt,'segments':None}

        if self.transformList:
             trans= Transforms_utils.transformImage(sample)
             sample=trans(self.transformList)
        return sample



class DataProviderUtil():
    def __init__(self, num_spix=500,batch_size=3):
        self.transformList=[
                        Transforms_utils.RandomHorizontalFlip(p=0.5),
                        Transforms_utils.RandomVerticalFlip(p=0.5),
                        Transforms_utils.Rescale(256),
                        Transforms_utils.RandomCrop(224),
                        Transforms_utils.imagespixels(n_segments=num_spix,sigma=2),
                        Transforms_utils.ToTensor()
                        ]

        self.getData={'train': DataLoader(DataProvider(transformList=self.transformList,data='train'), batch_size=batch_size, shuffle=True, num_workers=8,pin_memory=True),
                'val': DataLoader(DataProvider(transformList=self.transformList,data='val'), batch_size= batch_size ,shuffle=True, num_workers=8,pin_memory=True)}

if __name__=='__main__':            
          
    for i,sample in enumerate(getData['train']):
        a=sample['image'][0]
        b=sample['gt'][0] 
        segments=sample['segments'][0] 
        plt.subplot(1,3,1)    
        plt.imshow(a.numpy().transpose(1,2,0))  
        plt.subplot(1,3,2)    
        plt.imshow(b)  
        plt.subplot(1,3,3)    
        plt.imshow(mark_boundaries(a.numpy().transpose(1,2,0),segments.numpy()))
        plt.show()
