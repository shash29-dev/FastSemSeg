import torch
import torch.nn as nn
import pdb
from PIL import Image
import matplotlib.pyplot as plt
from DataProvider import DataProviderUtil
from model import featurenet,Block
import time
import numpy as np
import postprocessing, LossFuncs,pix_2_spix
import os 
from imguidedfilter import imguidedfilter
from torchvision.transforms import functional as F
import sys

class demo():
    def __init__(self,image_name='table.png'):
        self.net=featurenet(Block,planes=[64,128,256]).cuda()
        self.pca_util=postprocessing.PCA_util()
        self.image_name=image_name
        self.demo_im_root='./demo_im/'
        if not os.path.exists('./Results_demo'): os.mkdir('./Results_demo')
    
    def start_demo(self,save_results=False):
        _=self.load_saved_model()  
        image=Image.open(self.demo_im_root+self.image_name).convert('RGB')
        image=F.to_tensor(np.array(image))
        image=image[None,:,:,:].cuda()
        outFeats=self.net(image)
        if save_results:
            self.save_figures(outFeats,image)
            
    def load_saved_model(self):
        try:
            checkpoint = torch.load('./saved_model/LatestSavednet.pth')
            self.net.load_state_dict(checkpoint['model_state_dict'])
            ep=checkpoint['epoch']+1
            print('loaded Latest Model and starting epoch:', ep)
        except Exception as e:
            print('No saved Model found... starting Training')
            ep=0
        return ep

    def save_figures(self,outFeats,image):
        simp_gui,simp_ung,_=self.pca_util.find_guidedfiltered_dom_feats(outFeats[0].permute(1,2,0).cpu().detach().numpy(),image[0].cpu().permute(1,2,0).numpy())
        plt.figure(figsize=(18,9))
        plt.subplot(1,2,1)
        plt.imshow(image[0].cpu().permute(1,2,0).numpy())
        plt.axis('off')
        plt.title('Original Image')

        plt.subplot(1,2,2)
        plt.imshow(simp_gui)
        plt.axis('off')
        plt.title('Guided Filtered Dominant Features')

        plt.savefig('./Results_demo/'+self.image_name)
        print('Result Saved...')

if __name__ == '__main__':
    image_name=sys.argv[1]
    inst=demo(image_name=image_name)
    inst.start_demo(save_results=True)
