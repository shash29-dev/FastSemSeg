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


class trainer():
    def __init__(self):
        self.net=featurenet(Block,planes=[64,128,256]).cuda()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),lr=0.0001)
        self.epochs=40
        self.num_spix=500
        self.batch_size=5
        self.loss_util=LossFuncs.ComputeLoss()
        self.pca_util=postprocessing.PCA_util()
        self.data_util=DataProviderUtil(num_spix=self.num_spix,batch_size=self.batch_size)
        self.pix_to_spix=pix_2_spix.pix_to_superpix(num_spix=self.num_spix)
        if not os.path.exists('./Results'): os.mkdir('./Results')
        if not os.path.exists('./saved_model'): os.mkdir('./saved_model')
    
    def start_train(self,save_results=False):
        ep=self.load_saved_model()
        for epoch in range(ep,self.epochs):
            running_loss=0
            for i,sample in enumerate(self.data_util.getData['train']):
                t0=time.time()
                image=sample['image'].cuda()
                gt_mask=sample['gt'].cuda()
                segments=sample['segments'].cuda()
                outFeats=self.net(image)
                Avg_feats,spix_gt=self.pix_to_spix.get_spix_data(gt_mask,segments,outFeats)
                Loss=self.loss_util.npair_loss(Avg_feats,spix_gt)
                self.optimizer.zero_grad()
                Loss.backward()
                self.optimizer.step()
                if i%5==0:
                    print('Epoch: {}\t TotalLoss: {:5f}\t Loss: {:5f}\t Iter: {}/{}\t Time per Iter:{:5f} s'.format(epoch,running_loss,Loss,i,int(len(self.data_util.getData['train'])),time.time()-t0))
                running_loss+=Loss
                if save_results:
                    self.save_figures(outFeats,i,sample,epoch)
            torch.save({
                       'model_state_dict': self.net.state_dict(),
                       'optimizer_state_dict': self.optimizer.state_dict(),
                       'epoch': epoch,
                   }, './saved_model/LatestSavednet_epoch_'+str(epoch)+'.pth')

    def load_saved_model(self):
        try:
            checkpoint = torch.load('./saved_model/LatestSavednet.pth')
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            ep=checkpoint['epoch']+1
            print('loaded Latest Model and starting epoch:', ep)
        except Exception as e:
            print('No saved Model found... starting Training')
            ep=0
        return ep

    def save_figures(self,outFeats,iteration,sample,epoch):
        if iteration%10==0:
            simp_gui,_,_=self.pca_util.find_guidedfiltered_dom_feats(outFeats[0].permute(1,2,0).cpu().detach().numpy(),sample['image'][0].permute(1,2,0).numpy())
            plt.figure(figsize=(18,9))
            plt.subplot(1,3,1)
            plt.imshow(sample['image'][0].permute(1,2,0).numpy())
            plt.axis('off')
            plt.title('Original Image')

            plt.subplot(1,3,2)
            plt.imshow(simp_gui)
            plt.axis('off')
            plt.title('Guided Filtered Dominant Features')

            plt.subplot(1,3,3)
            plt.imshow(sample['gt'][0].numpy())
            plt.axis('off')
            plt.title('Ground Truth')
            
            plt.savefig('./Results/Fig'+str(epoch)+'_'+str(iteration)+'.png')

if __name__ == '__main__':

    inst=trainer()
    inst.start_train(save_results=True)
