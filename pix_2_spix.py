import torch
import pdb

class pix_to_superpix():
    def __init__(self,num_spix):
        self.num_spix=num_spix

    def get_spix_data(self,gt_mask,segments,outFeats):
        self.gt_mask=gt_mask
        self.segments=segments
        self.outFeats=outFeats
        Avg_feats=self.get_PixSpix_avg()
        spix_gt=self.get_spix_gt()
        return Avg_feats,spix_gt
        
    def get_spix_gt(self):
        bs,H,W=self.gt_mask.shape
        feats_gt=torch.zeros((bs,self.num_spix)).cuda()
        gtm=self.gt_mask+1
        for i in range(self.num_spix):
            s0=self.segments==i
            x=gtm*s0
            sp_label=[]
            for tb in range(bs):
                if i<=self.segments[tb].max():
                    val,ind=torch.mode(x[tb][x[tb]!=0])
                    val=val-1
                    sp_label.append(int(val.cpu().numpy()))
                else:
                    sp_label.append(-1)
            feats_gt[:,i]=torch.Tensor(sp_label)
        return feats_gt
            

    def get_PixSpix_avg(self):
        bs,C,H,W=self.outFeats.shape
        sp_feats=torch.zeros((bs,self.num_spix,C)).cuda()
        
        for i in range (self.num_spix):
            s0=self.segments==i
            ex_dim_s0=s0[:,None,:,:]
            mask_nums=s0.sum(axis=1).sum(axis=1)
            mask_nums[mask_nums==0]=1
            mask_nums=mask_nums[:,None]
            masked=ex_dim_s0*self.outFeats
            sum_sup_feats=masked.sum(axis=2).sum(axis=2)
            avg_sup_feats=sum_sup_feats/mask_nums
            sp_feats[:,i,:]=avg_sup_feats
        return sp_feats
        