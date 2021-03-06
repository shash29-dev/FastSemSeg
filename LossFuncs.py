import torch
import pdb
class ComputeLoss():
    def __init__(self):
        pass

    def npair_loss(self,Avg_feats,spix_gt):
        bs,num_spix,C=Avg_feats.shape
        diff=Avg_feats[:,:,None]-Avg_feats[:,None]
        featDist=(diff**2).sum(axis=-1)

        Indicator_mat=spix_gt[:,:,None]-spix_gt[:,None]
        Indicator_mat[Indicator_mat!=0]=1

        same_label_pair_indicator=Indicator_mat==0
        diff_label_pair_indicator=Indicator_mat==1

        # soft_plus_func=torch.nn.Softplus(beta=1,threshold=50)
        # similarity_loss=soft_plus_func(featDist)*same_label_pair_indicator
        # dissimilarity_loss=soft_plus_func(-featDist)*diff_label_pair_indicator
        
        featDist=torch.clamp(featDist,0,50)
        similarity_loss= torch.mean(same_label_pair_indicator*torch.log((1+torch.exp(featDist))/2))
        dissimilarity_loss= torch.mean(diff_label_pair_indicator*torch.log(1+torch.exp(-featDist)/2))
 
        # similarity_loss=torch.mean(torch.logsumexp(featDist*same_label_pair_indicator,dim=1))
        # dissimilarity_loss=torch.mean(torch.logsumexp(-featDist*diff_label_pair_indicator,dim=1))
        
        total_loss=similarity_loss+dissimilarity_loss
        return total_loss
