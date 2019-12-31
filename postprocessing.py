
from imguidedfilter import imguidedfilter
import numpy as np
import scipy.sparse.linalg as sla


class PCA_util():
    def __init__(self):
        pass

    def pca_feats(self,feats,pca_dim):
        feats=np.double(feats)
        h,w,d=feats.shape
        feats=np.reshape(feats,(h*w,d))
        featmean=np.mean(feats,axis=0)
        features= feats- np.ones((h*w,1)) * featmean
        covar= features.T @ features
        _,eigvecs= sla.eigs(covar,3,which='LM')
        pcafeat = features @ eigvecs
        simp= np.double(np.reshape(pcafeat,(h,w,pca_dim)))
        return simp

    def find_guidedfiltered_dom_feats(self,feats,image):

        feats[feats<-5]=-5
        feats[feats>5]=5
        fdim=feats.shape[-1]

        # maxfd= fdim-fdim%3
        # for i in range(0,maxfd,3):
        #     feats[:, :, i : i + 3] = imguidedfilter(feats[:, :, i : i + 3], image, (10, 10), 0.001)
        # for i in range(maxfd,fdim):
        #     feats[:, :, i] = imguidedfilter(feats[:, :, i], image, (10, 10), 0.001)

        simp=self.pca_feats(feats,pca_dim=3)
        simp_gui=np.zeros(simp.shape)
        for i in range(3):
            simp_gui[:, :, i] = imguidedfilter(simp[:, :, i], image, (10, 10), 0.001)
        for i in range(simp.shape[-1]):
            simp_gui[:, :, i] = simp_gui[:, :, i] - simp_gui[:, :, i].min()
            simp_gui[:, :, i] = simp_gui[:, :, i] / simp_gui[:, :, i].max()
        return simp_gui,simp, feats

