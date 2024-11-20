
import torch.nn as nn

from sklearn.decomposition import PCA


class PCAEncoder(nn.Module):
    #I dont allow for variation of n_clusters in each prototype, as SWAV does
    def __init__(self):
        super(PCAEncoder, self).__init__()
 
        self.pca = PCA(n_components=0.99, svd_solver = 'full')


    def get_output_shape(self, image_dim):
        return (image_dim[0], self.pca.vccomponents_.shape[0])



    def forward(self, x):
        return self.pca.transform(x)


