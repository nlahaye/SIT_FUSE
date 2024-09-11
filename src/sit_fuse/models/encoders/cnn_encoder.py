
import torch
import torch.nn as nn
import torch.nn.functional as F




class DeepConvEncoder(nn.Module):
    #I dont allow for variation of n_clusters in each prototype, as SWAV does
    def __init__(self, in_chans, flatten=False):
        super(DeepConvEncoder, self).__init__()
        self.n_layers = 0
        self.flatten = flatten
        #self.add_module("flatten" + str(i), nn.Flatten())
        od = in_chans
        self.in_chans = in_chans
        print(od)
        i = 0
        j = 0
        self.add_module("batch_norm", nn.BatchNorm2d(od))
        #while od < 5000:
        self.n_layers =  self.n_layers + 1
        od1 = od*2
        self.add_module("prototypes" + str(i) + "_" + str(j), nn.Conv2d(od, od1, kernel_size=3,stride=1,padding=1))
        self.add_module("prototypes_act" + str(i) + "_" + str(j), nn.LeakyReLU(0.1, inplace=True))
        j = j + 1
        #while od > 1000:
        self.n_layers =  self.n_layers + 1
        #print(od1)
        od2 = int(od1*2)
        self.add_module("prototypes" + str(i) + "_" + str(j), nn.Conv2d(od1, od2, kernel_size=3,stride=1,padding=1))
        self.add_module("prototypes_act" + str(i) + "_" + str(j), nn.LeakyReLU(0.1, inplace=True))
        j = j + 1

        #od2 = od1
        od3 = int(od2*2)
        self.n_layers =  self.n_layers + 1
        self.add_module("prototypes" + str(i) + "_" + str(j), nn.Conv2d(od2, od3, kernel_size=3,stride=1,padding=1))
        self.add_module("prototypes_act" + str(i) + "_" + str(j), nn.LeakyReLU(0.1, inplace=True)) #n_classes, n_classes, bias=False))

        print(od, od1, od2, od3)


        if self.flatten:
            self.add_module("flatten_layer", nn.Flatten())

        self.initialize_weights()


    def get_output_shape(self, image_dim):
        with torch.no_grad():
            tmp = torch.rand(*(image_dim)).to(next(self.parameters()).device)
            return self.forward(tmp).data.shape


    def initialize_weights(self):
        for m in self.modules():
          if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()


    def forward(self, x):
        i = 0
        x = getattr(self, "batch_norm")(x)
        for j in range(0,self.n_layers):
            x = getattr(self, "prototypes" + str(i) + "_" + str(j))(x)
            x = getattr(self, "prototypes_act" + str(i) + "_" + str(j))(x)
        if self.flatten:
            x = getattr(self, "flatten_layer")(x)
        return x


