
import torch
import torch.nn as nn
import torch.nn.functional as F




#From SWAV
class MultiPrototypes(nn.Module):
    #I dont allow for variation of n_clusters in each prototype, as SWAV does
    def __init__(self, output_dim, n_classes, nmb_heads):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = nmb_heads
        for i in range(nmb_heads):
            self.n_layers = 0
            self.add_module("flatten" + str(i), nn.Flatten())
            self.add_module("batch_norm" + str(i), nn.BatchNorm1d(output_dim))
            #for j in range(0,3):
            #if output_dim <= n_classes*2.5:
            self.n_layers =  self.n_layers + 1
            self.add_module("prototypes" + str(i) + "_0", nn.Linear(output_dim, n_classes))
            #else:
            #    tmp = output_dim
            #    while tmp > n_classes*2.5:
            #        self.n_layers =  self.n_layers + 1
            #        self.add_module("prototypes" + str(i) + "_" + str(self.n_layers-1), nn.Linear(tmp, int(tmp/2)))
            #        tmp = int(tmp/2)
            #self.add_module("prototypes" + str(i) + "_0", nn.Linear(output_dim, output_dim*2)) 
            ##self.add_module("prototypes" + str(i) + "_1", nn.Linear(output_dim, n_classes))
            #self.add_module("prototypes" + str(i) + "_2", nn.Linear(n_classes*2, n_classes))
            self.n_layers =  self.n_layers + 1
            self.n_classes = n_classes
            self.add_module("prototypes" + str(i) + "_" + str(self.n_layers-1), nn.Softmax(dim=1)) #n_classes, n_classes, bias=False))
    
        self.initialize_weights()


    def initialize_weights(self):
        for m in self.modules():
          if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()


    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            x = getattr(self, "flatten" + str(i))(x)
            x = getattr(self, "batch_norm" + str(i))(x)
            for j in range(0,self.n_layers):
                x = getattr(self, "prototypes" + str(i) + "_" + str(j))(x)
            out.append(x)
        return out




#From SWAV
class DeepMultiPrototypes(nn.Module):
    #I dont allow for variation of n_clusters in each prototype, as SWAV does
    def __init__(self, output_dim, n_classes, nmb_heads):
        super(DeepMultiPrototypes, self).__init__()
        self.nmb_heads = nmb_heads
        for i in range(nmb_heads):
            self.n_layers = 0
            self.add_module("flatten" + str(i), nn.Flatten())
            od = output_dim
            print(od)
            j = 0
            while od < 2000:
                self.n_layers =  self.n_layers + 1
                self.add_module("prototypes" + str(i) + "_" + str(j), nn.Linear(od, od*2))
                self.add_module("prototypes_act" + str(i) + "_" + str(j), nn.SELU()) #LeakyReLU(0.1))
                od = od*2
                j = j + 1
            self.add_module("batch_norm", nn.BatchNorm1d(od))
            #self.n_layers =  self.n_layers + 1
            #print(od)
            #self.add_module("prototypes" + str(i) + "_" + str(j), nn.Linear(od, int(od/2)))
            #self.add_module("prototypes_act" + str(i) + "_" + str(j), nn.SELU()) #LeakyReLU(0.1))
            #od = int(od/2)
            #j = j + 1            
            #self.n_layers =  self.n_layers + 1
            #print(od)
            #self.add_module("prototypes" + str(i) + "_" + str(j), nn.Linear(od, int(od/2)))
            #self.add_module("prototypes_act" + str(i) + "_" + str(j), nn.SELU()) #LeakyReLU(0.1))
            #od = int(od/2)
            #j = j + 1
            self.n_layers =  self.n_layers + 1
            self.add_module("prototypes" + str(i) + "_" + str(j), nn.Linear(od, n_classes))
            self.add_module("prototypes_act" + str(i) + "_" + str(j), nn.Softmax(dim=1))
            self.n_classes = n_classes

        self.initialize_weights()


    def initialize_weights(self):
        for m in self.modules():
          if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()


    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            x = getattr(self, "flatten" + str(i))(x)
            x = getattr(self, "batch_norm")(x)
            for j in range(0,self.n_layers-1):
                x = getattr(self, "prototypes" + str(i) + "_" + str(j))(x)
                x = getattr(self, "prototypes_act" + str(i) + "_" + str(j))(x)
            x = getattr(self, "batch_norm")(x)
            x = getattr(self, "prototypes" + str(i) + "_" + str(self.n_layers-1))(x)
            x = getattr(self, "prototypes_act" + str(i) + "_" + str(self.n_layers-1))(x)
            out.append(x)
        return out



class DeepConvMultiPrototypes(nn.Module):
    #I dont allow for variation of n_clusters in each prototype, as SWAV does
    def __init__(self, output_dim, n_classes, nmb_heads):
        super(DeepConvMultiPrototypes, self).__init__()
        self.nmb_heads = nmb_heads
        for i in range(nmb_heads):
            self.n_layers = 0
            #self.add_module("flatten" + str(i), nn.Flatten())
            od = output_dim
            print(od)
            j = 0
            #while od < 5000:
            self.n_layers =  self.n_layers + 1
            od1 = od*2
            self.add_module("prototypes" + str(i) + "_" + str(j), nn.Conv2d(od, od1, kernel_size=2,stride=1,padding=1))
            self.add_module("prototypes_act" + str(i) + "_" + str(j), nn.LeakyReLU(0.1))
            j = j + 1
            #while od > 1000:
            self.n_layers =  self.n_layers + 1
            print(od1)
            od2 = int(od1*2)
            self.add_module("prototypes" + str(i) + "_" + str(j), nn.Conv2d(od1, od2, kernel_size=2,stride=1,padding=1))
            self.add_module("prototypes_act" + str(i) + "_" + str(j), nn.LeakyReLU(0.1))
            j = j + 1

            self.add_module("batch_norm", nn.BatchNorm2d(od2))

            self.n_layers =  self.n_layers + 1
            self.add_module("prototypes" + str(i) + "_" + str(j), nn.Conv2d(od2, n_classes, kernel_size=2,stride=1,padding=1))
            self.add_module("prototypes_act" + str(i) + "_" + str(j), nn.Softmax2d()) #n_classes, n_classes, bias=False))

        self.initialize_weights()


    def initialize_weights(self):
        for m in self.modules():
          if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()


    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            for j in range(0,self.n_layers-1):
                x = getattr(self, "prototypes" + str(i) + "_" + str(j))(x)
                x = getattr(self, "prototypes_act" + str(i) + "_" + str(j))(x)
            x = getattr(self, "batch_norm")(x)
            x = getattr(self, "prototypes" + str(i) + "_" + str(self.n_layers-1))(x)
            x = getattr(self, "prototypes_act"  + str(i) + "_" + str(self.n_layers-1))(x)
            out.append(x)
        return out


