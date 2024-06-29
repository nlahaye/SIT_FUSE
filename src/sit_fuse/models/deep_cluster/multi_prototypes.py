
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

class Up_Linear(nn.Module):
    def __init__(self, in_ch, size, coef=1):
        super(Up_Linear, self).__init__()
        #self.insrm = nn.InstanceNorm2d(1024)
        self.shuffle = nn.PixelShuffle(upscale_factor=4)

        n_ch = int(coef * in_ch)
        
        self.ln = nn.Sequential(
            nn.Linear(in_ch, n_ch),
            nn.SELU(inplace=True),
            nn.Linear(n_ch, n_ch * 4),
            nn.SELU(inplace=True),
            nn.Linear(n_ch*4, n_ch * 16),
            nn.SELU(inplace=True),
            #nn.Linear(n_ch*16, n_ch * 32),
        )
        
        self.size = size

    def forward(self, x):
        print(x.shape, x.min(), x.max(), x.mean(), x.std())
        #x = self.insrm(x)
        x = self.ln(x)
        print(x.shape, x.min(), x.max(), x.mean(), x.std())
        x = x.permute(0, 2, 1)
        print(x.shape, x.min(), x.max(), x.mean(), x.std())
        x = torch.reshape(x, (x.shape[0], x.shape[1], self.size, self.size))
        print(x.shape, x.min(), x.max(), x.mean(), x.std())
        x = self.shuffle(x)
        print(x.shape, x.min(), x.max(), x.mean(), x.std())
        x = torch.reshape(x, (x.shape[0], x.shape[1], self.size*self.size*16))
        print(x.shape, x.min(), x.max(), x.mean(), x.std())
        x = x.permute(0, 2, 1)
        return x

class JEPA_Seg(nn.Module):
    def __init__(self):
        super(JEPA_Seg, self).__init__()

        self.insrm = nn.InstanceNorm2d(2048)
        self.ups3 = Up_Linear(2048, 16, 1)
        #self.ups2 = Up_Linear(512, 28, 1)
        #self.ups1 = Up_Linear(2048, 64, 2)
        #self.ups0 = Up_Linear(128, 112, 3)

 
        self.shuffle = nn.PixelShuffle(upscale_factor=2)

        self.out = nn.Sequential(
            #nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #nn.LeakyReLU(0.1), #nn.SELU(), #nn.LeakyReLU(0.1),
            #nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            #nn.LeakyReLU(0.1), #nn.SELU(), #nn.LeakyReLU(0.1),
            #nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            #nn.SELU(inplace=True), #nn.LeakyReLU(0.1),
            #nn.InstanceNorm2d(256),
            nn.Conv2d(2048, 500, kernel_size=1, stride=1),
        )
        self.smax = nn.Softmax(dim=1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
          if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='selu')
            m.bias.data.zero_()


    def forward(self, x):
        print("FIRST")
        x = self.insrm(x)
        x = self.ups3(x)
        #x = self.ups2(x)
        #x = self.ups1(x)
        #x = self.ups0(x)

        x = x.permute(0, 2, 1)
        x = torch.reshape(x, (x.shape[0], x.shape[1], 64, 64))
        #x = self.shuffle(x)
        print(x.shape, x.min(), x.max(), x.mean(), x.std())
        #x = transforms.Resize((224, 224))(x)
        print(x.shape, x.min(), x.max(), x.mean(), x.std())

        x = self.out(x)
        print(x.shape, x.min(), x.max(), x.mean(), x.std())
        x = self.smax(x)
        print(x.shape, x.min(), x.max(), x.mean(), x.std(), "FINAL LAYER")
        return x



class OutputProjection(nn.Module):
    def __init__(self, image_size, patch_size, embed_size, output_dims):
        super().__init__()
        self.patch_size = patch_size
        self.output_dims = output_dims
        self.add_module("batch_norm", nn.LayerNorm(embed_size))
        self.add_module("projection", nn.Linear(embed_size, patch_size * patch_size * output_dims))
        self.add_module("fold", nn.Fold(output_size=(image_size, image_size), kernel_size=patch_size, stride=patch_size))
        self.add_module("flatten", nn.Flatten())
        self.add_module("softmax", nn.Softmax(dim=1)) #n_classes, n_classes, bias=False))

        self.initialize_weights()



    def initialize_weights(self):
        for m in self.modules():
          if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()


    
    def forward(self, x):
        B, T, C = x.shape
        x = getattr(self, "batch_norm")(x)
        x = getattr(self, "projection")(x)
        x = x.permute(0, 2, 1)
        x = getattr(self, "fold")(x)
        #x = getattr(self, "flatten")(x)
        x = getattr(self, "softmax")(x) 

        return x


#From SWAV
class MultiPrototypes(nn.Module):
    #I dont allow for variation of n_clusters in each prototype, as SWAV does
    def __init__(self, output_dim, n_classes, nmb_heads):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = nmb_heads
        self.output_dim = output_dim
        for i in range(nmb_heads):
            self.n_layers = 0
            self.add_module("flatten" + str(i), nn.Flatten())
            #self.add_module("batch_norm" + str(i), nn.LayerNorm(output_dim))
            #for j in range(0,3):
            #if output_dim <= n_classes*2.5:
            self.n_layers =  self.n_layers + 1
            self.add_module("prototypes" + str(i) + "_0", nn.Linear(output_dim, n_classes))
            self.add_module("activ" + str(i) + "_0", nn.SELU())
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
            print(x.shape)
            x = getattr(self, "flatten" + str(i))(x)
            print(x.shape)
            #x = getattr(self, "batch_norm" + str(i))(x)
            print(x.shape)
            for j in range(0,self.n_layers):
                x = getattr(self, "prototypes" + str(i) + "_" + str(j))(x)
                if  j < self.n_layers - 1:
                    x = getattr(self, "activ" + str(i) + "_" + str(j))(x)
            out.append(x)
        print(len(out), x.shape)
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
            self.add_module("prototypes" + str(i) + "_" + str(j), nn.Conv2d(od, od1, kernel_size=3,stride=1,padding=1))
            self.add_module("prototypes_act" + str(i) + "_" + str(j), nn.LeakyReLU(0.1))
            j = j + 1
            #while od > 1000:
            self.n_layers =  self.n_layers + 1
            print(od1)
            od2 = int(od1*2)
            self.add_module("prototypes" + str(i) + "_" + str(j), nn.Conv2d(od1, od2, kernel_size=3,stride=1,padding=1))
            self.add_module("prototypes_act" + str(i) + "_" + str(j), nn.LeakyReLU(0.1))
            j = j + 1

            self.add_module("batch_norm", nn.BatchNorm2d(od2))

            self.n_layers =  self.n_layers + 1
            self.add_module("prototypes" + str(i) + "_" + str(j), nn.Conv2d(od2, n_classes, kernel_size=3,stride=1,padding=1))
            self.add_module("prototypes_act" + str(i) + "_" + str(j), nn.Softmax(dim=1)) #n_classes, n_classes, bias=False))

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
                print(x.shape)
                x = getattr(self, "prototypes" + str(i) + "_" + str(j))(x)
                print(x.shape)
                x = getattr(self, "prototypes_act" + str(i) + "_" + str(j))(x)
                print(x.shape)
            x = getattr(self, "batch_norm")(x)
            print(x.shape)
            x = getattr(self, "prototypes" + str(i) + "_" + str(self.n_layers-1))(x)
            print(x.shape)
            x = getattr(self, "prototypes_act"  + str(i) + "_" + str(self.n_layers-1))(x)
            print(x.shape)
            out.append(x)
        return out


