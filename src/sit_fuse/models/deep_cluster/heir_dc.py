import numpy as np
import pytorch_lightning as pl

import copy

import torch.nn as nn
import torch

from sit_fuse.losses.iid import IID_loss


from sit_fuse.models.deep_cluster.pca_dc import PCA_DC
from sit_fuse.models.deep_cluster.byol_dc import BYOL_DC
from sit_fuse.models.deep_cluster.ijepa_dc import IJEPA_DC
from sit_fuse.models.deep_cluster.dbn_dc import DBN_DC
from sit_fuse.models.deep_cluster.cdbn_dc import CDBN_DC
from sit_fuse.models.deep_cluster.clay_dc import Clay_DC
from sit_fuse.models.deep_cluster.mae_dc import MAE_DC
from sit_fuse.models.deep_cluster.dc import DeepCluster
from sit_fuse.models.deep_cluster.multi_prototypes import MultiPrototypes, OutputProjection, JEPA_Seg
from sit_fuse.datasets.sf_dataset import SFDataset
from sit_fuse.datasets.sf_dataset_conv import SFDatasetConv
from sit_fuse.utils import read_yaml, get_output_shape

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import uuid

#TODO add back in multi-layer heir
class Heir_DC(pl.LightningModule):
    #take pretrained model path, number of classes, learning rate, weight decay, and drop path as input
    def __init__(self, data, pretrained_model_path, num_classes, yml_conf, lr=1e-3, weight_decay=0, encoder_type=None, number_heads=1, min_samples=1000, encoder=None, clust_tree_ckpt = None, generate_labels = True):

        super().__init__()
        self.save_hyperparameters(ignore=['data'])

        self.min_samples = min_samples

        #set parameters
        self.lr = lr
        self.weight_decay = weight_decay

        self.key = -1


        self.number_heads = number_heads
        #define model layers
        self.pretrained_model = None
        if encoder_type is None:
            self.pretrained_model = DeepCluster.load_from_checkpoint(pretrained_model_path, img_size=int(yml_conf["data"]["tile_size"][0]*yml_conf["data"]["tile_size"][1]), in_chans=yml_conf["data"]["tile_size"][2]) #Why arent these being saved
        elif "pca" in encoder_type:
            self.pretrained_model = PCA_DC.load_from_checkpoint(pretrained_model_path, pretrained_model=encoder)
            self.encoder = encoder
            for param in self.pretrained_model.mlp_head.parameters():
                param.requires_grad = False
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        elif "dbn" in encoder_type:
            if encoder_type == "conv_dbn":
                self.pretrained_model = CDBN_DC.load_from_checkpoint(pretrained_model_path, pretrained_model=encoder)
                self.encoder = self.pretrained_model.segmentor.encoder
                self.pretrained_model.eval()
                self.pretrained_model.pretrained_model.eval()
                self.pretrained_model.mlp_head.eval()
            else:
                self.pretrained_model = DBN_DC.load_from_checkpoint(pretrained_model_path, pretrained_model=encoder)
                self.encoder = encoder
                self.pretrained_model.eval()
                self.pretrained_model.pretrained_model.eval()
                self.pretrained_model.mlp_head.eval()

            for param in self.pretrained_model.pretrained_model.parameters():
                param.requires_grad = False
            for param in self.pretrained_model.mlp_head.parameters():
                param.requires_grad = False
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        elif encoder_type == "ijepa":
            self.pretrained_model = IJEPA_DC.load_from_checkpoint(pretrained_model_path)
            self.pretrained_model.pretrained_model.model.mode = "test"
            self.pretrained_model.eval()
            self.pretrained_model.pretrained_model.eval()
            self.pretrained_model.mlp_head.eval()
            self.pretrained_model.pretrained_model.model.eval()

            self.encoder = self.pretrained_model.segmentor.encoder
            for param in self.pretrained_model.pretrained_model.model.parameters():
                param.requires_grad = False
            for param in self.pretrained_model.pretrained_model.parameters():
                param.requires_grad = False
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

            #getattr(self.pretrained_model.mlp_head, "batch_norm0").track_running_stats = True
            #self.pretrained_model.pretrained_model.model.layer_dropout = 0.0
        elif encoder_type == "byol":
            self.encoder = encoder
            self.pretrained_model = BYOL_DC.load_from_checkpoint(pretrained_model_path)
            self.pretrained_model.eval()
            self.pretrained_model.pretrained_model.eval()

            for param in self.pretrained_model.pretrained_model.parameters():
                param.requires_grad = False
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        elif encoder_type == "clay":
            self.pretrained_model = Clay_DC.load_from_checkpoint(pretrained_model_path)
            self.encoder = self.pretrained_model.pretrained_model.encoder
        elif encoder_type == "mae":
            self.pretrained_model = MAE_DC.load_from_checkpoint(pretrained_model_path)
            self.encoder = self.pretrained_model.segmentor.encoder
        else: 
            self.pretrained_model = DeepCluster.load_from_checkpoint(pretrained_model_path) #Why arent these being saved

        self.encoder_type = encoder_type
        self.num_classes = num_classes       

        self.clust_tree = {"0": {"-1": self.pretrained_model}, "1": {}}

        #define loss
        self.criterion = IID_loss
        self.rng = np.random.default_rng(None)

        self.lab_full = {}

        self.module_list = nn.ModuleList([self.pretrained_model])

 
        self.clust_tree_ckpt = clust_tree_ckpt
        if clust_tree_ckpt is None and generate_labels: 
            self.generate_label_set(data)
        elif clust_tree_ckpt is not None:
            
            encoder_output_size = None
            if self.encoder_type is None:
                n_visible = yml_conf["data"]["tile_size"][0]*yml_conf["data"]["tile_size"][1]*yml_conf["data"]["tile_size"][2]
            elif self.encoder_type == "ijepa":
                #TODO encoder_output_size = get_output_shape(self.encoder, (1, yml_conf["data"]["tile_size"][2],self.pretrained_model.pretrained_model.img_size,self.pretrained_model.pretrained_model.img_size))
                #n_visible = encoder_output_size[2] #*encoder_output_size[2]
                n_visible = 512 #1024 # 2048
                #print(encoder_output_size, yml_conf["data"]["tile_size"], self.pretrained_model.pretrained_model.img_size)
            elif self.encoder_type == "clay":
                n_visible = 768
            elif self.encoder_type == "mae":
                n_visible = 1024
            elif self.encoder_type == "dbn":
                encoder_output_size = (1, self.pretrained_model.pretrained_model.models[-1].n_hidden)
                n_visible = encoder_output_size[1]
            elif self.encoder_type == "pca":
                encoder_output_size = (1, self.pretrained_model.pretrained_model.pca.components_.shape[0])
                n_visible = encoder_output_size[1]
            elif self.encoder_type == "conv_dbn":
                #encoder_output_size = get_output_shape(self.encoder, (1, yml_conf["data"]["tile_size"][2], yml_conf["data"]["tile_size"][0], yml_conf["data"]["tile_size"][1]))
                n_visible = 900 #encoder_output_size[1]
            elif self.encoder_type == "byol":
                encoder_output_size = get_output_shape(self.encoder, (1, yml_conf["data"]["tile_size"][2], yml_conf["data"]["tile_size"][0], yml_conf["data"]["tile_size"][1]))
                n_visible = encoder_output_size[1]

            #print(encoder_output_size, "HERE")

            state_dict = torch.load(clust_tree_ckpt)
            pytorch_total_params = sum(p.numel() for p in self.pretrained_model.pretrained_model.parameters()) + sum(p.numel() for p in self.pretrained_model.mlp_head.parameters())
            print("PARAMS 1", pytorch_total_params)
            self.clust_tree, self.lab_full = \
                load_model(self.clust_tree, n_visible, self, state_dict, self.pretrained_model.device, (self.encoder_type == "ijepa"), self.num_classes) 
                #list(self.pretrained_model.mlp_head.children())[1].num_features, self, state_dict)        
 
        del data

    def generate_label_set(self, data):
        count = 0
        self.lab_full = {}
        if isinstance(data, SFDatasetConv):
            batch_size = min(10, self.min_samples)
        else:
            batch_size = min(700, self.min_samples)

        output_sze = data.data_full.shape[0]

        test_loader = DataLoader(data, batch_size=batch_size, shuffle=False, \
        num_workers = 0, drop_last = True, pin_memory = True)
        ind = 0
        ind2 = 0

        for data2 in tqdm(test_loader):
            dat_dev = data2[0].to(device=self.pretrained_model.device, non_blocking=True, dtype=torch.float32)

            lab = self.pretrained_model.forward(dat_dev)
            if lab.ndim > 2: #or B,C,H,W
                lab = lab.flatten(start_dim=2).permute(0,2,1).flatten(start_dim=0, end_dim=1)
            lab = torch.argmax(lab, axis = 1)
            lab = lab.detach().cpu()
            dat_dev = dat_dev.detach().cpu()

            ind1 = ind2
            ind2 += dat_dev.shape[0]
            if ind2 > data.data_full.shape[0]:
                ind2 = data.data_full.shape[0]

            lab_unq = torch.unique(lab)
            for l in lab_unq:
                inds = torch.where(lab == l)
                key = str(l.detach().cpu().numpy())
                if key in self.lab_full.keys():
                    self.lab_full[key] = torch.cat((self.lab_full[key],(inds[0] + ind1)))
                else:
                    self.lab_full[key] = inds[0] + ind1

            ind = ind + 1
            count = count + 1
            del dat_dev

 
    def forward(self, x, perturb = False, train=False, return_embed=False):
        #TODO fix for multi-head

        tile_size = x.shape[0]
        if x.ndim > 2:
            tile_size = x.shape[2]

        dt = x.dtype
        y = self.pretrained_model.forward(x)
        #print(y.shape, "Y_SHAPE")
 
        #if hasattr(self.pretrained_model, 'pretrained_model'):
        #    #if hasattr(self.pretrained_model.pretrained_model, 'model'):
        #        x = self.pretrained_model.pretrained_model.model(x) #.flatten(start_dim=1)
        #    else:
        #        x = self.pretrained_model.pretrained_model.forward(x) #.flatten(start_dim=1) 
        #else:
        if hasattr(self, 'encoder') and self.encoder is not None:

            if self.encoder_type == "byol" and self.pretrained_model.model_type == "Unet":
                x, x1, x2, x3, x4 = self.encoder[0].full_forward(x)   
                x4 = F.interpolate(x4, size=x3.size()[2:], mode='bilinear', align_corners=True)
                x3 = F.interpolate(self.pretrained_model.pretrained_model.br5(x3 + x4), size=x2.size()[2:], mode='bilinear', align_corners=True)
                x2 = F.interpolate(self.pretrained_model.pretrained_model.br6(x2 + x3), size=x1.size()[2:], mode='bilinear', align_corners=True)
                x1 = F.interpolate(self.pretrained_model.pretrained_model.br7(x1 + x2), size=conv1_sz, mode='bilinear', align_corners=True)

                x = self.pretrained_model.pretrained_model.br9(F.interpolate(self.pretrained_model.pretrained_model.br8(x1), size=x.size()[2:], mode='bilinear', align_corners=True))
                if perturb:
                    x = x + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (x.shape))).type(x.dtype).to(x.device) 

            elif self.encoder_type == "byol" and self.pretrained_model.model_type == "GCN":
                x1, x2, x3, x4, conv1_sz = self.encoder.backbone(x)
                x1 = self.encoder.br1(self.encoder.gcn1(x1))
                x2 = self.encoder.br2(self.encoder.gcn2(x2))
                x3 = self.encoder.br3(self.encoder.gcn3(x3))
                x4 = self.encoder.br4(self.encoder.gcn4(x4))
            
                if self.encoder.use_deconv:
                    # Padding because when using deconv, if the size is odd, we'll have an alignment error
                    x4 = self.encoder.decon4(x4)
                    if x4.size() != x3.size(): x4 = self.encoder._pad(x4, x3)
                    x3 = self.encoder.decon3(self.encoder.br5(x3 + x4))
                    if x3.size() != x2.size(): x3 = self.encoder._pad(x3, x2)
                    x2 = self.encoder.decon2(self.encoder.br6(x2 + x3))
                    if x2.size() != x1.size(): x2 = self.encoder._pad(x2, x1)
                    x1 = self.encoder.decon1(self.encoder.br7(x1 + x2))
 
                    x = self.encoder.br9(self.encoder.decon5(self.encoder.br8(x1)))
                else:
                    x4 = F.interpolate(x4, size=x3.size()[2:], mode='bilinear', align_corners=True)
                    x3 = F.interpolate(self.encoder.br5(x3 + x4), size=x2.size()[2:], mode='bilinear', align_corners=True)
                    x2 = F.interpolate(self.encoder.br6(x2 + x3), size=x1.size()[2:], mode='bilinear', align_corners=True)
                    x1 = F.interpolate(self.encoder.br7(x1 + x2), size=conv1_sz, mode='bilinear', align_corners=True)

                    x = self.encoder.br9(F.interpolate(self.encoder.br8(x1), size=x.size()[2:], mode='bilinear', align_corners=True))
                x = self.encoder.final_conv(x)
                if perturb:
                    x = x + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (x.shape))).type(x.dtype).to(x.device)



            elif self.encoder_type == "pca":
                x = torch.from_numpy(self.pretrained_model.pretrained_model(x.cpu().numpy())).type(x.dtype).to(x.device)         
                if perturb:
                    x = x + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (x.shape))).type(x.dtype).to(x.device)
            elif self.encoder_type == "conv_dbn":
                x = self.encoder(x)
                for i in range(len(x)):

                    if perturb:
                        x[i] = x[i] + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (x[i].shape))).type(x[i].dtype).to(x[i].device)
                    x[i] = self.pretrained_model.segmentor.upsamples[i](x[i])

                x = torch.cat(x, dim=1)
                x = self.pretrained_model.segmentor.fusion(x)

                x = F.interpolate(
                    x,
                    size=(tile_size, tile_size),
                    mode="bilinear",
                    align_corners=False,
                )  # Resize to match labels size
            elif self.encoder_type == "clay":
                dat_final = {
                        "pixels": x, #[0],
                        "latlon": torch.zeros((y.shape[0],4)),
                        "time": torch.zeros((y.shape[0],4)),
                        "gsd": self.pretrained_model.gsd,
                        "waves": self.pretrained_model.waves}
                x = self.encoder(dat_final)

                mn_tile_size = 99999
                mx_tile_size = -1
                for i in range(len(x)):
                
                    if perturb:
                        x[i] = x[i] + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (x[i].shape))).type(x[i].dtype).to(x[i].device)
                    x[i] = self.pretrained_model.pretrained_model.upsamples[i](x[i])
                    mn_tile_size = min(mn_tile_size, x[i].shape[-1])
                    mx_tile_size = max(mx_tile_size, x[i].shape[-1])
 
                if mx_tile_size > mn_tile_size:
                    for i in range(len(x)):
                        if x[i].shape[-1] < mx_tile_size:
                            x[i] = F.interpolate(
                                       x[i],
                                       size=(mx_tile_size, mx_tile_size),
                                       mode="bilinear",
                                       align_corners=False,
                                   )

 
                x = torch.cat(x, dim=1)
                x = self.pretrained_model.pretrained_model.fusion(x)

                x = F.interpolate(
                    x,
                    size=(tile_size, tile_size),
                    mode="bilinear",
                    align_corners=False,
                )  # Resize to match labels size

            elif self.encoder_type == "ijepa":
                x = self.encoder(x)
                for i in range(len(x)):
                    if perturb:
                        x[i] = x[i] + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (x[i].shape))).type(x[i].dtype).to(x[i].device)
                    x[i] = self.pretrained_model.segmentor.upsamples[i](x[i])

                x = torch.cat(x, dim=1)
                x = self.pretrained_model.segmentor.fusion(x)

                x = F.interpolate(
                    x,
                    size=(tile_size, tile_size), #ize=(45, 45),
                    mode="bilinear",
                    align_corners=False,
                )  # Resize to match labels size

            elif self.encoder_type == "mae":
                x = self.encoder(x)     

                for i in range(len(x)):
                    if perturb:
                        x[i] = x[i] + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                (x[i].shape))).type(x[i].dtype).to(x[i].device)
                    x[i] = self.pretrained_model.segmentor.upsamples[i](x[i])
                x = torch.cat(x, dim=1)
                x = self.pretrained_model.segmentor.fusion(x)

                x = F.interpolate(
                    x,
                    size=(tile_size, tile_size),
                    mode="bilinear",
                    align_corners=False,
                )  # Resize to match labels size


                #y = F.softmax(self.segmentor.seg_head(y), dim=1)
                #y = torch.flatten(y.permute(0,2,3,1), start_dim=0, end_dim=2)


            else:
                x = self.encoder(x) #.flatten(start_dim=1)

                if perturb:
                    x = x + torch.from_numpy(self.rng.normal(0.0, 0.01, \
                                        x.shape)).type(x.dtype).to(x.device)


        if isinstance(y,tuple):
            y = y #[0]

        if isinstance(y,list):
            y = y #[0]

        #print("HERE OUTPUT", y.shape, x.shape)

  
        tmp_prob_subset = None
        tmp_subset = None
        tmp = y.clone()

        #print(y.shape, y.ndim, y.shape[1], (y.ndim == 4 and y.shape[1] > 1), torch.argmax(y, dim=1).shape, "CLAY")
        if (y.ndim == 2 and y.shape[1] > 1) or (y.ndim == 4 and y.shape[1] > 1):
            tmp = torch.argmax(y, dim=1)
        elif (y.ndim == 3 and y.shape[0] > 1):
            tmp = torch.argmax(y, dim=0) 

        tmp_full = None
        tmp_prob_full = None
        if y.ndim == 2:
            if train:
                tmp_full = torch.zeros((y.shape[0], self.num_classes), device=y.device, dtype=torch.float32)
            else:
                tmp_full = torch.zeros((y.shape[0], 1), device=y.device, dtype=torch.float32)
            tmp_prob_full = torch.zeros((y.shape[0], 1), device=y.device, dtype=torch.float32)
        elif y.ndim == 3:
            if train:
                tmp_full = torch.zeros((self.num_classes, y.shape[-2], y.shape[-1]), device=y.device, dtype=torch.float32)
            else:
                tmp_full = torch.zeros((1, y.shape[-2], y.shape[-1]), device=y.device, dtype=torch.float32)
            tmp_prob_full = torch.zeros((1, y.shape[-2], y.shape[-1]), device=y.device, dtype=torch.float32)
        elif y.ndim == 4:
            if train:
                tmp_full = torch.zeros((y.shape[0], self.num_classes, y.shape[-2], y.shape[-1]), device=y.device, dtype=torch.float32)
            else:
                tmp_full = torch.zeros((y.shape[0], 1, y.shape[-2], y.shape[-1]), device=y.device, dtype=torch.float32)
            tmp_prob_full = torch.zeros((y.shape[0], 1, y.shape[-2], y.shape[-1]), device=y.device, dtype=torch.float32)


        f = lambda z: str(z)
        #tmp2 = y  strings
        #tmp = y
        
        #BxHxW OR B 
        tmp2 = np.vectorize(f)(tmp.detach().cpu())
        #BxCxHxW OR BxC 
        tmp3 = tmp.clone()
        #x.requires_grad = True
        keys = np.unique(tmp2)
        #print("HERE KEYS", keys)
        for key in keys:
            #BxCxHxW OR BxC 
            tmp = y.clone()
            inds = None
            inds2 = None
            tmp2_2 = None
            tmp4 = None
            input_tmp = None
            if train and key != self.key:
                continue

            inds = np.where(tmp2 == key)

            if tmp.ndim == 1 or tmp.ndim == 2:
                inds2 = inds[0]
                tmp2_2 = copy.deepcopy(tmp2)
                tmp4 = tmp3.clone()
            elif tmp.ndim == 3: 
                tmp2_2 = tmp2.reshape(tmp2.shape[0], -1).transpose(1,0)
                tmp4 = torch.flatten(tmp3, start_dim=0)
                inds2 = np.where(tmp2_2 == key)[0]

            elif tmp.ndim == 4:
                
                tmp2_2 = tmp2.ravel() 
                tmp4 = torch.flatten(tmp3, start_dim=0)
                inds2 = np.where(tmp2_2 == key)[0]            

 
            #print(tmp.shape, tmp.ndim)
            #print(tmp2.shape, tmp2_2.shape, tmp4.shape, tmp3.shape, tmp.ndim, x.shape, y.shape, "HERE")
            if x.ndim == 2:
                input_tmp = x.clone()
            elif x.ndim == 3:
                input_tmp = x.clone() #ch.flatten(x, start_dim=2) #.permute(1,0)
            elif x.ndim == 4:
                input_tmp = x.clone()
                input_tmp = torch.flatten(x.permute(1,0,2,3), start_dim=1).permute(1,0)
                #input_tmp = torch_flatten(input_tmp, start_dim=0, end_dim=1)
              
            #print(input_tmp.shape, x.shape, tmp2_2.shape, tmp2.shape)
            #print(min(inds2), max(inds2), input_tmp.shape, "HERE", tmp2.shape, tmp2_2.shape, tmp4.shape, tmp.shape, tmp3.shape, x.shape)
            input_tmp = input_tmp[inds2,:] 


            tmp_prob = None

            if key in self.clust_tree["1"].keys() and self.clust_tree["1"][key] is not None:
                tmp = self.clust_tree["1"][key].forward(input_tmp) #torch.unsqueeze(x[inds],dim=0))
                if isinstance(tmp,tuple):
                    tmp = tmp[0]
                if isinstance(tmp,list):
                    tmp = tmp[0]

                if train == False:
                    tmp_prob = torch.max(tmp, dim=1).values
                    tmp = torch.argmax(tmp, dim=1)
                    tmp = tmp + self.num_classes*tmp4[inds2]
                    tmp = torch.unsqueeze(tmp, dim=1)
                    tmp_prob = torch.unsqueeze(tmp_prob, dim=1)
            elif train == False:
                tmp = self.num_classes*tmp4[inds2]
                tmp = torch.unsqueeze(tmp, dim=1)
            else:
                continue

            if tmp_subset is None:
                tmp_subset = tmp.clone()
                if tmp_prob is not None:
                    tmp_prob_subset = tmp_prob.clone()
            else:
                if tmp_prob is not None and tmp_prob_subset is not None:
                    tmp_prob_subset = torch.cat((tmp_prob_subset, tmp_prob), dim=0)
                tmp_subset = torch.cat((tmp_subset, tmp), dim=0)

            #print(y.shape, tmp.shape, tmp_full.shape, y.shape)


            if y.ndim == 2:
                tmp_full[inds[0],:] = tmp.type(tmp_full.dtype)
                if tmp_prob is not None:
                    tmp_prob_full[inds[0],:] = tmp_prob.type(tmp_prob_full.dtype)
            elif y.ndim == 3:
                tmp_full[:,inds[0],inds[1]] = tmp.type(tmp_full.dtype)
                if tmp_prob is not None:
                    tmp_prob_full[:,inds[0],inds[1]] = tmp_prob.type(tmp_prob_full.dtype)
            elif y.ndim == 4:
                tmp_full[inds[0],:,inds[1],inds[2]] = tmp.type(tmp_full.dtype)
                if tmp_prob is not None:
                    tmp_prob_full[inds[0],:,inds[1],inds[2]] = tmp_prob.type(tmp_prob_full.dtype)
            del tmp
            del tmp_prob

        if tmp_subset is None or tmp_full is None:
            return None

        if return_embed:
            return tmp_subset, tmp_full, x, tmp_prob_subset, tmp_prob_full
        return tmp_subset, tmp_full, tmp_prob_subset, tmp_prob_full

    
    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        x = batch

        output  = self.forward(x, train=True)
        y = None
        y2 = None
        if output is None:
            return None
        else:
            y = output[0]

        output2  = self.forward(x.clone(), perturb=True, train=True)
        if output2 is None:
            return None
        else:
            y2 = output2[0]

        loss, loss2 = self.criterion(y,y2, lamb = 2.0) #calculate loss
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch 

        output  = self.forward(x, train=True)
        y = None
        y2 = None
        if output is None:
            return None
        else:
            y = output[0]

        output2  = self.forward(x.clone(), perturb=True, train=True)
        if output2 is None:
            return None
        else:
            y2 = output2[0]

        loss = self.criterion(y,y2, lamb = 2.0)[0] #calculate loss
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx):
        
        _, y  = self.forward(x, train=False)
        return y
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }



def get_state_dict(clust_tree, lab_full):

    state_dict = {}
    for lab1 in clust_tree.keys():
        if lab1 == "0":
            continue
        if lab1 not in state_dict:
            state_dict[lab1] = {}
            for lab2 in clust_tree[lab1].keys():
                if lab2 not in state_dict[lab1].keys():
                    if clust_tree[lab1][lab2] is not None:
                        if lab2 not in state_dict[lab1].keys():
                            state_dict[lab1][lab2] = {}
                        state_dict[lab1][lab2]["model"] = clust_tree[lab1][lab2].state_dict()
                        uid = str(uuid.uuid1())
    state_dict["labels"] = lab_full
    return state_dict

def load_model(clust_tree, n_visible, model, state_dict, device, ijepa=False, num_classes=100):
        lab_full = state_dict["labels"]
        pytorch_total_params = 0
        for lab1 in clust_tree.keys():
            if lab1 == "0":
                continue
            for lab2 in lab_full.keys():
                clust_tree[lab1][lab2] = None
                if lab2 in state_dict[lab1].keys():
                    #clust_tree[lab1][lab2] = OutputProjection(224, model.pretrained_model.patch_size, model.pretrained_model.embed_dim, model.num_classes)
                    #print(lab1, lab2, n_visible, model.num_classes, model.number_heads)
                    #if ijepa:
                    #    clust_tree[lab1][lab2] = JEPA_Seg(num_classes)	
                    #else:
                    clust_tree[lab1][lab2] = MultiPrototypes(n_visible, model.num_classes, model.number_heads).to(device)
                    clust_tree[lab1][lab2].load_state_dict(state_dict[lab1][lab2]["model"])
                    pytorch_total_params = pytorch_total_params + sum(p.numel() for p in model.parameters())
        print("PARAMS 2", pytorch_total_params)
        return clust_tree, lab_full





