import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
from ijepa_model import IJEPA_base



class IJEPADataset(Dataset):
    def __init__(self,
                 data,
                 ):
        super().__init__()
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


'''Placeholder for datamodule in pytorch lightning'''
'''
Placeholder for datamodule in pytorch lightning
'''
class D2VDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_path,
                 batch_size=16,
                 num_workers=4,
                 pin_memory=True,
                 shuffle=True
                 ):
        super().__init__()
        
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        
    def setup(self, stage=None):
        data = np.load("/data/nlahaye/output/Learnergy/DBN_eMAS_FULL_STRAT_Segformer_DINO/train_data.npy")
        train_max = int(data.shape[0]*0.9)
        print(train_max, data.shape)
        self.train_dataset = IJEPADataset(torch.from_numpy(data[:train_max]))
        self.val_dataset = IJEPADataset(torch.from_numpy(data[train_max:]))
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

'''
pytorch lightning model
'''
class IJEPA(pl.LightningModule):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3, 
            embed_dim=64,
            enc_heads=8,
            enc_depth=8,
            decoder_depth=6,
            lr=1e-6,
            weight_decay=0.05,
            target_aspect_ratio = (0.75,1.5),
            target_scale = (0.15, .2),
            context_aspect_ratio = 1,
            context_scale = (0.85,1.0),
            M = 4, #number of different target blocks
            m=0.996, #momentum
            m_start_end = (.996, 1.)

    ):


        #number_channel = yml_conf["data"]["number_channels"]
        #tile_size = yml_conf["data"]["tile_size"]

        #pretrained = yml_conf["encoder"]["pretrained"]
        #depth = yml_conf["encoder"]["depth"]

        #encoder_type = yml_conf["encoder"]["encoder_type"]

        #out_dir = yml_conf["output"]["out_dir"]
        #os.makedirs(out_dir, exist_ok=True)

        #model_fname = yml_conf["output"]["model"]
        #model_file = os.path.join(out_dir, model_fname)

        if encoder_type == "resnet":
            pretrained = yml_conf["encoder"]["pretrained"]
            depth = yml_conf["encoder"]["depth"]
 
            self.encoder = multichannel_resnet.Resnet_multichannel(pretrained=pretrained, encoder_depth=depth, num_in_channels=number_channel)

            layer1 = "layer2"
            layer2 = -2

        else
            self.encoder = RegionViT(
                dim = (64, 128, 256, 512),      # tuple of size 4, indicating dimension at each stage
                depth = (2, 2, 14, 2),           # depth of the region to local transformer at each stage
                window_size = 7,                # window size, which should be either 7 or 14
                local_patch_size = 4,
                num_classes = 1000,             # number of output classes
                tokenize_local_3_conv = False,  # whether to use a 3 layer convolution to encode the local tokens from the image. the paper uses this for the smaller models, but uses only 1 conv (set to False) for the larger models
                channels = number_channel,
                use_peg = False,                # whether to use positional generating module. they used this for object detection for a boost in performance
             )
            layer1 = "tmp1"
            layer2 = "tmp2"
 

        self.learner = None
        self.learner = PixelCL(
            encoder,
            image_size = tile_size[0],
            hidden_layer_pixel = layer1,  # leads to output of 8x8 feature map for pixel-level learning
            hidden_layer_instance = layer2,     # leads to output for instance-level learning
            projection_size = 256,          # size of projection output, 256 was used in the paper
            projection_hidden_size = 2048,  # size of projection hidden dimension, paper used 2048
            moving_average_decay = 0.99,    # exponential moving average decay of target encoder
            ppm_num_layers = 1,             # number of layers for transform function in the pixel propagation module, 1 was optimal
            ppm_gamma = 2,                  # sharpness of the similarity in the pixel propagation module, already at optimal value of 2
            distance_thres = 0.7,           # ideal value is 0.7, as indicated in the paper, which makes the assumption of each feature map's pixel diagonal distance to be 1 (still unclear)
            similarity_temperature = 0.3,   # temperature for the cosine similarity for the pixel contrastive loss
            alpha = 1.,                      # weight of the pixel propagation loss (pixpro) vs pixel CL loss
            use_pixpro = True,               # do pixel pro instead of pixel contrast loss, defaults to pixpro, since it is the best one
            cutout_ratio_range = (0.6, 0.8)  # a random ratio is selected from this range for the random cutout
        )




        #define loss
        self.criterion = nn.MSELoss()
    
    def forward(self, x, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale):
        return self.model(x, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale)
    

    def training_step(self, batch, batch_idx):
        x = batch
        loss = self.learner(x) # if positive pixel pairs is equal to zero, the loss is equal to the instance level loss
        self.log('train_loss', loss)
                    
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        self.log('train_loss', loss)
        self.log('val_loss', loss)
        
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        x = batch
        self.encoder.mode = "test"
        return self(x)

    def on_after_backward(self):
        self.learner.update_moving_average() # update moving average of target encoder


    def configure_optimizers(self):
        opt = torch.optim.Adam(self.learner.parameters(), lr=self.lr #e-4)
        return opt
 

if __name__ == '__main__':
    dataset = D2VDataModule(dataset_path='data')

    model = IJEPA(img_size=224, patch_size=16, in_chans=3, embed_dim=64, enc_heads=8, enc_depth=8, decoder_depth=6, lr=1e-3)
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    wandb_logger = WandbLogger(project="SIT-FUSE", log_model=True, save_dir = "/home/nlahaye/SIT_FUSE_DEV/wandb/")

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=16,
        max_epochs=100000,
        callbacks=[lr_monitor, model_summary],
        gradient_clip_val=.1,
        logger=wandb_logger
    )

    trainer.fit(model, dataset)
