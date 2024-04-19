from pixel_level_contrastive_learning import PixelCL
from torchvision import models
from tqdm import tqdm

import torch
#from segformer_pytorch import Segformer
from vit_pytorch.regionvit import RegionViT
 
from vit_pytorch import Dino
import numpy as np


model = RegionViT(
    dim = (64, 128, 256, 512),      # tuple of size 4, indicating dimension at each stage
    depth = (2, 2, 14, 2),           # depth of the region to local transformer at each stage
    window_size = 7,                # window size, which should be either 7 or 14
    local_patch_size = 4,
    num_classes = 1000,             # number of output classes
    tokenize_local_3_conv = False,  # whether to use a 3 layer convolution to encode the local tokens from the image. the paper uses this for the smaller models, but uses only 1 conv (set to False) for the larger models
    channels = 3,
    use_peg = False,                # whether to use positional generating module. they used this for object detection for a boost in performance
)

#model = Segformer(
#    dims = (16, 32, 80, 128),      # dimensions of each stage
#    heads = (1, 2, 5, 8),           # heads of each stage
#    ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
#    reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
#    num_layers = 2,                 # num layers of each stage
#    decoder_dim = 256,              # decoder dimension
#    num_classes = 1000,                 # number of segmentation classes
#    channels = 3
#)

#from fastervit import create_model
#model = create_model('faster_vit_0_any_res', 
#                          resolution=[256, 256],
#                          window_size=[7, 7, 12, 6],
#                          ct_size=2,
#                          dim=64,
#                          in_chans=34,
#                          pretrained=True)




#learner = Dino(
#    model,
#    image_size = 256,
#    hidden_layer = 'to_latent',        # hidden layer name or index, from which to extract the embedding
#    projection_hidden_size = 256,      # projector network hidden dimension
#    projection_layers = 4,             # number of layers in projection network
#    num_classes_K = 65336,             # output logits dimensions (referenced as K in paper)
#    student_temp = 0.9,                # student temperature
#    teacher_temp = 0.04,               # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
#    local_upper_crop_scale = 0.4,      # upper bound for local crop - 0.4 was recommended in the paper 
#    global_lower_crop_scale = 0.5,     # lower bound for global crop - 0.5 was recommended in the paper
#    moving_average_decay = 0.9,        # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
#    center_moving_average_decay = 0.9, # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
#)

learner = PixelCL(
    model,
    image_size = 224,
    in_channels = 3,
    hidden_layer_pixel = 'tmp2',  # leads to output of 8x8 feature map for pixel-level learning
    hidden_layer_instance = 'tmp1',     # leads to output for instance-level learning
    projection_size = 224,          # size of projection output, 256 was used in the paper
    projection_hidden_size = 2048,  # size of projection hidden dimension, paper used 2048
    moving_average_decay = 0.99,    # exponential moving average decay of target encoder
    ppm_num_layers = 1,             # number of layers for transform function in the pixel propagation module, 1 was optimal
    ppm_gamma = 2,                  # sharpness of the similarity in the pixel propagation module, already at optimal value of 2
    distance_thres = 0.7,           # ideal value is 0.7, as indicated in the paper, which makes the assumption of each feature map's pixel diagonal distance to be 1 (still unclear)
    similarity_temperature = 0.3,   # temperature for the cosine similarity for the pixel contrastive loss
    alpha = 1.,                      # weight of the pixel propagation loss (pixpro) vs pixel CL loss
    use_pixpro = True,               # do pixel pro instead of pixel contrast loss, defaults to pixpro, since it is the best one
    cutout_ratio_range = (0.6, 0.8)  # a random ratio is selected from this range for the random cutout
).cuda()


opt = torch.optim.Adam(learner.parameters(), lr = 3e-4)
data_train = np.load("/data/nlahaye/output/Learnergy/DBN_eMAS_FULL_STRAT_Segformer_DINO/train_data.npy")


def sample_batch_images():
    inds = list(range(0,data_train.shape[0]))
    inds2 = np.random.choice(inds, 10, replace=False)
    return torch.from_numpy(data_train[inds2]).cuda()

for _ in tqdm(range(100000)):
    images = sample_batch_images()
    print(images.shape)
    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average() # update moving average of teacher encoder and teacher centers

# save your improved network
torch.save(model.state_dict(), '/data/nlahaye/output/Learnergy/DBN_eMAS_FULL_STRAT_Segformer_DINO/pretrained-vit_RGB.pt')


