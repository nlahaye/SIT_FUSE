import re
import os
import numpy as np
import glob

import torch
import zarr

dirpth = "/data/nlahaye/remoteSensing/MADOS/MADOS/"

valpth = dirpth + "splits/val_X.txt"
testpth = dirpth  + "splits/test_X.txt"
trainpth = dirpth  + "splits/train_X.txt"
 
embed_dirpth = "/data/nlahaye/output/Pangaea/embeddings/MADOS/"

models = ["croma_optical", "gfm_swin", "remoteclip_encoder", "ScaleMAE", "ssl4eo_data2vec", "ssl4eo_mae_optical", "unet_encoder",\
"dofa_encoder", "Prithvi", "satlas_pretrain", "SpectralGPT", "ssl4eo_dino", "ssl4eo_moco", "vit_encoder"]
  
dir1_re = "(Scene_\d+)_(\d+)"
dir2_re = "(Scene_\d+_L2R_rhorc)_\d+_(\d+)"
#embd_Scene_132_L2R_rhorc_443_1.npy

lnes = []
with open(testpth, "r") as f:
    lnes = f.readlines()

ten_ms = ["4*", "5*", "6*", "8*"]
twenty_ms = ["704", "740", "783", "865", "1614", "2202"]
sixty_ms =  ["44*"]


#Scene_12_L2R_rhorc_443_45.tif
scenes = []
embed_scenes = []

for lne in lnes:
    scene = []
    embed_scene = []
    pth = []
    mtch = re.search(dir1_re, lne)
    for i in range(len(sixty_ms)):
        basename = str(mtch.group(1)) +  "_L2R_rhorc_" + sixty_ms[i] + "_" + str(mtch.group(2))
        pth = dirpth + str(mtch.group(1)) + "/60/" + basename + ".tif"
        print(pth)
        fglob = glob.glob(pth)
        fname_tmp = fglob[0]
        scene.append(fname_tmp)
        final_basename = os.path.splitext(os.path.basename(fname_tmp))[0]
        embed_scene.append("embd_" + final_basename + ".npy")
       
    embed_scenes.append(embed_scene)
    scenes.append(scene)

 
for m in range(len(models)):
    drpth = os.path.join(os.path.join(embed_dirpth, models[m]), "test")
    scale_factor = -1
    pixel_shuffle = None
    print(models[m])
    for s in range(len(embed_scenes)):
        scene = embed_scenes[s]
 
        scene_embed = None
        for c in range(len(scene)):
            scene_file = os.path.join(drpth, scene[c])
            embd = np.load(scene_file)
            if scene_embed is None:
                scene_embed = embd
            else:
                scene_embed = np.concatenate((scene_embed, embd))

        if scene_embed is None:
            continue
            
 
        if scale_factor < 1:
            scale_factor = int(round(240 / scene_embed.shape[1]))
            print(scale_factor, scene_embed.shape[1])
            pixel_shuffle = torch.nn.PixelShuffle(scale_factor).cuda()
        scene_embed = torch.from_numpy(scene_embed).cuda()

        scene_embed = pixel_shuffle(scene_embed).detach().cpu().numpy()
        
        print(scene_embed.shape)

        print(dir2_re, scene[0])
        out_base_re = re.search(dir2_re, scene[0])
        out_base = out_base_re.group(1) + out_base_re.group(2) + ".zarr"
        output_flepth = os.path.join(drpth, out_base)

        print(output_flepth)
        zarr.save(output_flepth, scene_embed)


    if pixel_shuffle is not None:
        pixel_shuffle = pixel_shuffle.cpu()
        del pixel_shuffle








