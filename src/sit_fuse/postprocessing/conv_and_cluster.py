from osgeo import gdal
import torch
from skimage.util import view_as_windows
import numpy as np
from tqdm import tqdm
from torchsummary import summary
from sklearn.cluster import MiniBatchKMeans, AffinityPropagation
import os
import joblib
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import argparse
from sit_fuse.utils import read_yaml

 
def initialize_weights(model):
    for layer in model.children():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)



def write_geotiff(dat, imgData, fname):

    nx = imgData.shape[1]
    ny = imgData.shape[0]
    geoTransform = dat.GetGeoTransform()
    wkt = dat.GetProjection()
    gcpcount = dat.GetGCPCount()
    gcp = None
    gcpproj = None
    if gcpcount > 0:
        gcp = dat.GetGCPs()
        gcpproj = dat.GetGCPProjection()
    out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Float32)
    print(fname)
    out_ds.SetGeoTransform(geoTransform)
    out_ds.SetProjection(wkt)
    if gcpcount > 0:
        out_ds.SetGCPs(gcp, gcpproj)
    out_ds.GetRasterBand(1).WriteArray(imgData)
    out_ds.FlushCache()
    out_ds = None


def load_model(model_fpath):

    # Step 1: Initialize model with the best available weights
    model = resnet152(weights=None)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #here 4 indicates 4-channel input
    model.load_state_dict(torch.load(model_fpath))

    model.eval()
    return_nodes = {'flatten': 'flatten'}
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
    return model, feature_extractor

def get_model():

    # Step 1: Initialize model with the best available weights
    model = resnet152(weights=None)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #here 4 indicates 4-channel input
    initialize_weights(model)
    model.eval()
    return_nodes = {'flatten': 'flatten'}
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
    return model, feature_extractor


 
def get_data(fname, tile, stride):
    img_test = gdal.Open(fname)
    img_data = img_test.ReadAsArray().astype(np.float32)
    print(img_data.shape)

    tgts = np.indices(img_data.shape)

    tmp = np.squeeze(view_as_windows(img_data, [tile, tile], step=[stride, stride]))
    tmp = tmp.reshape(-1,tile,tile)

    delete_inds = []
    for i in range(tmp.shape[0]):
        if np.max(tmp[i,:,:]) <= 0.0:
            delete_inds.append(i)

    tmp = np.delete(tmp, delete_inds, axis=0)

    print(tgts.shape, img_data.shape)
    tgts2 = np.squeeze(view_as_windows(tgts, [2,tile,tile], step=[2,stride,stride]))
    print(tgts2.shape)
    print(tgts2.shape)
    tgts2 = tgts2.reshape(-1,2,tile,tile)
    tgts2 = np.delete(tgts2, delete_inds, axis=0)

    tgts2 = torch.from_numpy(tgts2)
    tmp = torch.from_numpy(tmp)
    tmp = torch.unsqueeze(tmp,1)

    dataset = torch.utils.data.TensorDataset(tmp, tgts2)

    loader = torch.utils.data.DataLoader(dataset, batch_size=100,
                    pin_memory=True, shuffle=False)
 
    return loader, tmp, tgts2, img_data.shape, img_test



def get_edge_tiles(fname, tile, stride):
    img_test = gdal.Open(fname)
    img_data = img_test.ReadAsArray().astype(np.float32)
    print(img_data.shape)

    tgts = np.indices(img_data.shape)

    tgts = np.flip(np.flip(tgts, axis = 1), axis=2)
    img_data = np.flip(np.flip(img_data, axis = 0), axis=1)

    img_data[tile:,tile:] = -1.0
    tgts[tile:,tile:] = -1.0


    tmp = np.squeeze(view_as_windows(img_data, [tile, tile], step=[tile, tile]))
    tmp = tmp.reshape(-1,tile,tile)

    delete_inds = []
    for i in range(tmp.shape[0]):
        if np.max(tmp[i,:,:]) <= 0.0:
            delete_inds.append(i)

    tmp = np.delete(tmp, delete_inds, axis=0)

    print(tgts.shape, img_data.shape)
    tgts2 = np.squeeze(view_as_windows(tgts, [2,tile,tile], step=[2,tile,tile]))
    print(tgts2.shape)
    print(tgts2.shape)
    tgts2 = tgts2.reshape(-1,2,tile,tile)
    tgts2 = np.delete(tgts2, delete_inds, axis=0)


    tgts2 = torch.from_numpy(tgts2)
    tmp = torch.from_numpy(tmp)
    tmp = torch.unsqueeze(tmp,1)
 

    dataset = torch.utils.data.TensorDataset(tmp, tgts2)

    loader = torch.utils.data.DataLoader(dataset, batch_size=100,
                    pin_memory=True, shuffle=False)

    return loader, tgts2



 
def generate_features(model, feature_extractor, loader):
    dat_full = None
    for i,data in enumerate(tqdm(loader)):
        images, _ = data
        fill_only = False
        if torch.max(images) <= 0.0:
            fill_only = True
            flatten_fts = np.zeros((images.shape[0],2048)) - 1.0
        else:
            #images = preprocess(images)
            images = images.cuda()


            #images = model(images).detach().cpu()
            features = feature_extractor(images)
            flatten_fts = features["flatten"].squeeze().detach().cpu()
            print(flatten_fts.shape)


        if flatten_fts.ndim < 2:
            flatten_fts = torch.unsqueeze(flatten_fts, 0)

        if dat_full is None:
            if fill_only:
                dat_full = flatten_fts
            else:
                dat_full = flatten_fts.numpy() #.reshape(.shape[0],-1)
        else:
            if fill_only:
                dat_full = np.concatenate((dat_full, flatten_fts), axis=0)
            else:
                dat_full = np.concatenate((dat_full, flatten_fts.numpy()), axis=0) #.reshape(images.shape[0],-1), dat_full), axis=0)

        if dat_full.ndim < 2:
            dat_full = np.expand_dims(dat_full, 0)

    return dat_full 



 
def train_cluster(dat_full):

    clust = AffinityPropagation(random_state=5).fit(dat_full)
    return clust



def cluster_and_output(output, clust, tgts2, img_data_shape, fname, img_test, tile):
    print(np.unique(output))
 
    line_ind = 0
    samp_ind = 1
    strt_dim1 = 0
    strt_dim2 = 0
 
    outp = np.zeros(img_data_shape) - 2
    print("POST KMEANS", output.shape, tgts2.shape)
 
    for i in range(output.shape[0]):
        #print(tgts2[i,line_ind,0,0], tgts2[i,samp_ind,0,0], outp.shape)
        #print(tgts2[i,line_ind,-1,-1], tgts2[i,samp_ind,-1,-1], output.shape, tmp.shape, tgts2.shape)
 
        #max_line = min(outp.shape[0],   tgts2[i,line_ind]+output[i].shape[0])
        #max_samp = min(outp.shape[1],   tgts2[i,samp_ind]+output[i].shape[1])
  
       if tgts2[i,line_ind,-1,-1] > tgts2[i,line_ind,0,0]:
           outp[tgts2[i,line_ind,0,0]:tgts2[i,line_ind,-1,-1], tgts2[i,samp_ind,0,0]:tgts2[i,samp_ind,-1,-1]] = output[i] #output[i, :max_line - tgts2[i,line_ind], :max_samp - tgts2[i,samp_ind]]
       else: #Edge cases
           outp[tgts2[i,line_ind,-1,-1]:tgts2[i,line_ind,0,0], tgts2[i,samp_ind,-1,-1]:tgts2[i,samp_ind,0,0]] = output[i]  

    write_geotiff(img_test, outp, fname + ".tile_cluster." + str(tile) + ".tif")


def pretrained_conv_and_cluster(clust, model, test_fnames, tile):

    stride = int(tile*0.6)

    for i in range(len(test_fnames)):
        loader, tmp, tgts2, img_data_shape, img_test = get_data(test_fnames[i], tile, stride)
        loader_edge, tgts2_edge = get_edge_tiles(test_fnames[i], tile, stride)

        with torch.no_grad():
            dat_full = generate_features(model, feature_extractor, loader)
            dat_full_edge = generate_features(model, feature_extractor, loader_edge)

            if dat_full is None:
                if dat_full_edge is None:
                    continue
                dat_full = dat_full_edge
                tgts2 = tgts2_edge
            elif dat_full_edge is not None:
                if dat_full.ndim == 1:
                    dat_full = np.expand_dims(dat_full, 0)
                if dat_full_edge.ndim == 1:
                    dat_full_edge = np.expand_dims(dat_full_edge, 0)
                dat_full = np.concatenate((dat_full, dat_full_edge), axis=0)
                tgts2 = np.concatenate((tgts2, tgts2_edge), axis=0)
            else:
                continue



            print(dat_full.shape)
            output = clust.predict(dat_full)
            cluster_and_output(output, clust, tgts2, img_data_shape, test_fnames[i], img_test, tile)



 
def run_conv_and_cluster(train_fname, test_fnames, tiles): 
    for tle in range(len(tiles)): 

        tile = tiles[tle]

        stride = int(tile*0.6)

        model, feature_extractor = get_model()
        loader, tmp, tgts2, img_data_shape, img_test = get_data(train_fname, tile, stride)
        loader_edge, tgts2_edge = get_edge_tiles(train_fname, tile, stride)
 
        print(tmp.shape)

        dat_full = None
        clust = None

        with torch.no_grad():
            model = model.cuda()
            feature_extractor = feature_extractor.cuda()
            print(summary(model, (1,tile,tile)))  

            dat_full = generate_features(model, feature_extractor, loader)
            dat_full_edge = generate_features(model, feature_extractor, loader_edge)

            print(dat_full.shape, dat_full_edge.shape, tgts2.shape, tgts2_edge.shape)
            dat_full = np.concatenate((dat_full, dat_full_edge), axis=0)
            tgts2 = np.concatenate((tgts2, tgts2_edge), axis=0)

            clust = train_cluster(dat_full)
     
            #model = model.cpu()

        print(tmp.shape)

        output = clust.predict(dat_full)
        cluster_and_output(output, clust, tgts2, img_data_shape, train_fname, img_test, tile)

        del loader
        del tmp
        del tgts2
        del dat_full

        del loader_edge
        del tgts2_edge
    
        for i in range(len(test_fnames)):
            loader, tmp, tgts2, img_data_shape, img_test = get_data(test_fnames[i], tile, stride)
            loader_edge, tgts2_edge = get_edge_tiles(test_fnames[i], tile, stride)

            with torch.no_grad():
                dat_full = generate_features(model, feature_extractor, loader)
                dat_full_edge = generate_features(model, feature_extractor, loader_edge)

                if dat_full is None:
                    if dat_full_edge is None:
                        continue
                    dat_full = dat_full_edge
                    tgts2 = tgts2_edge
                elif dat_full_edge is not None:
                    if dat_full.ndim == 1:
                        dat_full = np.expand_dims(dat_full, 0)
                    if dat_full_edge.ndim == 1:
                        dat_full_edge = np.expand_dims(dat_full_edge, 0)
                    dat_full = np.concatenate((dat_full, dat_full_edge), axis=0)
                    tgts2 = np.concatenate((tgts2, tgts2_edge), axis=0)        
                else:
                    continue

            

                print(dat_full.shape)
                output = clust.predict(dat_full)
                cluster_and_output(output, clust, tgts2, img_data_shape, test_fnames[i], img_test, tile)
        torch.save(model.state_dict(), os.path.dirname(test_fnames[0]) + "/model_" + str(tiles[tle]) + ".ckpt")
        joblib.dump(clust, os.path.dirname(test_fnames[0]) + "/clust_" + str(tiles[tle]) + ".joblib") 


def run_pretrained_conv_and_cluster(test_fnames, tiles):

    for tle in range(len(tiles)):
  
        model_fpath = os.path.dirname(test_fnames[0]) + "/model_" + str(tiles[tle]) + ".ckpt"
        model, feature_extractor = load_model(model_fpath)

        clust = joblib.load(os.path.dirname(test_fnames[0]) + "/clust_" + str(tiles[tle]) + ".joblib")

        pretrained_conv_and_cluster(clust, model, test_fnames, tiles[tle])
 

def conv_and_cluster_outside(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    conv_and_cluster(yml_conf)

def conv_and_cluster(yml_conf):

    #Run 
    train_fname = yml_conf["train_fname"]
    test_fnames = yml_conf["test_fnames"]
    tiles = yml_conf["tile_size"]


    if os.path.exists(os.path.dirname(test_fnames[0]) + "/clust_" + str(tiles[tle]) + ".joblib"):
        run_pretrained_conv_and_cluster(test_fnames, tiles)
    else:
        run_conv_and_cluster(train_fname, test_fnames, tiles)
 



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)
