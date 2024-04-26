import pytorch_lightning as pl
import torch

from learnergy.models.deep import DBN

from sit_fuse.models.deep_cluster.dc import DeepCluster
from sit_fuse.models.deep_cluster.ijepa_dc import IJEPA_DC
from sit_fuse.models.deep_cluster.dbn_dc import DBN_DC
from sit_fuse.datasets.dataset_utils import get_prediction_dataset
from sit_fuse.models.deep_cluster.heir_dc import Heir_DC

from sit_fuse.utils import read_yaml

import argparse
import os

def generate_output(dat, mdl, use_gpu, out_dir, output_fle, mse_fle, pin_mem = True, conv = False):
    output_full = None
    count = 0
    dat.current_subset = -1
    dat.next_subset()

    ind = 0
    output_batch_size = min(5000, max(int(dat.data_full.shape[0] / 5), dat.data_full.shape[0]))

    output_sze = dat.data_full.shape[0]
    append_remainder = int(output_batch_size - (output_sze % output_batch_size))

    if isinstance(dat.data_full,torch.Tensor):
        dat.data_full = torch.cat((dat.data_full,dat.data_full[0:append_remainder]))
        dat.targets_full = torch.cat((dat.targets_full,dat.targets_full[0:append_remainder]))
    else:
        dat.data_full = np.concatenate((dat.data_full,dat.data_full[0:append_remainder]))
        dat.targets_full = np.concatenate((dat.targets_full,dat.targets_full[0:append_remainder]))

    test_loader = DataLoader(dat, batch_size=output_batch_size, shuffle=False, \
    num_workers = 0, drop_last = False, pin_memory = pin_mem)
    ind = 0
    ind2 = 0
    for data in tqdm(test_loader):
        if use_gpu:
            dat_dev, lab_dev = data[0].cuda(), data[1].cuda()
        else:
            dat_dev, lab_dev = data[0].cuda(), data[1].cuda()

        with torch.no_grad():
            output = mdl.forward(dat_dev)
        if isinstance(output, list):
            output = output[0] #TODO improve usage uf multi-headed output after single-headed approach validated
        output = torch.unsqueeze(torch.argmax(output, axis = 1), axis=1)
 
        if use_gpu == True:
            output = output.detach().cpu()

        dat_dev = dat_dev.detach().cpu()
        lab_dev = lab_dev.detach().cpu()

        if output_full is None:
            output_full = torch.zeros(dat.data_full.shape[0], output.shape[1], dtype=torch.float32)
        ind1 = ind2
        ind2 += dat_dev.shape[0]
        if ind2 > output_full.shape[0]:
            ind2 = output_full.shape[0]
        output_full[ind1:ind2,:] = output
        ind = ind + 1
        del output
        del dat_dev
        del lab_dev
        del loader
        count = count + 1

    print("SAVING", os.path.join(out_dir, output_fle))
    torch.save(output_full, os.path.join(out_dir, output_fle), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(dat.targets_full, os.path.join(out_dir, output_fle + ".indices"), pickle_protocol=pickle.HIGHEST_PROTOCOL)

    plot_clusters(dat.targets_full, output_full, os.path.join(out_dir, output_fle)) 

def plot_clusters(coord, output_data, output_basename, pixel_padding=1):

        max_cluster = output_data.shape[1]
        min_cluster = 0
        labels = None
        if output_data.shape[1] > 1:
            max_cluster = output_data.shape[1]
            labels = np.argmax(output_data, axis = 1)
        else:
            labels = output_data.astype(np.int32)
            max_cluster = disc_data.max() #TODO this better!!!

        print(np.unique(disc_data).shape, "UNIQUE LABELS", np.unique(disc_data))


        n_clusters_local = max_clust - min_clust

        data = []
        max_dim1 = max(coord[:,1])
        max_dim2 = max(coord[:,2])
        strt_dim1 = 0
        strt_dim2 = 0

        #1 subtracted to separate No Data from areas that have cluster value 0.
        data = np.zeros((((int)(max_dim1)+1+pixel_padding), ((int)(max_dim2)+pixel_padding+1))) - 1
        labels = np.array(labels)
        print("ASSIGNING LABELS", min_clust, max_clust)
        print(data.shape, labels.shape, coord.shape)
        for i in range(labels.shape[0]):
            data[coord[i,1], coord[i,2]] = labels[i]
            #print(data.shape, coord[i,1], coord[i,2], labels[i], max_dim1, max_dim2)

        print("FINISHED WITH LABEL ASSIGNMENT")
        print("FINAL DATA TO DASK")
        data = data.astype(np.float32)
        print(data)
        data = (data/1000.0).astype(np.float32)
        print(data)
        data2 = da.from_array(data)
        #del data

        da.to_zarr(data2,output_basename + "_" + str(n_clusters_local) + "clusters.zarr", overwrite=True)
        img = plt.imshow(data, vmin=-1, vmax=max_clust)
        print("HERE CLUSTERS MIN MAX MEAN STD", data.min(), data.max(), data.mean(), data.std(), data.shape)
        cmap = ListedColormap(CMAP_COLORS[0:int(max_clust - (-1) + 1)])
        img.set_cmap(cmap)
        plt.colorbar()
        plt.savefig(output_basename + "_" + str(n_clusters_local) + "clusters.png", dpi=400, bbox_inches='tight')
        plt.clf()

        file_ext = ".no_geo"
        fname = output_basename + "_" + str(n_clusters_local) + "clusters" + file_ext + ".tif"

        print("HERE", data.min(), data.max(), data.mean(), data.std(), data.shape)
        out_ds = gdal.GetDriverByName("GTiff").Create(fname, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
        out_ds.GetRasterBand(1).WriteArray(data)
        out_ds.FlushCache()
        out_ds = None



def get_model(yml_conf, n_visible):

    use_wandb_logger = yml_conf["logger"]["use_wandb"]
    log_model = None
    save_dir = yml_conf["output"]["out_dir"]
    project = None
    if use_wandb_logger:
        log_model = yml_conf["logger"]["log_model"]
        save_dir = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
        project = yml_conf["logger"]["project"]
 
    heir_model_dir = os.path.join(save_dir, "full_model_heir")
    full_model_dir = os.path.join(save_dir, "full_model")

    accelerator = yml_conf["cluster"]["heir"]["training"]["accelerator"]
    devices = yml_conf["cluster"]["heir"]["training"]["devices"]
    precision = yml_conf["cluster"]["heir"]["training"]["precision"]
    max_epochs = yml_conf["cluster"]["heir"]["training"]["epochs"]
    gradient_clip_val = yml_conf["encoder"]["training"]["gradient_clip_val"]
    heir_model_tiers = yml_conf["cluster"]["heir"]["tiers"] #TODO incorporate
    heir_gauss_stdev = yml_conf["cluster"]["heir"]["gauss_noise_stdev"] #TODO incorporate
    lambda_iid = yml_conf["cluster"]["heir"]["lambda"] #TODO incorporate 

    num_classes = yml_conf["cluster"]["heir"]["num_classes"]
    min_samples = yml_conf["cluster"]["heir"]["training"]["min_samples"]

    heir_ckpt_path = os.path.join(heir_model_dir, "heir_fc.ckpt")
    ckpt_path = os.path.join(full_model_dir, "checkpoint.ckpt")

    encoder_type=None
    if "encoder_type" in yml_conf:
        encoder_type=yml_conf["encoder_type"]

    encoder = None
    if encoder_type is not None and  encoder_type == "dbn":
        model_type = tuple(yml_conf["dbn"]["model_type"])
        dbn_arch = tuple(yml_conf["dbn"]["dbn_arch"])
        gibbs_steps = tuple(yml_conf["dbn"]["gibbs_steps"])
        normalize_learnergy = tuple(yml_conf["dbn"]["normalize_learnergy"])
        batch_normalize = tuple(yml_conf["dbn"]["batch_normalize"])
        temp = tuple(yml_conf["dbn"]["temp"])

        learning_rate = tuple(yml_conf["encoder"]["training"]["learning_rate"])
        momentum = tuple(yml_conf["encoder"]["training"]["momentum"])
        decay = tuple(yml_conf["encoder"]["training"]["weight_decay"])
        nesterov_accel = tuple(yml_conf["encoder"]["training"]["nesterov_accel"])


        save_dir_dbn = yml_conf["output"]["out_dir"]
        use_wandb_logger = yml_conf["logger"]["use_wandb"]
        if use_wandb_logger:
            save_dir_dbn = os.path.join(yml_conf["output"]["out_dir"], yml_conf["logger"]["log_out_dir"])
        encoder_dir = os.path.join(save_dir_dbn, "encoder")

        enc_ckpt_path = os.path.join(encoder_dir, "dbn.ckpt")

        encoder = DBN(model=model_type, n_visible=n_visible, n_hidden=dbn_arch, steps=gibbs_steps,
            learning_rate=learning_rate, momentum=momentum, decay=decay, temperature=temp, use_gpu=True)

        encoder.load_state_dict(torch.load(enc_ckpt_path))

        encoder.eval()
 
    model = Heir_DC(None, pretrained_model_path=ckpt_path, num_classes=num_classes, \
        encoder_type=encoder_type, encoder=encoder, clust_tree_ckpt = heir_ckpt_path)
    model.eval()

    for key in model.clust_tree["1"].keys():
        model.clust_tree["1"][key].eval()
         
    return model



def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)

    test_fnames = yml_conf["data"]["files_test"]
    train_fnames = yml_conf["data"]["files_train"]
 
    data, _  = get_prediction_dataset(yml_conf, test_fnames[0])

    model = get_model(yml_conf, data.data_full.shape[1])


    for i in range(len(test_fnames)):
        data, output_file  = get_prediction_dataset(yml_conf, test_fnames[i])
        generate_output(data, model, use_gpu, out_dir, output_fle + ".clust", conv = conv)
    for i in range(len(train_fnames)):
        data, output_file = get_prediction_dataset(yml_conf, train_fnames[i])
        generate_output(data, model, use_gpu, out_dir, output_fle + ".clust", conv = conv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)




