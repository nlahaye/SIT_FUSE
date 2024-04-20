import pytorch_lightning as pl
import torch

from sit_fuse.models.deep_cluster.dc import DeepCluster
from sit_fuse.models.deep_cluster.ijepa_dc import IJEPA_DC
from sit_fuse.models.deep_cluster.dbn_dc import DBN_DC
from sit_fuse.datasets.dataset_utils import get_prediction_dataset




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

        with torch.no_grad()
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



def main(yml_fpath):

    yml_conf = read_yaml(yml_fpath)

    num_loader_workers = int(yml_conf["data"]["num_loader_workers"])
    val_percent = int(yml_conf["data"]["val_percent"])
    batch_size = yml_conf["cluster"]["training"]["batch_size"]
    use_gpu = yml_conf["encoder"]["training"]["use_gpu"]
    out_dir = yml_conf["output"]["out_dir"]

    model = None
    if "encoder_type" in yml_conf:
        if yml_conf["encoder_type"] == "dbn":
            model = dc_DBN.load_from_checkpoint(yml_conf["model_checkpoint"])
        elif yml_conf["encoder_type"] == "ijepa":
            model = dc_IJEPA.load_from_checkpoint(yml_conf["model_checkpoint"])
    else:
        model = DeepCluster.load_from_checkpoint(yml_conf["model_checkpoint"])

    test_fnames = yml_conf["data"]["files_test"]
    train_fnames = yml_conf["data"]["files_train"]

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




