




def generate_output(dat, mdl, use_gpu, out_dir, output_fle, mse_fle, pin_mem = False, conv = False):
    output_full = None
    count = 0
    dat.current_subset = -1
    dat.next_subset()

    local_rank = 0
    if "LOCAL_RANK" in os.environ.keys():
        local_rank = int(os.environ["LOCAL_RANK"])
    if use_gpu:
        device = torch.device("cuda:{}".format(local_rank))
    else:
        device = torch.device("cpu:{}".format(local_rank))

    ind = 0
    while(count == 0 or dat.has_next_subset() or (dat.subset > 1 and dat.current_subset > (dat.subset-2))):
        output_batch_size = min(5000, max(int(dat.data_full.shape[0] / 5), dat.data_full.shape[0]))

        print("HERE GENERATING OUTPUT", output_batch_size, dat.current_subset, dat.subset)

        if count == 0:
            output_sze = dat.data_full.shape[0]
            append_remainder = int(output_batch_size - (output_sze % output_batch_size))

            if isinstance(dat.data_full,torch.Tensor):
                dat.data_full = torch.cat((dat.data_full,dat.data_full[0:append_remainder]))
                dat.targets_full = torch.cat((dat.targets_full,dat.targets_full[0:append_remainder]))
            else:
                dat.data_full = np.concatenate((dat.data_full,dat.data_full[0:append_remainder]))
                dat.targets_full = np.concatenate((dat.targets_full,dat.targets_full[0:append_remainder]))

            dat.current_subset = -1
            dat.next_subset()

        test_loader = DataLoader(dat, batch_size=output_batch_size, shuffle=False, \
        num_workers = 0, drop_last = False, pin_memory = pin_mem)
        ind = 0
        ind2 = 0
        for data in tqdm(test_loader):
            dat_dev, lab_dev = data[0].to(device=device, non_blocking=True), data[1].to(device=device, non_blocking=True)
            dev_ds = TensorDataset(dat_dev, lab_dev)

            output = mdl.forward(dat_dev)
            if isinstance(output, list):
                output = output[0] #TODO improve usage uf multi-headed output after single-headed approach validated
            output = torch.unsqueeze(torch.argmax(output, axis = 1), axis=1)

            if use_gpu == True:
                output = output.detach().cpu()
            loader = DataLoader(dev_ds, batch_size=output_batch_size, shuffle=False, \
                num_workers = 0, drop_last = False, pin_memory = False)
            dat_dev = dat_dev.detach().cpu()
            lab_dev = lab_dev.detach().cpu()
            del dev_ds

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
        if dat.has_next_subset():
            dat.next_subset()
        else:
            break
    #Save training output
    print("SAVING", os.path.join(out_dir, output_fle))
    torch.save(output_full, os.path.join(out_dir, output_fle), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(dat.targets_full, os.path.join(out_dir, output_fle + ".indices"), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(dat.data_full, os.path.join(out_dir, output_fle + ".input"), pickle_protocol=pickle.HIGHEST_PROTOCOL)


