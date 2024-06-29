





def get_deep_cluster(yml_conf, encoder, clust_visible_shape):

    auto_clust = yml_conf["deep_cluster"]
    out_dir = yml_conf["output"]["out_dir"]

    deep_cluster = DeepCluster(encoder, clust_visible_shape, auto_clust, True)
    deep_cluster.fc = DDP(deep_cluster.fc, device_ids=[local_rank], output_device=local_rank)
    final_model = deep_cluster

    return final_model

def get_deep_cluster_heirarchical(yml_conf, deep_cluster, train_data=None):

    
    out_dir = yml_conf["output"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    use_gpu = yml_conf["encoder"]["training"]["use_gpu"]
  
    model_fname = yml_conf["output"]["model"]
    model_file = os.path.join(out_dir, model_fname)
    heir_model_file = os.path.join(out_dir, "heir_" + model_fname)
    heir_model_tiers = yml_conf["clust"]["heir_tiers"]

    heir_min_samples = yml_conf["clust"]["training"]["heir_cluster_min_samples"]
    heir_gauss_stdevs = yml_conf["clust"]["training"]["heir_cluster_gauss_noise_stdev"]
    heir_epochs = yml_conf["clust"]["training"]["heir_epochs"]
    heir_tune_subtrees = yml_conf["clust"]["training"]["heir_tune_subtrees"]
    heir_tune_subtree_list = yml_conf["clust"]["training"]["heir_tune_subtree_list"]
    n_heir_classes = yml_conf["clust"]["training"]["heir_deep_cluster"]

    heir_clust = None
 
    for tiers in range(0,heir_model_tiers):

        heir_mdl_file = heir_model_file + ""
        if tiers > 0:
            heir_mdl_file = heir_model_file + "_" + str(tiers)

        if not os.path.exists(heir_mdl_file + ".ckpt") or overwrite_model:
            heir_clust = HeirClust(final_model, n_heir_classes, use_gpu=use_gpu, min_samples=heir_min_samples, gauss_stdevs = heir_gauss_stdevs, train_data)

        else:
            heir_clust = HeirClust(final_model, n_heir_classes, use_gpu=use_gpu, min_samples=heir_min_samples, gauss_stdevs = heir_gauss_stdevs, train_data)
            heir_dict = torch.load(heir_mdl_file + ".ckpt")
            heir_clust.load_model(heir_dict)

    return heir_clust




