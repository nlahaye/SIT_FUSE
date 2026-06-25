---
description: Command examples used to inference models
---

# Inference Command

Before running the commands, we must make sure the model weights are organized.

* If you followed the previous training steps, the model weights will be located in the specified `output_dir` of the YAML. **They will already be in the correct format, so you can directly run the commands at the bottom of the page.**
* If you are using other pre-trained weights, find the folder/directory in which they are stored and follow the steps below.

The folder (e.g. `home/data/MODEL_WEIGHTS`)  should include files like:

```
encoder_scaler.pkl, dbn.ckpt, encoder.ckpt, deep_cluster.ckpt, heir_fc.ckpt
```

It also should include two `.npy` files, which, if missing, can be created with:

```
touch home/data/MODEL_WEIGHTS/train_data.indices.npy  ## replace with your path
touch home/data/MODEL_WEIGHTS/train_data.npy
```

Now modify the directory structure to look like this:

<div data-full-width="false"><figure><img src="../../.gitbook/assets/Screenshot 2025-06-30 at 10.58.19 AM.png" alt="" width="308"><figcaption></figcaption></figure></div>

The script expects the files in specific folders, so it is important to not skip this step.

* Tip: Use the `mkdir` and `mv`  commands to create this structure.

Now we can run our commands:

```
cd <path_to_SIT_FUSE>/SIT_FUSE/src/sit_fuse/inference
```

```
python3 generate_output.py -y <path_to_yaml>
# Same YAML as in the previous steps. See Code configuration
```

Outputs zarr files
