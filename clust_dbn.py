"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""




class ClustDBN():

    def __init__(self, dbn_trunk, input_fc , n_classes):
        self.dbn_trunk = dbn_trunk
        self.input_fc = input_fc
        self.n_classes = n_classes

        #fc = nn.Linear(input_fc , n_classes)
        number_heads = 1 #TODO try out multi
        fc = MultiPrototypes(self, input_fc, n_classes, number_heads)
        fc.to(self.dbn_trunk.torch_device)

        # Cross-Entropy loss is used for the discriminative fine-tuning
        #criterion = nn.CrossEntropyLoss()
 

        #TODO configurable? What does FaceBook and IID paper do, arch-wise?
        # Creating the optimzers
        optimizer = [
            #optim.Adam(model.parameters(), lr=0.0001), TODO Test altering all layers? Last DBN Layer? Only Head?
            optim.Adam(fc.parameters(), lr=0.001),
        ]
 
        # Creating training and validation batches
        train_batch = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0)
        val_batch = DataLoader(test, batch_size=10000, shuffle=False, num_workers=0)

        # For amount of fine-tuning epochs
        for e in range(fine_tune_epochs):
            print(f"Epoch {e+1}/{fine_tune_epochs}")

            # Resetting metrics
            train_loss, val_acc = 0, 0

            # For every possible batch
            for x_batch, _ in tqdm(train_batch):
                # For every possible optimizer
                for opt in optimizer:
                    # Resets the optimizer
                    opt.zero_grad()

                x2 = scipy.ndimage.gaussian_filter1d(x_batch,3)
                x3 = scipy.ndimage.gaussian_filter1d(x_batch,6)

                # Checking whether GPU is avaliable and if it should be used
                if model.device == "cuda":
                    # Applies the GPU usage to the data and labels
                    x_batch = x_batch.to(self.dbn_trunk.torch_device, non_blocking = True)
                    x2 = x2.to(self.dbn_trunk.torch_device, non_blocking = True)
                    x3 = x3.to(self.dbn_trunk.torch_device, non_blocking = True)
                   

                

                # Passing the batch down the model
                y = model(x_batch)
                y2 = model(x2)
                y3 = model(x3)

                # Reshaping the outputs
                y = y.reshape(
                    x_batch.size(0), input_fc)
                y2 = y2.reshape(
                    x_batch.size(0), input_fc)
                y3 = y3.reshape(
                    x_batch.size(0), input_fc)

                # Calculating the fully-connected outputs
                y = fc(y)
                y2 = fc(y2)
                y3 = fc(y2)
          
                temperature = 1 #TODO toggle
                # Calculating loss
                for h in number_prototypes:
                    loss += IID_segmentation_loss(y, y2, y3) #criterion(y, y_batch)
                loss /= number_prototypes

                # Propagating the loss to calculate the gradients
                loss.backward()

                # For every possible optimizer
                for opt in optimizer:
                    # Performs the gradient update
                    opt.step()

                # Adding current batch loss
                train_loss += loss.item()

            """
            # Calculate the test accuracy for the model:
            for x_batch, y_batch in tqdm(val_batch):
                x2 = scipy.ndimage.gaussian_filter1d(x_batch,3)
                x3 = scipy.ndimage.gaussian_filter1d(x_batch,6)

                # Checking whether GPU is avaliable and if it should be used
                if model.device == "cuda":
                    # Applies the GPU usage to the data and labels
                    x_batch = x_batch.to(self.dbn_trunk.torch_device, non_blocking = True)
                    x2 = x2.to(self.dbn_trunk.torch_device, non_blocking = True)
                    x3 = x3.to(self.dbn_trunk.torch_device, non_blocking = True)




                # Passing the batch down the model
                y = model(x_batch)
                y2 = model(x2)
                y3 = model(x3)

                # Reshaping the outputs
                y = y.reshape(
                    x_batch.size(0), input_fc)
                y2 = y2.reshape(
                    x_batch.size(0), input_fc)
                y3 = y3.reshape(
                    x_batch.size(0), input_fc)

                # Calculating the fully-connected outputs
                y = fc(y)
                y2 = fc(y2)
                y3 = fc(y3)
                  

                # Calculating predictions
                _, preds = torch.max(y, 1)

                # Calculating validation set accuracy
                val_acc = torch.mean((torch.sum(preds == y_batch).float()) / x_batch.size(0))

            print(f"Loss: {train_loss / len(train_batch)} | Val Accuracy: {val_acc}")
            """

## Saving the fine-tuned model
#torch.save(model, "tuned_model.pth")

## Checking the model's history
#print(model.history)


#From SWAV
class MultiPrototypes(nn.Module):
    #I dont allow for variation of n_clusters in each prototype, as SWAV does
    def __init__(self, output_dim, n_classes, nmb_heads):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = nmb_heads
        for i in range(nmb_heads):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, n_classes, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return 






#From IIC
def IID_segmentation_loss(x1_outs, x2_outs, x3_outs) # all_affine2_to_1=None,
                          #all_mask_img1=None, lamb=1.0,
                          #half_T_side_dense=None,
                          #half_T_side_sparse_min=None,
                          #half_T_side_sparse_max=None):
  #assert (x1_outs.requires_grad)
  #assert (x2_outs.requires_grad)
  #assert (not all_affine2_to_1.requires_grad)
  #assert (not all_mask_img1.requires_grad)

  #assert (x1_outs.shape == x2_outs.shape)

  # bring x2 back into x1's spatial frame
  #x2_outs_inv = perform_affine_tf(x2_outs, all_affine2_to_1)

  #if (half_T_side_sparse_min != 0) or (half_T_side_sparse_max != 0):
  #  x2_outs_inv = random_translation_multiple(x2_outs_inv,
  #                                            half_side_min=half_T_side_sparse_min,
  #                                            half_side_max=half_T_side_sparse_max)

  #if RENDER:
  #  # indices added to each name by render()
  #  render(x1_outs, mode="image_as_feat", name="invert_img1_")
  #  render(x2_outs, mode="image_as_feat", name="invert_img2_pre_")
  #  render(x2_outs_inv, mode="image_as_feat", name="invert_img2_post_")
  #  render(all_mask_img1, mode="mask", name="invert_mask_")

  # zero out all irrelevant patches
  #bn, k, h, w = x1_outs.shape
  #all_mask_img1 = all_mask_img1.view(bn, 1, h, w)  # mult, already float32
  #x1_outs = x1_outs * all_mask_img1  # broadcasts
  #x2_outs_inv = x2_outs_inv * all_mask_img1

  # sum over everything except classes, by convolving x1_outs with x2_outs_inv
  # which is symmetric, so doesn't matter which one is the filter
  #x1_outs = x1_outs.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w
  #x2_outs_inv = x2_outs_inv.permute(1, 0, 2, 3).contiguous()  # k, ni, h, w

  # k, k, 2 * half_T_side_dense + 1,2 * half_T_side_dense + 1
  p_i_j = F.conv2d(x1_outs, weight=x2_outs_inv, padding=(half_T_side_dense,
                                                         half_T_side_dense))
  p_i_j = F.conv1d(x1_outs, weight=x2_outs)
  p_i_j = F.conv1d(p_i_j, weight=x3_outs)
  p_i_j = p_i_j.sum(dim=2, keepdim=False).sum(dim=2, keepdim=False)  # k, k

  # normalise, use sum, not bn * h * w * T_side * T_side, because we use a mask
  # also, some pixels did not have a completely unmasked box neighbourhood,
  # but it's fine - just less samples from that pixel
  current_norm = float(p_i_j.sum())
  p_i_j = p_i_j / current_norm

  # symmetrise
  p_i_j = (p_i_j + p_i_j.t()) / 2.

  # compute marginals
  p_i_mat = p_i_j.sum(dim=1).unsqueeze(1)  # k, 1
  p_j_mat = p_i_j.sum(dim=0).unsqueeze(0)  # 1, k

  # for log stability; tiny values cancelled out by mult with p_i_j anyway
  p_i_j[(p_i_j < EPS).data] = EPS
  p_i_mat[(p_i_mat < EPS).data] = EPS
  p_j_mat[(p_j_mat < EPS).data] = EPS

  # maximise information
  loss = (-p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_i_mat) -
                    lamb * torch.log(p_j_mat))).sum()

  # for analysis only
  loss_no_lamb = (-p_i_j * (torch.log(p_i_j) - torch.log(p_i_mat) -
                            torch.log(p_j_mat))).sum()

  return loss, loss_no_lamb

