import torch
import time
import sys

def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
  # has had softmax applied
  _, k = x_out.size()

  start_time = time.monotonic()
  p_i_j = compute_joint(x_out, x_tf_out)
  end_time = time.monotonic()

  start_time = time.monotonic()

  p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
  p_j = p_i_j.sum(dim=0).view(1, k).expand(k,
                                           k)  # but should be same, symmetric

  p_i = p_i.contiguous()
  p_j = p_j.contiguous()
  p_i_j = p_i_j.contiguous()


  # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway

  p_i_j = torch.clamp(p_i_j, min=EPS)
  p_i = torch.clamp(p_i, min=EPS)
  p_j = torch.clamp(p_j, min=EPS)


  loss = - p_i_j * (torch.log(p_i_j) \
                    - lamb * torch.log(p_j) \
                    - lamb * torch.log(p_i))

  loss = loss.sum()

  loss_no_lamb = - p_i_j * (torch.log(p_i_j) \
                            - torch.log(p_j) \
                            - torch.log(p_i))

  p_i = p_i.detach()
  del p_i
  p_j = p_j.detach()
  del p_j
  p_i_j = p_i_j.detach()
  del p_i_j

  loss_no_lamb = loss_no_lamb.sum()
  end_time = time.monotonic()

  return loss, loss_no_lamb



def compute_joint(x_out, x_tf_out):
  # produces variable that requires grad (since args require grad)

  bn, k = x_out.size()

  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
  p_i_j = p_i_j.sum(dim=0)  # k, k
  p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
  p_i_j = p_i_j / p_i_j.sum()  # normalise

  return p_i_j


