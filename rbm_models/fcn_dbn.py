"""Convolutional Deep Belief Network.
"""
import numpy as np

from GPUtil import showUtilization as gpu_usage

from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm

import learnergy.utils.exception as e
from learnergy.core import Dataset, Model
from learnergy.models.bernoulli import ConvRBM, ConvTransposeRBM
from learnergy.models.gaussian import GaussianConvRBM, GaussianConvTransposeRBM
from learnergy.utils import logging

logger = logging.get_logger(__name__)


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, filter_shape):
    """Make a 2D bilinear kernel suitable for upsampling"""

    factor = (filter_shape + 1) // 2
    if filter_shape % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:filter_shape, :filter_shape]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, filter_shape, filter_shape),
                      dtype=np.float64) #float64
    print(filt.shape, weight.shape)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float() #float()




class FCDBN(Model):
    """A FCDBN class provides the basic implementation for Fully Convolutional (FC) DBNs.

    References:
        H. Lee, et al.
        Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations.
        Proceedings of the 26th annual international conference on machine learning (2009).

    """

    def __init__(
        self,
        model: Optional[str] = "bernoulli",
        visible_shape: Optional[Tuple[int, int]] = (28, 28),
        #filter_shape: Optional[Tuple[Tuple[int, int], ...]] = ((7, 7),),
        #n_filters: Optional[Tuple[int, ...]] = (16,),
        n_channels: Optional[int] = 1,
        steps: Optional[Tuple[int, ...]] = (1,),
        learning_rate: Optional[Tuple[float, ...]] = (0.1,),
        momentum: Optional[Tuple[float, ...]] = (0.0,),
        decay: Optional[Tuple[float, ...]] = (0.0,),
        use_gpu: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            model: Indicates which type of Conv/ConvTranspose RBM should be used to compose the DBN.
            visible_shape: Shape of visible units.
            #filter_shape: Shape of filters per layer.
            #n_filters: Number of filters per layer.
            n_channels: Number of channels.
            steps: Number of Gibbs' sampling steps per layer.
            learning_rate: Learning rate per layer.
            momentum: Momentum parameter per layer.
            decay: Weight decay used for penalization per layer.
            use_gpu: Whether GPU should be used or not.

        """

        logger.info("Overriding class: Model -> FCDBN.")

        super(FCDBN, self).__init__(use_gpu=use_gpu)

        # Shape of visible units
        self.visible_shape = visible_shape
        self.current_layer_samples = None
        self.mse = []

        # Shape of filters
        #self.filter_shape = filter_shape

        # Number of filters
        #self.n_filters = n_filters

        # Number of channels
        self.n_channels = n_channels

        # Number of layers
        self.n_layers = 28 #TODO Fix

        # Number of steps Gibbs' sampling steps
        self.steps = steps

        # Learning rate
        self.lr = learning_rate

        # Momentum parameter
        self.momentum = momentum

        # Weight decay
        self.decay = decay

        # List of models (RBMs)
        #self.models = []
    
        conv = None
        contT = None  
        #TODO fix 
        #if model == "bernoulli":
        #    conv = ConvRBM
        #    convT = ConvTransposeRBM
        #elif model == "gaussian":
        conv = GaussianConvRBM
        convT = GaussianConvTransposeRBM


        self.output_shapes = []
        print("HERE VISIBLE SHAPE", visible_shape)
        # conv1
        self.conv1_1 = conv(visible_shape,
            (3,3), 64, n_channels, 1, 100, self.steps[0], self.lr[0],
            self.momentum[0], self.decay[0], False, None, use_gpu) 

        #[(I - F +2 *P) / S] +1 x D
        v = np.array(self.conv1_1.hidden_shape, dtype = np.int16)
        print("HERE VISIBLE SHAPE2", v) 
        self.output_shapes.append(v)
        c1_v = v

        self.conv1_2 = conv(v,
            (3,3), 64, 64, 1, 1, self.steps[1], self.lr[1],
            self.momentum[1], self.decay[1], False, None, use_gpu)

        #[(I - F) / S] + 1 x D
        v = np.ceil(np.add(np.divide(np.subtract(self.conv1_2.hidden_shape,2),2),1)).astype(np.int16)
        print("HERE VISIBLE SHAPE3", v)   
        self.output_shapes.append(v)

        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = conv(v, 
            (3,3), 128, 64, 1, 1, self.steps[2], self.lr[2],
            self.momentum[2], self.decay[2], False, None, use_gpu)

        v = np.array(self.conv2_1.hidden_shape, dtype = np.int16) #np.add(np.divide(np.add(np.subtract(v,3),2),1),1)
        print("HERE VISIBLE SHAPE4", v)   
        self.output_shapes.append(v)

        self.conv2_2 = conv(v,
            (3,3), 128, 128, 1, 1, self.steps[3], self.lr[3],
            self.momentum[3], self.decay[3], False, None, use_gpu)
       
        v = np.ceil(np.add(np.divide(np.subtract(self.conv2_2.hidden_shape,2),2),1)).astype(np.int16)
        print("HERE VISIBLE SHAPE5", v)   
        self.output_shapes.append(v)

        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = conv(v,
            (3,3), 256, 128, 1, 1, self.steps[4], self.lr[4],
            self.momentum[4], self.decay[4], False, None, use_gpu)

        v = np.array(self.conv3_1.hidden_shape, dtype = np.int16)
        print("HERE VISIBLE SHAPE6", v)   
        self.output_shapes.append(v)

        self.conv3_2 = conv(v,
            (3,3), 256, 256, 1, 1, self.steps[5], self.lr[5],
            self.momentum[5], self.decay[5], False, None, use_gpu)
 
        v = np.array(self.conv3_2.hidden_shape, dtype = np.int16)
        print("HERE VISIBLE SHAPE7", v)   
        self.output_shapes.append(v)

        self.conv3_3 = conv(v,
            (3,3), 256, 256, 1, 1, self.steps[6], self.lr[6],
            self.momentum[6], self.decay[6], True, None, use_gpu)
  
        v = np.ceil(np.add(np.divide(np.subtract(self.conv3_3.hidden_shape,2),2),1)).astype(np.int16)
        print("HERE VISIBLE SHAPE8", v)   
        self.output_shapes.append(v)

        self.pool3 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8
        v_pool3 = v

        ## conv4
        self.conv4_1 = conv(v,
            (3,3), 512, 256, 1, 1, self.steps[7], self.lr[7],
            self.momentum[7], self.decay[7], False, None, use_gpu)

        v =  np.array(self.conv4_1.hidden_shape, dtype = np.int16)
        print("HERE VISIBLE SHAPE9", v)   
        self.output_shapes.append(v)

        self.conv4_2 = conv(v,
            (3,3), 512, 512, 1, 1, self.steps[8], self.lr[8],
            self.momentum[8], self.decay[8], False, None, use_gpu)
 
        v = np.array(self.conv4_2.hidden_shape, dtype = np.int16)
        print("HERE VISIBLE SHAPE10", v)   
        self.output_shapes.append(v)

        self.conv4_3 = conv(v,
            (3,3), 512, 512, 1, 1, self.steps[9], self.lr[9],
            self.momentum[9], self.decay[9], True, None, use_gpu)

        v = np.ceil(np.add(np.divide(np.subtract(self.conv4_3.hidden_shape,2),2),1)).astype(np.int16)
        print("HERE VISIBLE SHAPE11", v)   
        self.output_shapes.append(v)
        self.pool4 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16
        v_pool4 = v

        # conv5
        self.conv5_1 = conv(v,
            (3,3), 512, 512, 1, 1, self.steps[10], self.lr[10],
            self.momentum[10], self.decay[10], False, None, use_gpu)

        v = np.array(self.conv5_1.hidden_shape, dtype = np.int16)
        print("HERE VISIBLE SHAPE12", v)   
        self.output_shapes.append(v)

        self.conv5_2 = conv(v,
            (3,3), 512, 512, 1, 1, self.steps[11], self.lr[11],
            self.momentum[11], self.decay[11], False, None, use_gpu)
 
        v = np.array(self.conv5_2.hidden_shape, dtype = np.int16)
        print("HERE VISIBLE SHAPE13", v)   
        self.output_shapes.append(v)

        self.conv5_3 = conv(v,
            (3,3), 512, 512, 1, 1, self.steps[12], self.lr[12],
            self.momentum[12], self.decay[12], True, None, use_gpu)

        v = np.ceil(np.add(np.divide(np.subtract(self.conv5_3.hidden_shape,2),2),1)).astype(np.int16)
        print("HERE VISIBLE SHAPE14", v)   
        self.output_shapes.append(v)

        self.pool5 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

#[(I - F +2 *P) / S] +1 x D 

        filter_size = np.ceil(np.divide(c1_v, 32.0)).astype(np.int16)
        # fc6
        self.fc6 = conv(v,
            filter_size, 4096, 512, 1, 0, self.steps[13], self.lr[13],
            self.momentum[13], self.decay[13], False, None, use_gpu)

        v = np.array(self.fc6.hidden_shape, dtype = np.int16)
        print("HERE VISIBLE SHAPE15", v)   
        self.output_shapes.append(v)
        self.drop6 = torch.nn.Dropout2d()
 
        # fc7
        self.fc7 = conv(v,
            (1,1), 4096, 4096, 1, 0, self.steps[14], self.lr[14],
            self.momentum[14], self.decay[14], False, None, use_gpu)

        v = np.array(self.fc7.hidden_shape, dtype = np.int16)
        print("HERE VISIBLE SHAPE16", v)   
        self.output_shapes.append(v)
        self.drop7 = torch.nn.Dropout2d()


        #N Filters here is number of final classes - experimental - can make configurable
        self.score_fr = conv(v,
            (1,1), 500, 4096, 1, 0, self.steps[15], self.lr[15],
            self.momentum[15], self.decay[15], False, None, use_gpu)
     
        v = np.array(self.score_fr.hidden_shape, dtype = np.int16)
        print("HERE VISIBLE SHAPE17", v)   
        self.output_shapes.append(v)

        self.upscore2 = convT(v,
            (4,4), 500, 500, 2, 0, self.steps[16], self.lr[16],
            self.momentum[16], self.decay[16], use_gpu) #False, None, use_gpu)        

 
        v = np.array(self.upscore2.hidden_shape, dtype = np.int16)
        print("HERE VISIBLE SHAPE18", v)   
        self.output_shapes.append(v)

        self.score_pool4 = conv(v_pool4,
            (1,1), 500, 512, 1, 0, self.steps[17], self.lr[17],
            self.momentum[17], self.decay[17], use_gpu) #False, None, use_gpu)
        self.output_shapes.append(v_pool4) 

        self.upscore_pool4 = convT(v, 
            (4,4), 500, 500, 2, 0, self.steps[18], self.lr[18],
            self.momentum[18], self.decay[18], use_gpu) #False, None, use_gpu)
  
        v = np.array(self.upscore_pool4.hidden_shape, dtype = np.int16)
        print("HERE VISIBLE SHAPE19", v)   
        self.output_shapes.append(v)

        self.score_pool3 = conv(v_pool3,
            (1,1), 500, 256, 1, 0, self.steps[19], self.lr[19],
            self.momentum[19], self.decay[19], False, None, use_gpu)
        self.output_shapes.append(v_pool3)

        self.upscore8 = convT(v,
            (16,16), 500, 500, 8, 0, self.steps[20], self.lr[20],
            self.momentum[20], self.decay[20], use_gpu) #False, None, use_gpu)
        self.output_shapes.append(v)

        self._initialize_weights()

 
        self.models = [self.conv1_1, self.conv1_2, self.pool1, self.conv2_1, 
            self.conv2_2, self.pool2, self.conv3_1, self.conv3_2, self.conv3_3, 
            self.pool3, self.conv4_1, self.conv4_2, self.conv4_3, self.pool4, self.conv5_1, 
            self.conv5_2, self.conv5_3, self.pool5, self.fc6, self.fc7, 
            self.score_fr, self.upscore2, self.score_pool4, self.upscore_pool4,
            self.score_pool3, self.upscore8]  
        #self.models = [self.conv1_1, self.conv1_2, self.pool1, self.conv2_1, 
        #    self.conv2_2, self.pool2, self.conv3_1, self.conv3_2, self.conv3_3,
        #    self.pool3, self.conv5_1,
        #    self.conv5_2, self.conv5_3, self.pool5, self.fc6, self.fc7,
        #    self.score_fr, self.upscore2, self.score_pool4, self.upscore_pool4,
        #    self.score_pool3, self.upscore8]


        self.to(self.torch_device, non_blocking=True, memory_format=torch.channels_last)

        logger.info("Class overrided.")


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ConvRBM):
                m.W.data.zero_()
                #m.W.data = m.W.data.half()
                if m.b is not None:
                    m.b.data.zero_()
                    #m.b.data = m.b.data.half()
                if m.a is not None:
                    m.a.data.zero_()
                    #m.a.data = m.a.data.half()
            if isinstance(m, ConvTransposeRBM):
                assert m.filter_shape[0] == m.filter_shape[1]
                initial_weight = get_upsampling_weight(
                    m.n_channels, m.n_filters, m.filter_shape[0])
                m.W.data.copy_(initial_weight)
                if m.b is not None:
                    m.b.data.zero_()
                    #m.b.data = m.b.data.half()
                if m.a is not None:
                    m.a.data.zero_()
                    #m.a.data = m.a.data.half()
 
            

                #m.W = m.W.data.to(self.torch_device, non_blocking=True) 
 
    @property
    def visible_shape(self) -> Tuple[int, int]:
        """Shape of visible units."""

        return self._visible_shape

    @visible_shape.setter
    def visible_shape(self, visible_shape: Tuple[int, int]) -> None:
        self._visible_shape = visible_shape

    #@property
    #def filter_shape(self) -> Tuple[Tuple[int, int], ...]:
    #    """Shape of filters."""
    #
    #    return self._filter_shape

    #@filter_shape.setter
    #def filter_shape(self, filter_shape: Tuple[Tuple[int, int], ...]) -> None:
    #    self._filter_shape = filter_shape

    #@property
    #def n_filters(self) -> Tuple[int, ...]:
    #    """Number of filters."""
    #
    #    return self._n_filters

    #@n_filters.setter
    #def n_filters(self, n_filters: Tuple[int, ...]) -> None:
    #    self._n_filters = n_filters

    @property
    def n_channels(self) -> int:
        """Number of channels."""

        return self._n_channels

    @n_channels.setter
    def n_channels(self, n_channels: int) -> None:
        if n_channels <= 0:
            raise e.ValueError("`n_channels` should be > 0")

        self._n_channels = n_channels

    @property
    def n_layers(self) -> int:
        """Number of layers."""

        return self._n_layers

    @n_layers.setter
    def n_layers(self, n_layers: int) -> None:
        if n_layers <= 0:
            raise e.ValueError("`n_layers` should be > 0")

        self._n_layers = n_layers

    @property
    def steps(self) -> Tuple[int, ...]:
        """Number of steps Gibbs' sampling steps per layer."""

        return self._steps

    @steps.setter
    def steps(self, steps: Tuple[int, ...]) -> None:
        if len(steps) != self.n_layers:
            raise e.SizeError(f"`steps` should have size equal as {self.n_layers}")

        self._steps = steps

    @property
    def lr(self) -> Tuple[float, ...]:
        """Learning rate per layer."""

        return self._lr

    @lr.setter
    def lr(self, lr: Tuple[float, ...]) -> None:
        if len(lr) != self.n_layers:
            raise e.SizeError(f"`lr` should have size equal as {self.n_layers}")

        self._lr = lr

    @property
    def momentum(self) -> Tuple[float, ...]:
        """Momentum parameter per layer."""

        return self._momentum

    @momentum.setter
    def momentum(self, momentum: Tuple[float, ...]) -> None:
        if len(momentum) != self.n_layers:
            raise e.SizeError(f"`momentum` should have size equal as {self.n_layers}")

        self._momentum = momentum

    @property
    def decay(self) -> Tuple[float, ...]:
        """Weight decay per layer."""

        return self._decay

    @decay.setter
    def decay(self, decay: Tuple[float, ...]) -> None:
        if len(decay) != self.n_layers:
            raise e.SizeError(f"`decay` should have size equal as {self.n_layers}")

        self._decay = decay

    @property
    def models(self) -> List[torch.nn.Module]:
        """List of models (RBMs)."""

        return self._models

    @models.setter
    def models(self, models: List[torch.nn.Module]) -> None:
        self._models = models

    def fit(
        self,
        dataset: Union[torch.utils.data.Dataset, Dataset],
        batch_size: Optional[int] = 128,
        epochs: Optional[Tuple[int, ...]] = (10, 10),
        is_distributed: Optional[bool] = False,
        num_loader_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = False
    ) -> float:
        """Fits a new FCDBN model.

        Args:
            dataset: A Dataset object containing the training data.
            batch_size: Amount of samples per batch.
            epochs: Number of training epochs per layer.
            is_distributed: Whether or not implementation is using PyTorch Distributed Data Parallel (DDP)
            num_loader_workers: Number of workers to be used for DataLoader.
            pin_memory: Whether or not to use pinned memory with the DataLoader

        Returns:
            (float): MSE (mean squared error) from the training step.

        """

        # Checking if the length of number of epochs' list is correct
        #if len(epochs) != self.n_layers:
        #    # If not, raises an error
        #    raise e.SizeError(("`epochs` should have size equal as %d", self.n_layers))

        # Initializing MSE as a list
        mse = []
        self.n_samples = dataset.data.shape[0]

        print("PRE TRAIN")
        gpu_usage()
        dt = torch.float16
        if self.device == "cpu":
            dt = torch.bfloat16 
        print(epochs)
        logger.info("Fitting layer conv1_1")
        self.fit_layer(self.conv1_1, dataset, None, epochs[0], batch_size, 0, mse, num_loader_workers, pin_memory, is_distributed)
        #Data is on GPU, cant use multiple workers
        num_loader_workers = 0
        pin_memory = False
      
        dataset.data = dataset.data.cpu()
        dataset.targets = dataset.targets.cpu()
        dataset.data = dataset.data.detach()
        dataset.targets = dataset.targets.detach()
        torch.cuda.empty_cache()
        print("LAYER 1")
        gpu_usage()
        self.fit_layer(self.conv1_2, self.current_layer_samples, None, epochs[1], batch_size, 1, mse, num_loader_workers, pin_memory, is_distributed)
        self.current_layer_samples = TensorDataset(self.pool1(self.current_layer_samples.tensors[0].to(self.torch_device, non_blocking=True, memory_format=torch.channels_last)), self.current_layer_samples.tensors[1])
        self.current_layer_samples = TensorDataset(self.current_layer_samples.tensors[0].cpu().detach(), self.current_layer_samples.tensors[1].cpu().detach())

        print("LAYER 2")
        gpu_usage()
        self.fit_layer(self.conv2_1, self.current_layer_samples, None, epochs[2], batch_size, 2, mse, num_loader_workers, pin_memory, is_distributed)  
        print("LAYER 3")
        gpu_usage()
        self.fit_layer(self.conv2_2, self.current_layer_samples, None, epochs[3], batch_size, 3, mse, num_loader_workers, pin_memory, is_distributed)
        self.current_layer_samples = TensorDataset(self.pool2(self.current_layer_samples.tensors[0].to(self.torch_device, non_blocking=True, memory_format=torch.channels_last)), self.current_layer_samples.tensors[1])
        self.current_layer_samples = TensorDataset(self.current_layer_samples.tensors[0].cpu().detach(), self.current_layer_samples.tensors[1].cpu().detach())

        print("LAYER 4")
        gpu_usage()
        self.fit_layer(self.conv3_1, self.current_layer_samples, None, epochs[4], batch_size, 4, mse, num_loader_workers, pin_memory, is_distributed)
        print("LAYER 5")
        gpu_usage()
        self.fit_layer(self.conv3_2, self.current_layer_samples, None, epochs[5], batch_size, 5, mse, num_loader_workers, pin_memory, is_distributed)
        print("LAYER 6")
        gpu_usage()
        self.fit_layer(self.conv3_3, self.current_layer_samples, None, epochs[6], batch_size, 6, mse, num_loader_workers, pin_memory, is_distributed)

        self.current_layer_samples = TensorDataset(self.pool3(self.current_layer_samples.tensors[0].to(self.torch_device, non_blocking=True, memory_format=torch.channels_last)), self.current_layer_samples.tensors[1])
        self.current_layer_samples = TensorDataset(self.current_layer_samples.tensors[0].cpu().detach(), self.current_layer_samples.tensors[1].cpu().detach())
        #dataset3 = TensorDataset(self.current_layer_samples.tensors[0], self.current_layer_samples.tensors[1])
        print("LAYER 7")
        gpu_usage()
        self.fit_layer(self.conv4_1, dataset3, None, epochs[7], batch_size, 7, mse, num_loader_workers, pin_memory, is_distributed)
        dataset3 = TensorDataset(dataset3.tensors[0].cpu().detach(), dataset3.tensors[1].cpu().detach())  
        torch.cuda.empty_cache()
        dist.barrier() 
        print("LAYER 8")
        gpu_usage()
        self.fit_layer(self.conv4_2, self.current_layer_samples, None, epochs[8], batch_size, 8, mse, num_loader_workers, pin_memory, is_distributed)
        self.fit_layer(self.conv4_3, self.current_layer_samples, None, epochs[9], batch_size, 9, mse, num_loader_workers, pin_memory, is_distributed)

        self.current_layer_samples = TensorDataset(self.pool4(self.current_layer_samples.tensors[0].to(self.torch_device, non_blocking=True, memory_format=torch.channels_last)), self.current_layer_samples.tensors[1])
        self.current_layer_samples = TensorDataset(self.current_layer_samples.tensors[0].cpu().detach(), self.current_layer_samples.tensors[1].cpu().detach())
        dataset4 = TensorDataset(self.current_layer_samples.tensors[0], self.current_layer_samples.tensors[1])
 
        torch.cuda.empty_cache()
        dist.barrier()
        self.fit_layer(self.conv5_1, self.current_layer_samples, None, epochs[10], batch_size, 10, mse, num_loader_workers, pin_memory, is_distributed)

        self.fit_layer(self.conv5_2, self.current_layer_samples, None, epochs[11], batch_size, 11, mse, num_loader_workers, pin_memory, is_distributed)

        self.fit_layer(self.conv5_3, self.current_layer_samples, None, epochs[12], batch_size, 12, mse, num_loader_workers, pin_memory, is_distributed)

        self.current_layer_samples = TensorDataset(self.pool5(self.current_layer_samples.tensors[0].to(self.torch_device, non_blocking=True, memory_format=torch.channels_last)), self.current_layer_samples.tensors[1])
        self.current_layer_samples = TensorDataset(self.current_layer_samples.tensors[0].cpu().detach(), self.current_layer_samples.tensors[1].cpu().detach())
        dataset4 = TensorDataset(self.current_layer_samples.tensors[0], self.current_layer_samples.tensors[1])

        self.fit_layer(self.fc6, self.current_layer_samples, None, epochs[13], batch_size, 13, mse, num_loader_workers, pin_memory, is_distributed)
        self.fit_layer(self.fc7, self.current_layer_samples, None, epochs[14], batch_size, 14, mse, num_loader_workers, pin_memory, is_distributed)
        self.fit_layer(self.score_fr, self.current_layer_samples, None, epochs[15], batch_size, 15, mse, num_loader_workers, pin_memory, is_distributed)
        self.fit_layer(self.upscore2, self.current_layer_samples, None, epochs[16], batch_size, 16, mse, num_loader_workers, pin_memory, is_distributed)
        dataset5 = TensorDataset(self.current_layer_samples.tensors[0].cpu().detach(), self.current_layer_samples.tensors[1].cpu().detach())
        torch.cuda.empty_cache()
        dist.barrier()
        #dataset4.tensors[0] = dataset4.tensors[0].to(self.torch_device, non_blocking=True, memory_format=torch.channels_last)
        #dataset4.tensors[1] = dataset4.tensors[1].to(self.torch_device, non_blocking=True)
        self.fit_layer(self.score_pool4, dataset4, None, epochs[17], batch_size, 17, mse, num_loader_workers, pin_memory, is_distributed)
        del dataset4
        torch.cuda.empty_cache()
        dist.barrier()
        dataset6 = TensorDataset(self.current_layer_samples.tensors[0].cpu().detach(), self.current_layer_samples.tensors[1].cpu().detach())        
        del self.current_layer_samples
        torch.cuda.empty_cache()
        dist.barrier()
        dataset6.tensors[0] = dataset6.tensors[0] + dataset5.tensors[0] #.to(self.torch_device, non_blocking=True, memory_format=torch.channels_last)
        del dataset5
        torch.cuda.empty_cache()
        dist.barrier()        

        #TODO figure this out
        del samples_upscore2
        del samples_score_pool4 
        self.fit_layer(self.upscore_pool4, dataset6, None, epochs[18], batch_size, 18, mse, num_loader_workers, pin_memory, is_distributed)
        del dataset6
      
        #dataset3.tensors[0] = dataset3.tensors[0].to(self.torch_device, non_blocking=True, memory_format=torch.channels_last)
        #dataset3.tensors[1] = dataset3.tensors[1].to(self.torch_device, non_blocking=True)
        self.fit_layer(self.score_pool3, dataset3, None, epochs[19], batch_size, 19, mse, num_loader_workers, pin_memory, is_distributed)
        del dataset3

        samples = samples_upscore_pool4 + samples_score_pool3
        dataset2 = TensorDataset(samples, dataset2.tensors[1])
        del samples
        del samples_upscore_pool4
        del samples_score_pool3
        self.fit_layer(self.upscore8, dataset2, None, epochs[20], batch_size, 20, mse, num_loader_workers, pin_memory, is_distributed)
        
        return self.mse[-1]

    def fit_layer(
        self,
        model,
        dataset,
        transform,
        epochs,
        batch_size,
        i,
        mse,
        num_loader_workers = 0,
        pin_memory = False,
        is_distributed = True
    ): 
        # For every possible model (ConvRBM)
        logger.info("Fitting layer %d/%d ...", i + 1, self.n_layers)


        loader = None
        sampler = None
        if is_distributed:
            sampler = DistributedSampler(dataset, shuffle=True)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                sampler=sampler, num_workers = num_loader_workers, pin_memory =pin_memory,
                drop_last=True)

        print("INIT USAGE FIT_LAYER")
        gpu_usage()
        # Fits the RBM
        model_mse = model.fit(dataset, batch_size, epochs, loader, sampler).item()

        print("0 USAGE FIT_LAYER")
        gpu_usage()
        #Synchronizes all processes
        dist.barrier()
        #Only run sample generation with one task
        samples = None
        # Appending the metrics
        self.mse.append(model_mse)
    
        if i < len(self.models)-1:

            with torch.no_grad():

                # If the dataset has a transform
                if transform:
                    if isinstance(dataset, TensorDataset):
                        samples = dataset.tensors[0]
                    else:
                        # Applies the transform over the samples
                        samples = transform(dataset.data)

                # If there is no transform
                else:
                    if isinstance(dataset, TensorDataset):
                        samples = dataset.tensors[0]
                    else:
                        # Just gather the samples
                        samples = dataset.data


                    # Gathers the transform callable from current dataset
                    transform = None

                    print("HERE PRE BROADCAST1", samples.shape) #samples.min(), samples.max(), samples.mean(), samples.std())
                #dist.barrier()
                print("1 USAGE FIT_LAYER")
                gpu_usage()
                dt = torch.float16
                if self.device == "cpu":
                    dt = torch.bfloat16
                with torch.autocast(device_type=self.device, dtype=dt):
           
                    # Reshape the samples into an appropriate shape
                    if self.local_rank == 0:
                        samples = samples.reshape(
                            self.n_samples,
                            model.n_channels,
                            model.visible_shape[0],
                            model.visible_shape[1],
                        )
                    #else:
                    #    samples = torch.zeros((self.n_samples,model.n_filters,self.output_shapes[i][0], self.output_shapes[i][1]))
 
                    print("HERE SAMPLES 1", samples.shape)
                    samples = samples.to(self.torch_device, non_blocking=True, memory_format=torch.channels_last) #dtype=dt
                    print("2 USAGE FIT_LAYER")
                    gpu_usage()
 
                    # TODO? if self.local_rank == 0:
              
                    # Performs a forward pass over the samples to get their probabilities
                    samples, samples2 = model.hidden_sampling(samples)
                    samples2 = samples2.detach().cpu()
                    del samples2

                    print("1 USAGE FIT_LAYER")
                    gpu_usage()
                    # Detaches the variable from the computing graph
                    #samples = samples.detach()
                    print("HERE PRE BROADCAST2", samples.shape) #, samples.min(), samples.max(), samples.mean(), samples.std())                

       
                    # TODO?dist.broadcast(samples, src=0, async_op=False)
                    samples = samples.detach()
                    torch.cuda.empty_cache()
                #Synchronize again after sample generation
                dist.barrier()         
 
                print("HERE POST BROADCAST", samples.shape, self.local_rank)
                print("HERE POST BROADCAST2") #, samples.min(), samples.max(), samples.mean(), samples.std())
                if isinstance(dataset, TensorDataset):
                    targets = dataset.tensors[1]
                else:
                    targets = dataset.targets


                del self.current_layer_samples 
                torch.cuda.empty_cache()
                self.current_layer_samples = TensorDataset(samples.cpu().detach(), targets.cpu().detach()) 
            del targets
            del samples
            del dataset
            model.W.detach()
            model.a.detach()
            model.b.detach()
        torch.cuda.empty_cache()
        print("HERE AFTER CACHE EMPTY")
        dist.barrier()
        #return mse, samples

    def reconstruct(
        self, dataset: torch.utils.data.Dataset,
        batches: Optional[torch.utils.data.DataLoader] = None
    ) -> Tuple[float, torch.Tensor]:
        """Reconstructs batches of new samples.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the training data.
            batches: DataLoader to be used. If None (default), DataLoader is initialized in function.

        Returns:
            (Tuple[float, torch.Tensor]): Reconstruction error and visible probabilities, i.e., P(v|h).

        """

        logger.info("Reconstructing new samples ...")

        # Resetting MSE to zero
        mse = 0

        # Defining the batch size as the amount of samples in the dataset
        batch_size = len(dataset)

        # Transforming the dataset into training batches
        if batches is None:
            batches = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

        # For every batch
        for samples, _ in tqdm(batches):
            # Flattening the samples' batch
            samples = samples.reshape(
                len(samples),
                self.n_channels,
                self.visible_shape[0],
                self.visible_shape[1],
            )

            samples = samples.to(self.torch_device, non_blocking=True, memory_format=torch.channels_last)

            # Applying the initial hidden probabilities as the samples
            hidden_probs = samples

            # For every possible model (CRBM)
            for model in self.models:
                # Performing a hidden layer sampling
                hidden_probs, _ = model.hidden_sampling(hidden_probs)

            # Applying the initial visible probabilities as the hidden probabilities
            visible_probs = hidden_probs

            # For every possible model (CRBM)
            #for model in reversed(self.models):
            #    # Performing a visible layer sampling
            #    visible_probs, visible_states = model.visible_sampling(visible_probs)

            visible_probs, visible_states = self.upscore8.visible_sampling(visible_probs)
            #Oversimplification - as on forward 
            #h = upscore_pool4 + score_pool3c
            #score_pool3c = h 
            #h = h[:, :,
            #  9:9 + upscore_pool4.size()[2],
            #  9:9 + upscore_pool4.size()[3]]
            visible_probs_pool3, visible_states = self.score_pool3.visible_sampling(visible_probs)
            #upscore_pool4 = h
            visible_probs, visible_states = self.upscore_pool4.visible_sampling(visible_probs)
            #h = upscore2 + score_pool4c
            #score_pool4c = h
            #h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
            visible_probs_pool4, visible_states = self.score_pool4.visible_sampling(visible_probs)
            #upscore2 = h
            visible_probs, visible_states = self.upscore2.visible_sampling(visible_probs)
            visible_probs, visible_states = self.score_fr.visible_sampling(visible_probs) 
            #h = self.drop7(h)
            visible_probs, visible_states = self.fc7.visible_sampling(visible_probs)
            #h = self.drop6(h)
            visible_probs, visible_states = self.fc6.visible_sampling(visible_probs)
            ##h = self.pool5(h)
            visible_probs, visible_states = self.conv5_3.visible_sampling(visible_probs)
            visible_probs, visible_states = self.conv5_2.visible_sampling(visible_probs)
            visible_probs, visible_states = self.conv5_1.visible_sampling(visible_probs)
            ##h = self.pool4(h)
            #pool4 = h 
            visible_probs, visible_states = self.conv4_3.visible_sampling(visible_probs)
            visible_probs, visible_states = self.conv4_2.visible_sampling(visible_probs)
            visible_probs, visible_states = self.conv4_1.visible_sampling(visible_probs)
            ##h = self.pool3(h)
            #pool3 = h 
            visible_probs, visible_states = self.conv3_3.visible_sampling(visible_probs)
            visible_probs, visible_states = self.conv3_2.visible_sampling(visible_probs)
            visible_probs, visible_states = self.conv3_1.visible_sampling(visible_probs)
            ##h = self.pool2(h)
            visible_probs, visible_states = self.conv2_2.visible_sampling(visible_probs)
            visible_probs, visible_states = self.conv2_1.visible_sampling(visible_probs)
            ##h = self.pool1(h)
            visible_probs, visible_states = self.conv1_2.visible_sampling(visible_probs)
            visible_probs, visible_states = self.conv1_1.visible_sampling(visible_probs)

            # Calculating current's batch reconstruction MSE
            batch_mse = torch.div(
                torch.sum(torch.pow(samples - visible_states, 2)), batch_size
            )

            samples = samples.detach()
            # Summing up to reconstruction's MSE
            mse += batch_mse.item()

        # Normalizing the MSE with the number of batches
        mse /= len(batches)

        logger.info("MSE: %f", mse)

        return mse, visible_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass over the data.

        Args:
            x: An input tensor for computing the forward pass.

        Returns:
            (torch.Tensor): A tensor containing the FCDBN's outputs.

        """
        h = x
        h, _ = self.conv1_1.hidden_sampling(h)
        h, _ = self.conv1_2.hidden_sampling(h)
        h = self.pool1(h)

        h, _ = self.conv2_1.hidden_sampling(h)
        h, _ = self.conv2_2.hidden_sampling(h)
        h = self.pool2(h)

        h, _ = self.conv3_1.hidden_sampling(h)
        h, _ = self.conv3_2.hidden_sampling(h)
        h, _ = self.conv3_3.hidden_sampling(h)
        h = self.pool3(h)
        pool3 = h  # 1/8

        h, _ = self.conv4_1.hidden_sampling(h)
        h, _ = self.conv4_2.hidden_sampling(h)
        h, _ = self.conv4_3.hidden_sampling(h)
        h = self.pool4(h)
        pool4 = h  # 1/16

        h, _ = self.conv5_1.hidden_sampling(h)
        h, _ = self.conv5_2.hidden_sampling(h)
        h, _ = self.conv5_3.hidden_sampling(h)
        h = self.pool5(h)

        h, _ = self.fc6.hidden_sampling(h)
        h = self.drop6(h)

        h, _ = self.fc7.hidden_sampling(h)
        h = self.drop7(h)

        h, _ = self.score_fr.hidden_sampling(h)
        h, _ = self.upscore2.hidden_sampling(h)
        upscore2 = h  # 1/16

        h, _ = self.score_pool4.hidden_sampling(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h, _ = self.upscore_pool4.hidden_sampling(h)
        upscore_pool4 = h  # 1/8

        h, _ = self.score_pool3.hidden_sampling(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h, _ = self.upscore8.hidden_sampling(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h
 

        # For every possible model
        #for model in self.models:
        #    # Calculates the outputs of the model
        #    x, _ = model.hidden_sampling(x)

        return h
