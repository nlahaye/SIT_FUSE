"""Convolutional Deep Belief Network.
"""
import numpy as np
import math

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

import copy

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




class DBNUnetBlock(Model):
    def __init__(self, model: Optional[str] = "bernoulli",
        visible_shape: Optional[Tuple[int, int]] = (28, 28),
        in_channels: Optional[int] = 1,
        out_channels: Optional[int] = 1,
        steps: Optional[Tuple[int, ...]] = (1,),
        learning_rate: Optional[Tuple[float, ...]] = (0.1,),
        momentum: Optional[Tuple[float, ...]] = (0.0,),
        decay: Optional[Tuple[float, ...]] = (0.0,),
        sample: Optional[int] = 0,
        input_layers = [],  #TODO - what is type here?
        skip_connection: Optional[int] = -1,
        use_gpu: Optional[bool] = False):


        logger.info("Overriding class: Model -> DBNUnetBlock.")

        super(DBNUnetBlock, self).__init__(use_gpu=use_gpu)

        # Shape of visible units
        self.visible_shape = visible_shape
        self.current_layer_samples = None
        self.mse = []
        self.input_layers = input_layers
        self.skip_connection = skip_connection

        # Shape of filters
        self.filter_shape = (3,3)

        # Number of filters
        self.n_filters = out_channels
 
        # Number of channels
        self.n_channels = in_channels

        # Number of layers
        self.n_layers = 10

        # Number of steps Gibbs' sampling steps
        self.steps = steps

        # Learning rate
        self.lr = learning_rate

        # Momentum parameter
        self.momentum = momentum

        # Weight decay
        self.decay = decay
 
        conv = None
        convT = None
        #TODO fix 
        #if model == "bernoulli":
        #    conv = ConvRBM
        #    convT = ConvTransposeRBM
        #elif model == "gaussian":
        conv = GaussianConvRBM
        convT = GaussianConvTransposeRBM

        self.output_shapes = []
        # 0 = Conv only, 1 = Downsample, 2 = Upsample
        self.sample = sample
        self.pool1 = None
        self.up = None
        conv_ind = 0
        v_shape = visible_shape
        if self.sample == 1:
            #[(I - F) / S] + 1 x D
            v = np.ceil(np.add(np.divide(np.subtract(self.visible_shape,2),2),1)).astype(np.int16)
            self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2
            self.output_shapes.append(v)
            v_shape = v
        elif self.sample == 2:
            conv_ind = 1
            self.up = convT(visible_shape,
                (2,2), self.n_channels // 2, self.n_channels // 2, 2, 0, self.steps[0], self.lr[0],
                self.momentum[0], self.decay[0], use_gpu)
            v = np.array(self.up.hidden_shape, dtype = np.int16)
            self.output_shapes.append(v)
            v_shape = v


        print("CHECKING OUT UP", self.up, self.pool1)
        self.conv1_1 = conv(v_shape,
            (3,3), self.n_filters, self.n_channels, 1, 1, self.steps[conv_ind], self.lr[conv_ind],
            self.momentum[conv_ind], self.decay[conv_ind], False, None, use_gpu)

        v = np.array(self.conv1_1.hidden_shape, dtype = np.int16)
        self.output_shapes.append(v)
        self.conv1_2 = conv(v,
            (3,3), self.n_filters, self.n_filters, 1, 1, self.steps[conv_ind+1], self.lr[conv_ind+1],
            self.momentum[conv_ind+1], self.decay[conv_ind+1], False, None, use_gpu)
        v = np.array(self.conv1_2.hidden_shape, dtype = np.int16)
        self.output_shapes.append(v)

        self._models = torch.nn.ModuleList()
        if self.sample == 1:
            self._models.extend([self.pool1, self.conv1_1, self.conv1_2])
        elif self.sample == 2:
            self._models.extend([self.up, self.conv1_1, self.conv1_2])
        else:
            self._models.extend([self.conv1_1, self.conv1_2])
        self._initialize_weights()


    @property
    def models(self) -> torch.nn.ModuleList:
        """List of models (RBMs)."""

        return self._models

    @models.setter
    def models(self, models: torch.nn.ModuleList) -> None:
        self._models = models

    @property
    def visible_shape(self) -> Tuple[int, int]:
        """Shape of visible units."""

        return self._visible_shape

    @visible_shape.setter
    def visible_shape(self, visible_shape: Tuple[int, int]) -> None:
        self._visible_shape = visible_shape

    @property
    def filter_shape(self) -> Tuple[Tuple[int, int], ...]:
        """Shape of filters."""
    
        return self._filter_shape

    @filter_shape.setter
    def filter_shape(self, filter_shape: Tuple[Tuple[int, int], ...]) -> None:
        self._filter_shape = filter_shape

    @property
    def n_filters(self) -> Tuple[int, ...]:
        """Number of filters."""
    
        return self._n_filters

    @n_filters.setter
    def n_filters(self, n_filters: Tuple[int, ...]) -> None:
        self._n_filters = n_filters

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
        #if len(steps) != self.n_layers:
        #    raise e.SizeError(f"`steps` should have size equal as {self.n_layers}")

        self._steps = steps

    @property
    def lr(self) -> Tuple[float, ...]:
        """Learning rate per layer."""

        return self._lr

    @lr.setter
    def lr(self, lr: Tuple[float, ...]) -> None:
        #if len(lr) != self.n_layers:
        #    raise e.SizeError(f"`lr` should have size equal as {self.n_layers}")

        self._lr = lr


    @property
    def momentum(self) -> Tuple[float, ...]:
        """Momentum parameter per layer."""

        return self._momentum

    @momentum.setter
    def momentum(self, momentum: Tuple[float, ...]) -> None:
        #if len(momentum) != self.n_layers:
        #    raise e.SizeError(f"`momentum` should have size equal as {self.n_layers}")

        self._momentum = momentum

    @property
    def decay(self) -> Tuple[float, ...]:
        """Weight decay per layer."""

        return self._decay

    @decay.setter
    def decay(self, decay: Tuple[float, ...]) -> None:
        #if len(decay) != self.n_layers:
        #    raise e.SizeError(f"`decay` should have size equal as {self.n_layers}")

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
        """Fits a new DBNUnetBlock model.

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

        print("PRE TRAIN", str(self.sample))
        gpu_usage()
        dt = torch.float16
        if self.device == "cpu":
            dt = torch.bfloat16
        print(epochs)

        conv_ind = 0
        print("INPUT_LAYERS1", len(self.input_layers))
        if self.sample == 1:
            #if len(self.input_layers) == 0:
            #    self.input_layers.append(self.pool1)
            #else:
            #    self.input_layers[0].append(self.pool1)
            self.input_layers.append(self.pool1)
            print("INPUT_LAYERS2", len(self.input_layers))
        next_inp = []
        if self.sample == 2:
            conv_ind = 1
            logger.info("Fitting layer up")
            self.fit_layer(self.up, dataset, None, epochs[0], batch_size, 0, 
                self.input_layers.copy(), 
                mse, num_loader_workers, pin_memory, is_distributed) 
            print("INPUT_LAYERS3", len(self.input_layers))
            #if len(self.input_layers) == 0:
            #    self.input_layers.append([self.up])
            #else:
            #    self.input_layers[0].append(self.up)
            self.input_layers.append(self.up)
            print("INPUT_LAYERS4", len(self.input_layers))
            next_inp = [[self.input_layers[:self.skip_connection+1].copy(), self.input_layers.copy()]]
        else:
            next_inp = self.input_layers.copy()
            print("INPUT_LAYERS4.5", len(self.input_layers))
        logger.info("Fitting layer conv1_1")
        self.fit_layer(self.conv1_1, dataset, None, epochs[conv_ind], batch_size, 
            conv_ind, next_inp, mse, num_loader_workers, pin_memory, is_distributed)
        print("INPUT_LAYERS5", len(self.input_layers))
        #if len(self.input_layers) == 0:
        #    self.input_layers.append([self.conv1_1])
        #else:
        #    self.input_layers[0].append(self.conv1_1)
        self.input_layers = next_inp
        self.input_layers.append(self.conv1_1)
        print("INPUT_LAYERS6", len(self.input_layers), self.input_layers, "\n\n\n")
        #Data is on GPU, cant use multiple workers
        num_loader_workers = 0
        pin_memory = False

        dataset.data = dataset.data.cpu()
        dataset.targets = dataset.targets.cpu()
        dataset.data = dataset.data.detach()
        dataset.targets = dataset.targets.detach()
        torch.cuda.empty_cache()
        print("LAYER 1")
        logger.info("Fitting layer conv1_2")
        gpu_usage()
        self.fit_layer(self.conv1_2, dataset, None, epochs[conv_ind+1], batch_size, 
            conv_ind + 1, self.input_layers.copy(), mse, num_loader_workers, pin_memory, is_distributed)
        #self.input_layers[0].append(self.conv1_2)
        self.input_layers.append(self.conv1_2)
        print("INPUT_LAYERS_LAST",self.input_layers)
        #self.current_layer_samples = Tensor

        return self.mse


    def fit_layer(
        self,
        model,
        dataset,
        transform,
        epochs,
        batch_size,
        i,
        input_layer,
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
        mods = input_layer
        if len(input_layer) == 0:
            mods = None
        model_mse = model.fit(dataset, batch_size, epochs, loader, sampler, mods).item()

        print("0 USAGE FIT_LAYER")
        gpu_usage()
        #Synchronizes all processes
        dist.barrier()
        # Appending the metrics
        self.mse.append(model_mse)

        model.W.detach()
        model.a.detach()
        model.b.detach()
        torch.cuda.empty_cache()
        dist.barrier()


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

            visible_probs = hidden_probs


            visible_probs, visible_states = self.conv1_2.visible_sampling(visible_probs)
            visible_probs, visible_states = self.conv1_1.visible_sampling(visible_probs)

            if self.sample == 2:
                visible_probs, visible_states = self.up.visible_sampling(visible_probs)
            if self.sample == 1:
                hidden_temp, indices = torch.nn.functional.max_pool2d(samples, 2, 2, ceil_mode=True, return_indices=True)           
                visible_probs = torch.nn.functional.max_unpool2d(visible_probs, indices, 2, stride=2) 
                visible_states = torch.nn.functional.max_unpool2d(visible_states, indices, 2, stride=2)

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



    def forward(self, x: torch.Tensor, x2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs a forward pass over the data.

        Args:
            x: An input tensor for computing the forward pass.

        Returns:
            (torch.Tensor): A tensor containing the Model's outputs.

        """
        h = x
        if self.sample == 1:
            h = self.pool1(h) 
        elif self.sample == 2:
            h = self.up(h)
            # input is CHW
            diffY = x2.size()[2] - h.size()[2]
            diffX = x2.size()[3] - h.size()[3]
            h = torch.nn.functional.pad(h, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
            h = torch.cat([x2, h], dim=1)

        h, _ = self.conv1_1.hidden_sampling(h)
        h, _ = self.conv1_2.hidden_sampling(h)

        return h
 

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ConvRBM):
                m.W.data.zero_()
                torch.nn.init.kaiming_normal_(m.W, mode='fan_out', nonlinearity='relu')
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



class DBNUnet(Model):
    def __init__(self, model: Optional[str] = "bernoulli",
        visible_shape: Optional[Tuple[int, int]] = (28, 28),
        in_channels: Optional[int] = 1,
        out_channels: Optional[int] = 1,
        steps: Optional[Tuple[int, ...]] = (1,),
        learning_rate: Optional[Tuple[float, ...]] = (0.1,),
        momentum: Optional[Tuple[float, ...]] = (0.0,),
        decay: Optional[Tuple[float, ...]] = (0.0,),
        sample: Optional[int] = 0,
        use_gpu: Optional[bool] = False):


        super(DBNUnet, self).__init__(use_gpu=use_gpu)

        logger.info("Overriding class: Model -> DBNUnet.")

        super(DBNUnet, self).__init__(use_gpu=use_gpu)

        # Shape of visible units
        self.visible_shape = visible_shape
        self.current_layer_samples = None
        self.mse = []

        # Shape of filters
        self.filter_shape = (3,3)

        # Number of filters
        self.n_filters = out_channels

        # Number of channels
        self.n_channels = in_channels

        # Number of layers
        self.n_layers = 2

        # Number of steps Gibbs' sampling steps
        self.steps = steps

        # Learning rate
        self.lr = learning_rate

        # Momentum parameter
        self.momentum = momentum

        # Weight decay
        self.decay = decay

        conv = None
        #TODO fix 
        #if model == "bernoulli":
        #    conv = ConvRBM
        #    convT = ConvTransposeRBM
        #elif model == "gaussian":
        conv = GaussianConvRBM

        channel_ratio = (64.0 / 3.0) * self.n_channels
        single = int(math.ceil((channel_ratio * 1) / 2.) * 2)
        #single = 64
        double = single*2 
        trip = double*2
        quad = trip*2
        quint = quad*2

        input_layers = []
        self.inConv = DBNUnetBlock(model, self.visible_shape,
            self.n_channels, single, self.steps[0:2], self.lr[0:2],
                self.momentum[0:2], self.decay[0:2], 0, input_layers, -1, use_gpu)

        self.enc1 = DBNUnetBlock(model, self.inConv.output_shapes[-1],
           single, double, self.steps[2:4], self.lr[2:4],
                self.momentum[2:4], self.decay[2:4], 1, 
                    self.inConv.input_layers, -1, use_gpu)


        self.enc2 = DBNUnetBlock(model, self.inConv.output_shapes[-1],
            double, trip, self.steps[4:6], self.lr[4:6],
                self.momentum[4:6], self.decay[4:6], 1, 
                    self.enc1.input_layers, -1, use_gpu)


        self.enc3 = DBNUnetBlock(model, self.inConv.output_shapes[-1],
            trip, quad, self.steps[6:8], self.lr[6:8],
                self.momentum[6:8], self.decay[6:8], 1, 
                    self.enc2.input_layers, -1, use_gpu)


        self.enc4 = DBNUnetBlock(model, self.inConv.output_shapes[-1],
            quad, quad, self.steps[8:10], self.lr[8:10],
                self.momentum[8:10], self.decay[8:10], 1, 
                    self.enc3.input_layers, -1, use_gpu) 


        self.dec4 = DBNUnetBlock(model, self.inConv.output_shapes[-1],
            quint, trip, self.steps[10:13], self.lr[10:13],
                self.momentum[10:13], self.decay[10:13], 2, 
                    self.enc4.input_layers, 10, use_gpu)



        self.dec3 = DBNUnetBlock(model, self.inConv.output_shapes[-1],
            quad, double, self.steps[13:16], self.lr[13:16],
                self.momentum[13:16], self.decay[13:16], 2, 
                    self.dec4.input_layers, 7, use_gpu)

        self.dec2 = DBNUnetBlock(model, self.inConv.output_shapes[-1],
            trip, single, self.steps[16:19], self.lr[16:19],
                self.momentum[16:19], self.decay[16:19], 2,
                self.dec3.input_layers, 4, use_gpu)

        self.dec1 = DBNUnetBlock(model, self.inConv.output_shapes[-1],
            double, single, self.steps[19:22], self.lr[19:22],
                self.momentum[19:22], self.decay[19:22], 2, 
                self.dec2.input_layers, 1, use_gpu)

        self.outConv = conv(self.dec1.output_shapes[-1],
            (1,1), self.n_filters, single, 1, 0, self.steps[22], self.lr[22],
            self.momentum[22], self.decay[22], False, None, use_gpu)

        self._models = torch.nn.ModuleList()
        self._models.extend([self.inConv, self.enc1, self.enc2, self.enc3, self.enc4,
            self.dec4, self.dec3, self.dec2, self.dec1, self.outConv])


    def fit(
        self,
        dataset: Union[torch.utils.data.Dataset, Dataset],
        batch_size: Optional[int] = 128,
        epochs: Optional[Tuple[int, ...]] = (10, 10),
        is_distributed: Optional[bool] = False,
        num_loader_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = False
    ) -> float:
        """Fits a new DBNUnet model.

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

        conv_ind = 0
        internal_ind = 0
        for i in range(len(self.models)):
            if i < len(self.models) - 1:
                logger.info("Fitting Unet Block " + str(i))
                increase = 3
                if i <= 4:
                    increase = 2    
                err = self.models[i].module.fit(dataset, batch_size, epochs[internal_ind: internal_ind + increase],
                    is_distributed, num_loader_workers, pin_memory)
                if i < len(self.models)-1:
                    if hasattr(self.models[i+1], "module"):    
                        self.models[i+1].module.input_layers = self.models[i].module.input_layers.copy()
                    else:
                        self.models[i+1].input_layers = self.models[i].input_layers.copy()
                mse.extend(err)
                internal_ind = internal_ind + increase
            else:  
                # For every possible model (ConvRBM)
                logger.info("Fitting OutConv ...")
 
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
                model_mse = self.models[i].module.fit(dataset, batch_size, epochs[-1], loader, sampler, self.dec1.input_layers).item()

                print("0 USAGE FIT_LAYER")
                gpu_usage()
                #Synchronizes all processes
                dist.barrier()
                # Appending the metrics
                self.mse.append(model_mse)
         
                if hasattr(self.models[i+1], "module"): 
                    self.models[i].module.W.detach()
                    self.models[i].module.a.detach()
                    self.models[i].module.b.detach()
                else:
                    self.models[i].W.detach()
                    self.models[i].a.detach()
                    self.models[i].b.detach()
                torch.cuda.empty_cache()
                dist.barrier()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass over the data.

        Args:
            x: An input tensor for computing the forward pass.

        Returns:
            (torch.Tensor): A tensor containing the Model's outputs.

        """
        h = x
        x1 = self.inConv(h)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        h = self.dec1.forward(x5,x4)
        del x5, x4
        h = self.dec2.forward(h,x3)
        del x3
        h = self.dec3.forward(h,x2)
        del x2
        h = self.dec4.forward(h,x1)
        del x1
        h = self.outConv.forward(h)        
        return h

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
   

            for i in range(len(self.models)):
                 if i < len(self.models) - 1:
                     # Performing a hidden layer sampling
                     for j in range(len(self.models[i].models)):
                         hidden_probs, _ = self.models[i].models[j].hidden_sampling(hidden_probs)
                 else:
                     hidden_probs, _ = self.models[i].hidden_sampling(hidden_probs)

            visible_probs = hidden_probs


            for i in range(len(self.models)):
                 if i < len(self.models) - 1:
                     # Performing a hidden layer sampling
                     for j in range(len(self.models[i].models)):
                         visible_probs, visible_states = self.models[i].models[j].visible_sampling(visible_probs)
                         
                 else:
                     visible_probs, visible_states = self.models[i].visible_sampling(visible_probs)
                    

    @property
    def visible_shape(self) -> Tuple[int, int]:
        """Shape of visible units."""

        return self._visible_shape

    @visible_shape.setter
    def visible_shape(self, visible_shape: Tuple[int, int]) -> None:
        self._visible_shape = visible_shape


    @property
    def models(self) -> torch.nn.ModuleList:
        """List of models (RBMs)."""

        return self._models

    @models.setter
    def models(self, models: torch.nn.ModuleList) -> None:
        self._models = models


