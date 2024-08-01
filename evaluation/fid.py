# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Generic, Iterable, List, Optional, TypeVar, Union

import torch

# pyre-fixme[24]: Generic type `Metric` expects 1 type parameter.
TSelf = TypeVar("TSelf", bound="Metric")
TComputeReturn = TypeVar("TComputeReturn")
# pyre-ignore[33]: Flexible key data type for dictionary
TState = Union[torch.Tensor, List[torch.Tensor], Dict[Any, torch.Tensor], int, float]
# pyre-strict
import csv
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse
import os

from pytorch_fid.inception import InceptionV3 as FIDInceptionV3
import warnings
from importlib.util import find_spec
from typing import Any, Iterable, Optional, TypeVar, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
#from torcheval.metrics.metric import Metric

if find_spec("torchvision") is not None:
    from torchvision import models

    _TORCHVISION_AVAILABLE = True
else:
    _TORCHVISION_AVAILABLE = False

TFrechetInceptionDistance = TypeVar("TFrechetInceptionDistance")

# pyre-ignore[2]: Type checking for ``value`` which could be any type.
def _check_state_variable_type(name: str, value: Any) -> None:
    """
    Check the type of a state variable value.
    It should be a type of TState.
    """
    if (
        not isinstance(value, torch.Tensor)
        and not (
            isinstance(value, list) and all(isinstance(x, torch.Tensor) for x in value)
        )
        and not (
            isinstance(value, dict)
            and all(isinstance(x, torch.Tensor) for x in value.values())
        )
        and not isinstance(value, int)
        and not isinstance(value, float)
    ):
        raise TypeError(
            "The value of state variable must be a ``torch.Tensor``, a list of ``torch.Tensor``, "
            f"a dictionary with ``torch.Tensor``, int, or float as values."
            f"Got {name}={value} instead."
        )
    
# pyre-ignore-all-errors[16]: Undefined attribute of metric states.
class Metric(Generic[TComputeReturn], ABC):
    """
    Base class for all metrics present in the Metrics API.

    Implement __init__(), update(), compute(), merge_state() functions
    to implement your own metric.
    """

    def __init__(
        self: TSelf,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize a metric object and its internal states.

        Use ``self._add_state()`` to initialize state variables of your metric class.
        The state variables should be either ``torch.Tensor``, a list of
        ``torch.Tensor``, or a dictionary with ``torch.Tensor`` as values
        """
        torch._C._log_api_usage_once(f"torcheval.metrics.{self.__class__.__name__}")

        # limit state variable type to tensor/[tensor] to avoid working with nested
        # data structures when move/detach/clone tensors. Can open more types up
        # upon user requests in the future.
        self._state_name_to_default: Dict[str, TState] = {}
        self._device: torch.device = torch.device("cpu") if device is None else device

    def _add_state(self: TSelf, name: str, default: TState) -> None:
        """
        Used in subclass ``__init__()`` to add a metric state variable.

        Args:
            name: The name of the state variable. The variable can be accessed
                with ``self.name``.
            default: Default value of the state. It should be a type of TState.
                The state will be reset to this value when ``self.reset()`` is called.
        Raises:
            TypeError: If ``default`` is not a type of TState.
        """
        _check_state_variable_type(name, default)
        # deepcopy makes sure the input/initial value/default value of the state
        # variable are independent.
        setattr(self, name, deepcopy(default))
        self._state_name_to_default[name] = deepcopy(default)

    @abstractmethod
    @torch.inference_mode()
    def update(self: TSelf, *_: Any, **__: Any) -> TSelf:
        """
        Implement this method to update the state variables of your metric class.

        Decorate update() with @torch.inference_mode() which gives better
        performance by disabling view tracking.
        """

    @abstractmethod
    @torch.inference_mode()
    def compute(self: TSelf) -> TComputeReturn:
        """
        Implement this method to compute and return the final metric value
        from state variables.

        Decorate compute() with @torch.inference_mode() which gives better
        performance by disabling view tracking.
        """

    @abstractmethod
    @torch.inference_mode()
    def merge_state(self: TSelf, metrics: Iterable[TSelf]) -> TSelf:
        """
        Implement this method to update the current metric's state variables
        to be the merged states of the current metric and input metrics. The state
        variables of input metrics should stay unchanged.

        Decorate merge_state() with @torch.inference_mode() which gives better
        performance by disabling view tracking.

        ``self.merge_state`` might change the size/shape of state variables.
        Make sure ``self.update`` and ``self.compute`` can still be called
        without exceptions when state variables are merged.

        This method can be used as a building block for syncing metric states
        in distributed training. For example, ``sync_and_compute`` in the metric
        toolkit will use this method to merge metric objects gathered from the
        process group.
        """

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TSelf) -> None:
        """
        Called before syncing metrics in ``toolkit._sync_metric_object()``.

        It can be utilized to adjust metric states to accelerate syncing.
        For example, concatenated metric state from a list of tensors to
        one tensor. See ``torcheval.metrics.BinaryAUROC`` as an example.
        """
        pass

    def reset(self: TSelf) -> TSelf:
        """
        Reset the metric state variables to their default value.
        The tensors in the default values are also moved to the device of
        the last ``self.to(device)`` call.
        """
        for state_name, default in self._state_name_to_default.items():
            if isinstance(default, torch.Tensor):
                setattr(self, state_name, default.clone().to(self.device))
            elif isinstance(default, list):
                setattr(
                    self,
                    state_name,
                    [tensor.clone().to(self.device) for tensor in default],
                )
            elif isinstance(default, dict):
                setattr(
                    self,
                    state_name,
                    defaultdict(
                        lambda: torch.tensor(0.0, device=self.device),
                        {
                            key: tensor.clone().to(self.device)
                            for key, tensor in default.items()
                        },
                    ),
                )
            elif isinstance(default, (int, float)):
                setattr(self, state_name, default)
            else:
                raise TypeError(
                    f"Invalid type for default value for {state_name}. Received {type(default)}, but expected ``torch.Tensor``, a list of ``torch.Tensor``,"
                    f"a dictionary with ``torch.Tensor``, int, or float."
                )
        return self

    def state_dict(self: TSelf) -> Dict[str, TState]:
        """
        Save metric state variables in state_dict.

        Raises:
            TypeError: If ``default`` is not a type of TState.
        """
        state_dict = {}
        for state_name in self._state_name_to_default:
            value = getattr(self, state_name)
            _check_state_variable_type(state_name, value)

            if isinstance(value, torch.Tensor):
                state_dict[state_name] = value.detach().clone()
            elif isinstance(value, list):
                state_dict[state_name] = [tensor.detach().clone() for tensor in value]
            elif isinstance(value, dict):
                state_dict[state_name] = {
                    key: tensor.detach().clone() for key, tensor in value.items()
                }
            elif isinstance(value, int):
                state_dict[state_name] = value
            elif isinstance(value, float):
                state_dict[state_name] = value
        return state_dict

    def load_state_dict(
        self: TSelf,
        state_dict: Dict[str, Any],
        strict: bool = True,
    ) -> None:
        """
        Loads metric state variables from state_dict.

        Args:
            state_dict (Dict[str, Any]): A dict containing metric state variables.
            strict (bool, Optional): Whether to strictly enforce that the keys in ``state_dict`` matches
                all names of the metric states.

        Raises:
            RuntimeError: If ``strict`` is ``True`` and keys in state_dict does not match
                all names of the metric states.
            TypeError: If ``default`` is not a type of TState.
        """
        state_dict = deepcopy(state_dict)
        metric_state_names = set(self._state_name_to_default.keys())
        for state_name in metric_state_names:
            if state_name in state_dict:
                value = state_dict[state_name]
                _check_state_variable_type(state_name, value)
                setattr(self, state_name, value)

        if strict:
            state_dict_keys = set(state_dict.keys())
            unexpected_keys = state_dict_keys.difference(metric_state_names)
            missing_keys = metric_state_names.difference(state_dict_keys)
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {self.__class__.__name__}. "
                    f"Encountered missing keys: {missing_keys} and unexpected "
                    f"keys: {unexpected_keys}."
                )

    def to(
        self: TSelf, device: Union[str, torch.device], *args: Any, **kwargs: Any
    ) -> TSelf:
        """
        Move tensors in metric state variables to device.

        Args:
            device: The destination device.
        Raises:
            TypeError: If ``default`` is not a type of TState.
        """
        device = torch.device(device) if isinstance(device, str) else device
        for state_name in self._state_name_to_default:
            value = getattr(self, state_name)
            _check_state_variable_type(state_name, value)
            if isinstance(value, torch.Tensor):
                setattr(self, state_name, value.to(device))
            elif isinstance(value, list):
                setattr(
                    self,
                    state_name,
                    [tensor.to(device, *args, **kwargs) for tensor in value],
                )
            elif isinstance(value, dict):
                setattr(
                    self,
                    state_name,
                    defaultdict(
                        lambda: torch.tensor(0.0, device=device),
                        {
                            key: tensor.to(device, *args, **kwargs)
                            for key, tensor in value.items()
                        },
                    ),
                )
        self._device = device
        return self

    @property
    def device(self: TSelf) -> torch.device:
        """
        The last input device of ``Metric.to()``.
        Default to ``torch.device("cpu")`` if ``Metric.to()`` is not called.
        """
        return self._device

def gaussian_frechet_distance(
    mu_x: torch.Tensor, cov_x: torch.Tensor, mu_y: torch.Tensor, cov_y: torch.Tensor
) -> torch.Tensor:
    r"""Computes the Fréchet distance between two multivariate normal distributions :cite:`dowson1982frechet`.

    The Fréchet distance is also known as the Wasserstein-2 distance.

    Concretely, for multivariate Gaussians :math:`X(\mu_X, \cov_X)`
    and :math:`Y(\mu_Y, \cov_Y)`, the function computes and returns :math:`F` as

    .. math::
        F(X, Y) = || \mu_X - \mu_Y ||_2^2
        + \text{Tr}\left( \cov_X + \cov_Y - 2 \sqrt{\cov_X \cov_Y} \right)

    Args:
        mu_x (torch.Tensor): mean :math:`\mu_X` of multivariate Gaussian :math:`X`, with shape `(N,)`.
        cov_x (torch.Tensor): covariance matrix :math:`\cov_X` of :math:`X`, with shape `(N, N)`.
        mu_y (torch.Tensor): mean :math:`\mu_Y` of multivariate Gaussian :math:`Y`, with shape `(N,)`.
        cov_y (torch.Tensor): covariance matrix :math:`\cov_Y` of :math:`Y`, with shape `(N, N)`.

    Returns:
        torch.Tensor: the Fréchet distance between :math:`X` and :math:`Y`.
    """
    if mu_x.ndim != 1:
        msg = f"Input mu_x must be one-dimensional; got dimension {mu_x.ndim}."
        raise ValueError(msg)
    if mu_y.ndim != 1:
        msg = f"Input mu_y must be one-dimensional; got dimension {mu_y.ndim}."
        raise ValueError(msg)
    if cov_x.ndim != 2:
        msg = f"Input cov_x must be two-dimensional; got dimension {cov_x.ndim}."
        raise ValueError(msg)
    if cov_y.ndim != 2:
        msg = f"Input cov_x must be two-dimensional; got dimension {cov_y.ndim}."
        raise ValueError(msg)
    if mu_x.shape != mu_y.shape:
        msg = f"Inputs mu_x and mu_y must have the same shape; got {mu_x.shape} and {mu_y.shape}."
        raise ValueError(msg)
    if cov_x.shape != cov_y.shape:
        msg = f"Inputs cov_x and cov_y must have the same shape; got {cov_x.shape} and {cov_y.shape}."
        raise ValueError(msg)

    a = (mu_x - mu_y).square().sum()
    b = cov_x.trace() + cov_y.trace()
    c = torch.linalg.eigvals(cov_x @ cov_y).sqrt().real.sum()
    return a + b - 2 * c



def _validate_torchvision_available() -> None:
    if not _TORCHVISION_AVAILABLE:
        raise RuntimeError(
            "You must have torchvision installed to use FID, please install torcheval[image]"
        )


class FIDInceptionV3_fb(nn.Module):
    def __init__(
        self,
        weights: Optional[str] = "DEFAULT",
    ) -> None:
        """
        This class wraps the InceptionV3 model to compute FID.

        Args:
            weights Optional[str]: Defines the pre-trained weights to use.
        """
        super().__init__()
        # pyre-ignore
        self.model = models.inception_v3(weights=weights)
        # Do not want fc layer
        self.model.fc = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # Interpolating the input image tensors to be of size 299 x 299
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = self.model(x)

        return x
    


class FrechetInceptionDistance(Metric[torch.Tensor]):
    def __init__(
        self: TFrechetInceptionDistance,
        model: Optional[nn.Module] = None,
        feature_dim: int = 2048,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Computes the Frechet Inception Distance (FID) between two distributions of images (real and generated).

        The original paper: https://arxiv.org/pdf/1706.08500.pdf

        Args:
            model (nn.Module): Module used to compute feature activations.
                If None, a default InceptionV3 model will be used.
            feature_dim (int): The number of features in the model's output,
                the default number is 2048 for default InceptionV3.
            device (torch.device): The device where the computations will be performed.
                If None, the default device will be used.
        """
        _validate_torchvision_available()

        super().__init__(device=device)

        self._FID_parameter_check(model=model, feature_dim=feature_dim)

        if model is None:
            model = FIDInceptionV3()

        # Set the model and put it in evaluation mode
        self.model = model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        # Initialize state variables used to compute FID
        self._add_state("real_sum", torch.zeros(feature_dim, device=device))
        self._add_state(
            "real_cov_sum", torch.zeros((feature_dim, feature_dim), device=device)
        )
        self._add_state("fake_sum", torch.zeros(feature_dim, device=device))
        self._add_state(
            "fake_cov_sum", torch.zeros((feature_dim, feature_dim), device=device)
        )
        self._add_state("num_real_images", torch.tensor(0, device=device).int())
        self._add_state("num_fake_images", torch.tensor(0, device=device).int())

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TFrechetInceptionDistance, images: Tensor, is_real: bool
    ) -> TFrechetInceptionDistance:
        """
        Update the states with a batch of real and fake images.

        Args:
            images (Tensor): A batch of images.
            is_real (Boolean): Denotes if images are real or not.
        """

        self._FID_update_input_check(images=images, is_real=is_real)

        images = images.to(self.device)

        # Compute activations for images using the given model
        activations = self.model(images)[-1]

        batch_size = images.shape[0]

        # Update the state variables used to compute FID
        if is_real:
            self.num_real_images += batch_size
            self.real_sum += torch.sum(activations, dim=0)
            self.real_cov_sum += torch.matmul(activations.T, activations)
        else:
            self.num_fake_images += batch_size
            self.fake_sum += torch.sum(activations, dim=0)
            self.fake_cov_sum += torch.matmul(activations.T, activations)

        return self

    @torch.inference_mode()
    def merge_state(
        self: TFrechetInceptionDistance, metrics: Iterable[TFrechetInceptionDistance]
    ) -> TFrechetInceptionDistance:
        """
        Merge the state of another FID instance into this instance.

        Args:
            metrics (Iterable[FID]): The other FID instance(s) whose state will be merged into this instance.
        """
        for metric in metrics:
            self.real_sum += metric.real_sum.to(self.device)
            self.real_cov_sum += metric.real_cov_sum.to(self.device)
            self.fake_sum += metric.fake_sum.to(self.device)
            self.fake_cov_sum += metric.fake_cov_sum.to(self.device)
            self.num_real_images += metric.num_real_images.to(self.device)
            self.num_fake_images += metric.num_fake_images.to(self.device)

        return self

    @torch.inference_mode()
    def compute(self: TFrechetInceptionDistance) -> Tensor:
        """
        Compute the FID.

        Returns:
            tensor: The FID.
        """

        # If the user has not already updated with at lease one
        # image from each distribution, then we raise an Error.
        if (self.num_real_images < 2) or (self.num_fake_images < 2):
            warnings.warn(
                "Computing FID requires at least 2 real images and 2 fake images,"
                f"but currently running with {self.num_real_images} real images and {self.num_fake_images} fake images."
                "Returning 0.0",
                RuntimeWarning,
                stacklevel=2,
            )

            return torch.tensor(0.0)

        # Compute the mean activations for each distribution
        real_mean = (self.real_sum / self.num_real_images).unsqueeze(0)
        fake_mean = (self.fake_sum / self.num_fake_images).unsqueeze(0)

        # Compute the covariance matrices for each distribution
        real_cov_num = self.real_cov_sum - self.num_real_images * torch.matmul(
            real_mean.T, real_mean
        )
        real_cov = real_cov_num / (self.num_real_images - 1)
        fake_cov_num = self.fake_cov_sum - self.num_fake_images * torch.matmul(
            fake_mean.T, fake_mean
        )
        fake_cov = fake_cov_num / (self.num_fake_images - 1)

        # Compute the Frechet Distance between the distributions
        fid = gaussian_frechet_distance(
            real_mean.squeeze(), real_cov, fake_mean.squeeze(), fake_cov
        )
        return fid

    def _FID_parameter_check(
        self: TFrechetInceptionDistance,
        model: Optional[nn.Module],
        feature_dim: int,
    ) -> None:
        # Whatever the model, the feature_dim needs to be set
        if feature_dim is None or feature_dim <= 0:
            raise RuntimeError("feature_dim has to be a positive integer")

        if model is None and feature_dim != 2048:
            raise RuntimeError(
                "When the default Inception v3 model is used, feature_dim needs to be set to 2048"
            )

    def _FID_update_input_check(
        self: TFrechetInceptionDistance, images: torch.Tensor, is_real: bool
    ) -> None:
        if not torch.is_tensor(images):
            raise ValueError(f"Expected tensor as input, but got {type(images)}.")

        if images.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor as input. But input has {images.dim()} dimenstions."
            )

        if images.size()[1] != 3:
            raise ValueError(f"Expected 3 channels as input. Got {images.size()[1]}.")

        if type(is_real) != bool:
            raise ValueError(
                f"Expected 'real' to be of type bool but got {type(is_real)}.",
            )

        if isinstance(self.model, FIDInceptionV3):
            if images.dtype != torch.float32:
                raise ValueError(
                    f"When default inception-v3 model is used, images expected to be `torch.float32`, but got {images.dtype}."
                )

            if images.min() < 0 or images.max() > 1:
                raise ValueError(
                    "When default inception-v3 model is used, images are expected to be in the [0, 1] interval"
                )

    def to(
        self: TFrechetInceptionDistance,
        device: Union[str, torch.device],
        *args: Any,
        **kwargs: Any,
    ) -> TFrechetInceptionDistance:
        super().to(device=device)
        self.model.to(self.device)
        return self

class InferenceDataset(Dataset):
    def __init__(self, datadir, inference_dir, eval_resolution=299, img_suffix='.jpg', inpainted_suffix='_removed.png'):
        self.inference_dir = inference_dir
        self.datadir = datadir
        if not datadir.endswith('/'):
            datadir += '/'
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask*.png'), recursive=True)))
        self.img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]
        self.file_names = [os.path.join(inference_dir, os.path.splitext(fname[len(datadir):])[0] + inpainted_suffix)
                               for fname in self.img_filenames]
        self.eval_resolution = eval_resolution
        self.ids = [file_name.rsplit('/', 1)[1].rsplit('_mask.png', 1)[0] for file_name in self.mask_filenames]
        #self.test_filenames = [os.path.join("/hy-tmp/DATA/test_sampled/", id + img_suffix) for id in self.ids]

    def __len__(self):
        return len(self.ids)
    
    def read_image(self, path):
        img = Image.open(path)
        img = img.convert('RGB')
        img = img.resize((self.eval_resolution,self.eval_resolution), Image.Resampling.BICUBIC)
        img = np.array(img, dtype=float) / 255
        img = np.moveaxis(img, [0,1,2], [1,2,0])
        img = torch.from_numpy(img).float()
        return img

    def __getitem__(self, idx):
        #scene_id = self.ids[idx]
        
        target_image = self.read_image(self.img_filenames[idx])
        #target_image = self.read_image(self.test_filenames[idx])
        inpainted_image = self.read_image(self.file_names[idx])
        return target_image, inpainted_image

class Inferencedataset_local(InferenceDataset):
    def __init__(self, datadir, inference_dir, test_scene, eval_resolution=512, img_suffix='.jpg', inpainted_suffix='_removed.png'):
        super().__init__(datadir, inference_dir, eval_resolution, img_suffix, inpainted_suffix)
        self.test_scene = self.read_csv_to_dict(test_scene)

    def read_csv_to_dict(self,file_path):
        data_dict = {}
        
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            header = next(reader)  # Skip header if there is one
            for row in reader:
                id = row[0].rsplit('.', 1)[0]
                LabelName = row[1]
                BoxXMin = float(row[2])
                BoxXMax = float(row[3])
                BoxYMin = float(row[4])
                BoxYMax = float(row[5])
                
                data_dict[id] = {
                    'LabelName': LabelName,
                    'BoxXMin': BoxXMin,
                    'BoxXMax': BoxXMax,
                    'BoxYMin': BoxYMin,
                    'BoxYMax': BoxYMax
                }
        return data_dict
    
    def read_image(self, path, object_bbox):
        img = Image.open(path).crop(object_bbox)
        img = img.convert('RGB')
        img = img.resize((self.eval_resolution,self.eval_resolution), Image.Resampling.BICUBIC)
        img = np.array(img, dtype=float) / 255
        img = np.moveaxis(img, [0,1,2], [1,2,0])
        img = torch.from_numpy(img).float()
        return img
    
    def __getitem__(self, idx):
        scene_id = self.ids[idx]
        object_bbox = (int(self.test_scene[scene_id]["BoxXMin"]*self.eval_resolution),
                       int(self.test_scene[scene_id]["BoxYMin"]*self.eval_resolution),
                       int(self.test_scene[scene_id]["BoxXMax"]*self.eval_resolution),
                       int(self.test_scene[scene_id]["BoxYMax"]*self.eval_resolution))
        
        target_image = self.read_image(self.img_filenames[idx], object_bbox)
        #target_image = self.read_image(self.test_filenames[idx], object_bbox)
        inpainted_image = self.read_image(self.file_names[idx], object_bbox)
        return target_image, inpainted_image

def get_frechet_inception_distance(dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fid = FrechetInceptionDistance(device=device)

    for _, inpainted_image in tqdm(dataloader, desc=f'FID - Fake Data Feature Extraction', total=len(dataloader)):
        #image_batch = torch.Tensor(inpainted_image).to(device)
        image_batch = torch.Tensor(inpainted_image)
        fid.update(image_batch, is_real=False)

    for target_image, _ in tqdm(dataloader, desc=f'FID - Real Data Feature Extraction', total=len(dataloader)):
        #image_batch = torch.Tensor(target_image).to(device)
        image_batch = torch.Tensor(target_image)
        fid.update(image_batch, is_real=True)
    result = fid.compute()
    return result.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datadir",
        type=str,
        default="outputs/gqa_inpaint_inference/",
        help="Directory of the original images and masks",
    )
    parser.add_argument(
        "--inference_dir",
        type=str,
        default="outputs/gqa_inpaint_inference/",
        help="Directory of the inference results",
    )
    parser.add_argument(
        "--test_scene",
        type=str,
        default="/hy-tmp/DATA/fetch_output.csv",
        help="path of the test scene",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/gqa_inpaint_eval/",
        help="Directory of evaluation outputs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Bath size of Inception v3 forward pass",
    )
    parser.add_argument(
        "--inpainted_suffix",
        type=str,
        default='_removed.png',
        help="inference_dir's suffix",
    )
    args = parser.parse_args()

    dataset = InferenceDataset(args.datadir, args.inference_dir, eval_resolution=512, img_suffix='.jpg', inpainted_suffix=args.inpainted_suffix)
    dataset_local = Inferencedataset_local(args.datadir, args.inference_dir, args.test_scene, eval_resolution=512, img_suffix='.jpg', inpainted_suffix=args.inpainted_suffix)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    dataloader_local = torch.utils.data.DataLoader(dataset_local, batch_size=args.batch_size, shuffle=False)
    print('start to calculate FID_local')
    fid_local = get_frechet_inception_distance(dataloader_local)
    print(f"FID_local: {fid_local}")
    print('start to calculate FID')
    fid = get_frechet_inception_distance(dataloader)
    print(f"FID: {fid}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    fid_str = f"FID: {fid}"
    fid_local_str = f"FID_local: {fid_local}"
    output_path = os.path.join(output_dir, f"fid_{dataset.eval_resolution}.txt")
    f = open(output_path, "w")
    f.write(fid_str + '\n')
    f.write(fid_local_str)
    f.close()

    print(fid_str)
    print(fid_local_str)