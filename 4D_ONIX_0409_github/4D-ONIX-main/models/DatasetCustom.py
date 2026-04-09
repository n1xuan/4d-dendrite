"""
    This is an example of preparing the dataset for training.
    We show the case of phase+attenuation training, for the case of attenuation only, please remove the phase part.
"""

from random import randint
import numpy as np
import torch
from models.mat_calc import get_world_mat
from models.utils import data_preproc, new_reg
from typing import (
    Iterator,
    Iterable,
    Optional,
    Sequence,
    List,
    TypeVar,
    Generic,
    Sized,
    Union,
)
from itertools import chain


class CustomDataset(torch.utils.data.Dataset):
    """Load data [n_views,H,W]"""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.n_views = opt.n_views
        self.use_time = opt.use_time
        path = opt.load_path
        self.total_views = 2  #  total number of views used in the training. Change it to match the experimental setup

        images_all = np.load(
            path
        )  # For ph+att: [dataset_size, num_projections, 2, H, W], otherwise [dataset_size, num_projections, H, W]
        # att = data_preproc(images)  # Preprocess
        images = images_all[:, :2, ...] # only use the first 2 views when model training

        # att = new_reg(att)
        # images =  new_reg(images)
        # images = np.concatenate((att, att), axis=2)  # copy channel(no need: later use --in_channel 1 when training)
        
        if images.ndim == 4:
            self.images_pool = torch.from_numpy(images)[: images.shape[0], :, None, ...]

        elif images.ndim == 5:
            self.images_pool = torch.from_numpy(images)

        else:
            raise NotImplementedError(
                f"Please check the dataloader for supported dataset shape."
            )
        self.generate_poses()
        if self.use_time:
            self.generate_timestamp()

    def generate_poses(self):
        # ProjAngles = np.linspace(0.0, 23.86 * 8, 9)[:2]  # NB: Modify this part to match the experimental setup
        print("Training Poses: 0.0, 90.0")
        ProjAngles = np.array([0.0, 90.0])
        theta_x = 0
        theta_z = 0
        theta_y = ProjAngles
        world_mat = []
        for i in range(ProjAngles.shape[0]):
            world_mat.append(get_world_mat(theta_x, theta_y[i], theta_z))
        world_matrix = np.asarray(world_mat).astype(float)
        self.tform_cam2world = torch.from_numpy(world_matrix).float()  # [100,4,4]

    def generate_timestamp(self):
        """
        Generate normalized time stamps for the simulation dataset. 
        Normalize the time frames to the range [-1.0, 1.0] to prevent 
        positional encoding aliasing and ensure stable MLP convergence.
        """
        num_frames = self.images_pool.shape[0]
        times = torch.arange(num_frames, dtype=torch.float32)
        
        if num_frames > 1:
            # 线性映射到 [-1.0, 1.0]
            self.images_time = (times / (num_frames - 1)) * 2.0 - 1.0
        else:
            # 如果只有一帧 (比如在某些Test情况下)，直接设为 0.0
            self.images_time = torch.zeros_like(times)

    def __len__(self):
        return self.images_pool.shape[0]

    def __getitem__(self, index):
        self.imgs = self.images_pool[index]
        self.poses = self.tform_cam2world
        if self.use_time:
            self.times = self.images_time[index]
            return self.imgs, self.poses, self.times
        else:
            return self.imgs, self.poses


class MyRandomBatchSampler(torch.utils.data.Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator=None,
        batch_size=2,
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self.batch_size = batch_size

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        for _ in range(self.num_samples // n):
            samples = torch.randperm(
                n - self.batch_size + 1, generator=generator
            ).tolist()
            sample_all = [
                list(range(sample, sample + self.batch_size)) for sample in samples
            ]
            yield from chain.from_iterable(sample_all)
        # yield from sample_all[: self.num_samples % n]

    def __len__(self) -> int:
        return (len(self.data_source) - self.batch_size + 1) * self.batch_size
