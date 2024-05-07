import os
import pathlib
import pickle
from typing import Dict, Optional, Tuple, Union, Sequence
from functools import partial
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch import Tensor
from .data_types import HeterogeneousBatch, HeterogeneousData
from . import fedot_pipeline_features_extractor


class HeteroPipelineAndDatasetFeaturesDataset(Dataset):
    # Load from no_meta_features_and_fedot_pipelines_raw
    # The folder is made with `GAMLET/scripts/generate_dataset_with_hyperparameters_raw.py`.
    def __init__(
        self,
        task_pipe_comb_file: str,
        meta_features_file: str,
        id2pipe: Dict[int, Union[pathlib.PosixPath, str]],
        id2dataset: Dict[int, str],
        encode_type: Optional[Union[str, None]] = "ordinal",
        pipelines_per_step: int = 5,  # Minimum is 2
        normalize: bool = True,
    ):
        self.normalize = normalize

        if pipelines_per_step < 2:
            raise ValueError("`pipelines_per_step`must be >= 2")

        self.pipelines_per_step = pipelines_per_step
        self.task_pipe_comb = pd.read_csv(task_pipe_comb_file)
        self.meta_features = pd.read_csv(meta_features_file, index_col=0)
        self.n_meta_features = self.meta_features.shape[1]

        self.groups = {k: v for k, v in self.task_pipe_comb.groupby("task_id")}
        self.id2pipe = id2pipe
        self.id2dataset = id2dataset
        self.dataset_ids = list(self.groups.keys())
        self.collate_fn = partial(HeteroPipelineAndDatasetFeaturesDataset.collate_fn, n_pipelines=self.pipelines_per_step)
        self.pipeline_extractor = fedot_pipeline_features_extractor.FEDOTPipelineFeaturesExtractor2(encode_type)

    @staticmethod
    def collate_fn(
        batch: Sequence[Tuple[Sequence[HeterogeneousData], Sequence[float], Tensor]],
        n_pipelines: int,
    ) -> Tuple[Sequence[HeterogeneousBatch], Tensor, Tensor]:
        pipes = [HeterogeneousBatch.from_heterogeneous_data_list([b[0][i] for b in batch]) for i in range(n_pipelines)]
        dataset_features = torch.vstack([b[1] for b in batch])
        y = torch.FloatTensor([b[2] for b in batch])
        # Pipes are [N_PIPES, BATCH_SIZE, N_FEATURES]
        return pipes, dataset_features, y

    def __len__(self):
        return len(self.task_pipe_comb)

    def _get_sample(self) -> Tuple[Sequence[HeterogeneousData], Tensor, Sequence[float]]:
        task_id = np.random.choice(self.dataset_ids, 1).item()
        dataset_id = self.id2dataset[task_id]

        dataset_features = torch.FloatTensor(self.meta_features.loc[dataset_id].to_numpy())

        group = self.groups[task_id]

        group = group.drop_duplicates(subset=["metric"])
        idxes = np.random.choice(group.index, self.pipelines_per_step)

        samples = group.loc[idxes]
        metrics = (-1 * samples.metric).to_numpy()  # In data metrics are multiplied by -1
        pipes = []
        for pipe in samples.pipeline_id.to_list():
            with open(self.id2pipe[pipe], "rb", os.O_NONBLOCK) as f:
                pipe_json_string = pickle.load(f)
            pipe = self.pipeline_extractor(pipe_json_string)
            pipes.append(pipe)
        if self.normalize:
            metrics = (metrics - metrics.min()) / metrics.std()
        # Pipes are [N_PIPES, N_FEATURES]
        return pipes, dataset_features, metrics

    def __getitem__(self, idx):
        return self._get_sample()


class HeteroPipelineDataset(Dataset):
    # Load from no_meta_features_and_fedot_pipelines_raw
    # The folder is made with `GAMLET/scripts/generate_dataset_with_hyperparameters_raw.py`.
    def __init__(
        self,
        task_pipe_comb_file: str,
        id2pipe: Dict[int, Union[pathlib.PosixPath, str]],
        id2dataset: Dict[int, str],
        is_val: bool = False,
        encode_type: Optional[Union[str, None]] = "ordinal",
        pipelines_per_step: int = 5,  # Minimum is 2
        use_dataset_with_id: int = None,  # To use single dataset (fold)
        normalize: bool = True,
    ):
        self.use_dataset_with_id = use_dataset_with_id
        self.normalize = normalize

        if pipelines_per_step < 2:
            raise ValueError("`pipelines_per_step`must be >= 2")

        self.pipelines_per_step = pipelines_per_step
        self.task_pipe_comb = pd.read_csv(task_pipe_comb_file)

        if self.use_dataset_with_id is not None:
            self.task_pipe_comb = self.task_pipe_comb[self.task_pipe_comb.task_id == use_dataset_with_id]

        self.groups = {k: v for k, v in self.task_pipe_comb.groupby("task_id")}
        self.id2pipe = id2pipe
        self.id2dataset = id2dataset
        self.dataset_ids = list(self.groups.keys())
        self.is_val = is_val
        self.collate_fn = partial(HeteroPipelineDataset.collate_fn, n_pipelines=self.pipelines_per_step)
        self.pipeline_extractor = fedot_pipeline_features_extractor.FEDOTPipelineFeaturesExtractor2(encode_type)

    @staticmethod
    def collate_fn(
        batch: Sequence[Tuple[Sequence[HeterogeneousData], Sequence[float]]],
        n_pipelines: int,
    ) -> Tuple[Sequence[HeterogeneousBatch], Tensor]:
        pipes = [HeterogeneousBatch.from_heterogeneous_data_list([b[0][i] for b in batch]) for i in range(n_pipelines)]
        y = torch.FloatTensor([b[1] for b in batch])
        # Pipes are [N_PIPES, BATCH_SIZE, N_FEATURES]
        return pipes, y

    def __len__(self):
        return len(self.task_pipe_comb)

    def _get_sample(self) -> Tuple[Sequence[HeterogeneousData], Sequence[float]]:
        task_id = np.random.choice(self.dataset_ids, 1).item()
        group = self.groups[task_id]

        group = group.drop_duplicates(subset=["metric"])
        idxes = np.random.choice(group.index, self.pipelines_per_step)

        samples = group.loc[idxes]
        metrics = (-1 * samples.metric).to_numpy()  # In data metrics are multiplied by -1
        pipes = []
        for pipe in samples.pipeline_id.to_list():
            with open(self.id2pipe[pipe], "rb", os.O_NONBLOCK) as f:
                pipe_json_string = pickle.load(f)
            pipe = self.pipeline_extractor(pipe_json_string)
            pipes.append(pipe)
        if self.normalize:
            metrics = (metrics - metrics.min()) / metrics.std()
        # Pipes are [N_PIPES, N_FEATURES]
        return pipes, metrics

    def __getitem__(self, idx):
        return self._get_sample()
