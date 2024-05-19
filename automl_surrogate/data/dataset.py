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
from torch_geometric.data import SQLiteDatabase
# from torch_geometric.loader import DataLoader
from itertools import chain
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import random


class HeteroPipelineAndDatasetFeaturesDataset(Dataset):
    def __init__(
        self,
        root: str,
        task_pipe_comb_file: str,
        meta_features_file: str = "pymfe.csv",
        pipelines_dir: str = "pipelines",
        encode_type: Optional[Union[str, None]] = "ordinal",
        pipelines_per_step: int = 5,  # Minimum is 2
        normalize: bool = False,
        normalize_meta_features: str = None,  # minmax or standart
        use_dataset_with_id: str = None
    ):
        split = task_pipe_comb_file.split("_")[0]
        datasets_dict_file = f"{split}_datasets_dict.pickle"
        metrics2occurences_file = f"{split}_metrics2occurences_dict.pickle"
        
        self.root = root
        self.pipelines_dir = pipelines_dir
        self.normalize = normalize
        self.normalize_meta_features = normalize_meta_features

        if pipelines_per_step < 2:
            raise ValueError("`pipelines_per_step`must be >= 2")
        self.pipelines_per_step = pipelines_per_step
        
        self.meta_features = pd.read_csv(os.path.join(self.root, meta_features_file), index_col=0)
        if self.normalize_meta_features == "minmax":
            data = MinMaxScaler().fit_transform(self.meta_features.to_numpy())
            self.meta_features = pd.DataFrame(data=data, columns=self.meta_features.columns, index=self.meta_features.index)
        self.n_meta_features = self.meta_features.shape[1]
        
        try:
            with open(os.path.join(root, datasets_dict_file), "rb") as f:
                self.datasets = pickle.load(f)
            with open(os.path.join(root, metrics2occurences_file), "rb") as f:
                self.metrics2occurences = pickle.load(f)
            self.dataset_ids = list(self.datasets.keys())
            self.n_records = sum([len(v) for v in self.datasets.values()])
        except FileNotFoundError:   
            raise FileNotFoundError
            # self.task_pipe_comb = pd.read_json(os.path.join(self.root, task_pipe_comb_file), orient="table")
            # if use_dataset_with_id is not None:
            #     self.task_pipe_comb = self.task_pipe_comb.loc[[use_dataset_with_id]]
            # # In data metrics are multiplied by -1
            # self.task_pipe_comb["metric"] = -1 * self.task_pipe_comb["metric"]
            # self.n_records = len(self.task_pipe_comb)
            # self.dataset_ids = self.task_pipe_comb.index.get_level_values(0).unique().to_list()
            
            # def process_dataset(dataset: pd.DataFrame):
            #     return dataset.groupby("metric").apply(lambda x: x.reset_index(drop=True)).drop(columns=["metric"])
                
            # self.datasets = {ds_id: process_dataset(self.task_pipe_comb.loc[ds_id]) for ds_id in self.dataset_ids}
            # self.metrics2occurences = {ds_id: self.datasets[ds_id].groupby("metric").apply(len) for ds_id in self.dataset_ids}
            
            # with open(os.path.join(root, "train_datasets_dict.pickle"), "wb") as f:
            #     pickle.dump(self.datasets, f)
            # with open(os.path.join(root, "train_metrics2occurences_dict.pickle"), "wb") as f:
            #     pickle.dump(self.metrics2occurences, f)
        
        self.dataset_ids = [e for e in self.dataset_ids if not e.startswith("mfeat-factors_5")]
            
        # self.groups = {k: v for k, v in self.task_pipe_comb.groupby("task_id")}




        # self.pipeline_extractor = fedot_pipeline_features_extractor.FEDOTPipelineFeaturesExtractor2(
        #     encode_type,
        # )
        # self.db = SQLiteDatabase("/home/cherniak/itmo_job/surrogate/pipelines.db", "pipelines")

        self.collate_fn = partial(
            HeteroPipelineAndDatasetFeaturesDataset.collate_fn,
            n_pipelines=self.pipelines_per_step,
            # db=self.db,
            # pipeline_extractor=self.pipeline_extractor
        )

    @staticmethod
    def collate_fn(
        batch: Sequence[Tuple[Sequence[HeterogeneousData], Sequence[float], Tensor]],
        n_pipelines: int,
        # db: SQLiteDatabase,
        # pipeline_extractor,
    ) -> Tuple[Sequence[HeterogeneousBatch], Tensor, Tensor]:
        # # db = SQLiteDatabase("/home/cherniak/itmo_job/surrogate/pipelines.db", "pipelines")
        # pipes = []
        # for i in range(n_pipelines):
        #     pipe_ids = [b[0][i] for b in batch]
        #     pipes_i = [db.get(index) for index in pipe_ids]
        #     pipes_i = list(map(pipeline_extractor, pipes_i))
        #     pipes.append(HeterogeneousBatch.from_heterogeneous_data_list(pipes_i))
        # # db.close()
        pipes = [HeterogeneousBatch.from_heterogeneous_data_list([b[0][i] for b in batch]) for i in range(n_pipelines)]
        dataset_features = torch.vstack([b[1] for b in batch])
        y = torch.FloatTensor([b[2] for b in batch])
        # Pipes are [N_PIPES, BATCH_SIZE, N_FEATURES]
        return pipes, dataset_features, y

    def __len__(self):
        return self.n_records

    def _get_pipeline_path(self, dataset_id: str, x: pd.Series) -> str:
            return os.path.join(self.root, self.pipelines_dir, dataset_id, x["architecture_id"], x["hparam_id"] + ".json")  # ".pickle"

    def _get_sample(self) -> Tuple[Sequence[HeterogeneousData], Tensor, Sequence[float]]:
        dataset_id = np.random.choice(self.dataset_ids, 1).item()
        dataset_features = torch.FloatTensor(self.meta_features.loc[dataset_id].to_numpy())
        dataset: pd.DataFrame = self.datasets[dataset_id]
        dataset_metrics: pd.Series = self.metrics2occurences[dataset_id]
        selected_metrics = np.random.choice(dataset_metrics.index, self.pipelines_per_step, replace=False)
        dataset_metrics = dataset_metrics[selected_metrics]
        tuples = []
        for metric in dataset_metrics.index:
            n_records = dataset_metrics[metric]
            record_idx = random.randint(0, n_records-1)
            tuples.append((metric, record_idx))
        selected_indexes = pd.MultiIndex.from_tuples(tuples, names=["metric", None])
        dataset = dataset.loc[selected_indexes]
        pipeline_paths = dataset.apply(lambda x: self._get_pipeline_path(dataset_id, x), axis=1).to_list()
        pipes = [HeterogeneousData.from_json(pipeline_path) for pipeline_path in pipeline_paths]
        metrics = dataset.index.get_level_values(0).to_list()

        # pipes = samples.pipeline_id.to_list() #list(range(self.pipelines_per_step)) # [self.db.get(i) for i in range(self.pipelines_per_step)]
        # for pipeline_path in pipeline_paths:
        #     with open(pipeline_path, "rb") as f:
        #         pipe = pickle.load(f)
        #     if self.pipelines_dir == "pipelines":
        #         pipe = self.pipeline_extractor(pipe)
        #     pipes.append(pipe)
        # if self.normalize:
        #     metrics = (metrics - metrics.min()) / metrics.std()
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
