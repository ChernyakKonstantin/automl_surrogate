"""The module contains custom method to train `lib.lightning_modules.GraphTransformer`."""

import os
import pickle
from typing import Any, Dict, List, Tuple
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from automl_surrogate.data import HeteroPipelineAndDatasetFeaturesDataset, HeteroPipelineDataset

import automl_surrogate.models as surrogate_module
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import numpy as np

torch.autograd.set_detect_anomaly(True)

def build_datasets(config: Dict[str, Any]) -> Tuple[Dataset, Dataset, Dataset]:
    print("Making train dataset")
    # train_dataset = HeteroPipelineDataset(
    #     train_task_pipe_comb_path,
    #     id2pipe,
    #     id2dataset,
    #     is_val=False,
    #     pipelines_per_step=config["train_dataset"]["pipelines_per_step"],
    #     use_dataset_with_id=config["train_dataset"]["use_dataset_with_id"],
    #     normalize=config["train_dataset"]["normalize"],
    # )
    train_dataset = HeteroPipelineAndDatasetFeaturesDataset(**config["train_dataset"])
    print("Making test dataset")
    # val_dataset = HeteroPipelineDataset(
    #     test_task_pipe_comb_path,
    #     id2pipe,
    #     id2dataset,
    #     is_val=True,
    #     pipelines_per_step=config["val_dataset"]["pipelines_per_step"],
    #     use_dataset_with_id=config["val_dataset"]["use_dataset_with_id"],
    #     normalize=config["val_dataset"]["normalize"],

    # )
    val_dataset = HeteroPipelineAndDatasetFeaturesDataset(**config["val_dataset"])
    assert len(train_dataset) != 0
    assert len(val_dataset) != 0
    return train_dataset, val_dataset, val_dataset

def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        batch_size=config["batch_size"],
        num_workers=config["num_dataloader_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=val_dataset.collate_fn,
        batch_size=config["batch_size"],
        num_workers=config["num_dataloader_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        collate_fn=test_dataset.collate_fn,
        batch_size=config["batch_size"],
        num_workers=config["num_dataloader_workers"],
    )
    return train_loader, val_loader, test_loader

def build_model(config: Dict[str, Any]) -> nn.Module: # TODO: implement
    model_class = getattr(surrogate_module, config["model"]["class"])
    dim_feedforward = 2 * config["model"]["model_parameters"]["pipeline_encoder"]["gnn"]["d_model"]
    config["model"]["model_parameters"]["pipeline_encoder"]["gnn"]["dim_feedforward"] = dim_feedforward
    model = model_class(**{k: v for k, v in config["model"].items() if k != "class"})
    return model

def train_hetero_surrogate_model(config: Dict[str, Any]): # -> List[Dict[str, float]]:
    """Create surrogate model and do training according to config parameters."""
    print("Making datasets")
    train_dataset, val_dataset, test_dataset = build_datasets(config["dataset_params"])

    print("Making dataloaders")
    train_loader, val_loader, test_loader = build_dataloaders(train_dataset, val_dataset, test_dataset, config)

    print("Making model")
    model = build_model(config)

    print("Making auxiliary stuff")
    if config["tensorboard_logger"] is not None:
        logger = TensorBoardLogger(**config["tensorboard_logger"])
    else:
        logger = None

    model_checkpoint_callback = ModelCheckpoint(**config["model_checkpoint_callback"])

    if config["early_stopping_callback"] is not None:
        early_stopping_callback = EarlyStopping(**config["early_stopping_callback"])
    else:
        early_stopping_callback = None

    print("Making trainer")
    trainer = Trainer(
        **config["trainer"],
        logger=logger,
        callbacks=[c for c in [model_checkpoint_callback, early_stopping_callback] if c is not None],
        gradient_clip_val=0.5,
    )
    print("Training")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # model = type(model).load_from_checkpoint(model_checkpoint_callback.best_model_path)
    # print(model_checkpoint_callback.best_model_path)

    # test_results = trainer.test(model, dataloaders=test_loader)
    # return test_results


# def test_ranking(config: Dict[str, Any]) -> List[Dict[str, float]]:
#     """Test surrogate model"""
#     print("Making datasets")
#     _, _, test_dataset, _ = build_datasets(config)
#     assert len(test_dataset) != 0

#     print("Making dataloaders")
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=config["batch_size"],
#         num_workers=0,  # Change to 1.
#         collate_fn=test_dataset.collate_fn,
#     )

#     model_class = getattr(surrogate_model, config["model"].pop("name"))
#     chpoint_dir = config["model_data"]["save_dir"] + "checkpoints/"
#     model = model_class.load_from_checkpoint(
#         checkpoint_path=chpoint_dir + os.listdir(chpoint_dir)[0],
#         hparams_file=config["model_data"]["save_dir"] + "hparams.yaml"
#     )
#     model.eval()

#     task_ids, pipe_ids, y_preds, y_trues = [], [], [], []
#     with torch.no_grad():
#         for batch in test_loader:
#             surrogate_model.test_step(batch)
#             res = surrogate_model.test_step_outputs.pop()
#             task_ids.append(res['task_id'])
#             pipe_ids.append(res['pipe_id'])
#             y_preds.append(res['y_pred'])
#             y_trues.append(res['y_true'])

#     df = pd.DataFrame({'task_id': np.concatenate(task_ids),
#                        'pipe_id': np.concatenate(pipe_ids),
#                        'y_pred': np.concatenate(y_preds),
#                        'y_true': np.concatenate(y_trues)})

#     with open(config["dataset_params"]["root_path"] + "/pipelines_fedot.pickle", "rb") as input_file:
#         pipelines_fedot = pickle.load(input_file)

#     res = df.loc[df.groupby(['task_id'])['y_pred'].idxmax()]
#     res['model_str'] = [str(pipelines_fedot[i]) for i in res.pipe_id.values]
#     res = res[['task_id', 'y_true', 'model_str']]
#     res['y_true'] = -res['y_true']
#     res.columns = ['dataset', 'fitness', 'model_str']

#     res.to_csv('surrogate_test_set_prediction.csv', index=False)
