from automl_surrogate.data import HeterogeneousData
from automl_surrogate.models.hyperparams_embedder import PretrainedHyperparametersEmbedder
import torch
from automl_surrogate.data.dataset import HeteroPipelineAndDatasetFeaturesDataset
import os
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse

def get_pipeline_path(root: str, pipelines_dir: str, dataset_id: str, architecture_id: str, hparam_id: str) -> str:
    return os.path.join(root, pipelines_dir, dataset_id, architecture_id, hparam_id + ".pickle")

def main(task_pipe_comb_file: str):
    embedder_params = {
        "autoencoder_ckpt_path": "/home/cherniak/itmo_job/surrogate/experiment_logs/embed_hyperparameters/to_8_with_learnables/mse_loss__no_dropout/checkpoints/epoch=9-step=980.ckpt",
        "out_dim": 8,
        "trainable": False,
    }

    op_hyperparams_embedder = PretrainedHyperparametersEmbedder(**embedder_params)

    dataset_params = {
        "root": "/home/cherniak/itmo_job/GAMLET/data/meta_features_and_fedot_pipelines_raw",
        "task_pipe_comb_file": task_pipe_comb_file,
        "pipelines_dir": "pipelines",
    }

    dataset = HeteroPipelineAndDatasetFeaturesDataset(**dataset_params)

    for (dataset_id, _), (architecture_id, hparam_id, _, _) in tqdm(dataset.task_pipe_comb.iterrows(), total=len(dataset.task_pipe_comb)):
        # TODO: Fix task_pipe_comb
        if dataset_id.startswith("mfeat"):
            continue

        orig_pipeline_path = get_pipeline_path(
            root=dataset_params["root"],
            pipelines_dir="pipelines",
            dataset_id=dataset_id,
            architecture_id=architecture_id,
            hparam_id=hparam_id,
        )
        new_pipeline_path = get_pipeline_path(
            root=dataset_params["root"],
            pipelines_dir="pipelines_homogeneous_hparams",
            dataset_id=dataset_id,
            architecture_id=architecture_id,
            hparam_id=hparam_id,
        )
        if os.path.exists(new_pipeline_path):
            continue

        with open(orig_pipeline_path, "rb") as f:
            pipe_json_string = pickle.load(f)
        pipe = dataset.pipeline_extractor(pipe_json_string)
        with torch.no_grad():
            pipe.hparams = op_hyperparams_embedder(pipe.hparams)

        Path(new_pipeline_path).parent.mkdir(parents=True, exist_ok=True)
        with open(new_pipeline_path, "wb") as f:
            pickle.dump(pipe, f)

if __name__ == "__main__":
    """Launch with command:
    `python3 -W ignore precompute.py -t train_task_pipe_comb.json`
    or
    `python3 -W ignore precompute.py -t "test_task_pipe_comb.json"`
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_pipe_comb_file")
    args = parser.parse_args()
    main(args.task_pipe_comb_file)