import numpy as np
from tqdm import tqdm
import os
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
import pickle
import json
from optimize_with_surrogate import create_and_test_pipeline

dataset_names = [
    "numerai28_6",
    "sf-police-incidents",
    "airlines",
    "click_prediction_small",  #
    "albert",
    "kddcup09_appetency",
    "higgs",  #
    "christine",
    "guillermo",
    "amazon_employee_access",
]


def get_xy(
        datasets_dir: "os.Pathlike",
        dataset_name: "str",
        fold_id: "int",
        split: "str",
    ) -> "Tuple[np.ndarray, np.ndarray]":
    x_name = f"{split}_{dataset_name}_fold{fold_id}.npy"
    y_name = f"{split}y_{dataset_name}_fold{fold_id}.npy"
    x = np.load(os.path.join(datasets_dir, x_name))
    y = np.load(os.path.join(datasets_dir, y_name))
    return x, y

def make_input_data(
        datasets_dir: "os.Pathlike",
        dataset_name: "str",
        fold_id: "int",
        split: "str",
    ) -> "InputData":
    x, y = get_xy(datasets_dir, dataset_name, fold_id, split)
    data = InputData(
        idx = np.arange(len(x)), # TODO: is it correct?
        task = Task(TaskTypesEnum.classification),
        data_type = DataTypesEnum.table,
        features=x,
        target=y,
    )
    return data

def update_pipeline_params(pipeline_json: "Dict[str, Any]", new_params: "Dict[str, Any]") -> "Dict[str, Any]":
    nodes = pipeline_json["nodes"]
    operation_id2node = {n["operation_id"]: n for n in nodes}
    for key, value in new_params.items():
        operation_id = int(key.split("||")[0].strip())
        param_name = key.split(" | ")[1].strip()

        operation_id2node[operation_id]["custom_params"].update({param_name: value})
    pipeline_json["nodes"] = list(operation_id2node.values())
    return pipeline_json

def make_pipeline(
    dataset_name: "str",
    fold_id: "int",
    model_name: "str",
    study_id: "int",
    study_dir: "os.Pathlike",
    pipeline_dir: "os.Pathlike",
) -> "Pipeline":
    study_path = f"{study_dir}/{dataset_name}_{fold_id}/{model_name}.pickle"
    with open(study_path, "rb") as f:
        study = pickle.load(f)

    pipeline_path = f"{pipeline_dir}/{dataset_name}_{fold_id}/models/{model_name}.json"
    with open(pipeline_path, "r") as f:
        pipeline_json = json.load(f)

    pipeline_json = update_pipeline_params(pipeline_json, study.trials[study_id].params)
    pipeline = Pipeline()
    pipeline.load(pipeline_json, dict_fitted_operations=None)
    return pipeline


def fit_pipeline(
        pipeline: "Pipeline",
        datasets_dir: "os.Pathlike",
        dataset_name: "str",
        fold_id: "int",
        split: "str",
    ):
    input_data = make_input_data(datasets_dir, dataset_name, fold_id, split)
    pipeline.fit_from_scratch(input_data)
    print(pipeline.get_metrics())

def main(
    dataset_name: "str",
    fold_id: "int",
    model_name: "str",
    study_id: "int",
    study_dir: "os.Pathlike",
    pipeline_dir: "os.Pathlike",
    datasets_dir: "os.Pathlike",
):
    pipeline = make_pipeline(dataset_name, fold_id, model_name, study_id, study_dir, pipeline_dir)
    fit_pipeline(pipeline, datasets_dir, dataset_name, fold_id, split="train")

def cross_validation_fedot_with_surrogate(
        datasets_dir: "os.Pathlike",
        dataset_name: "str",
        fold_id: "int",
):
    train_x, train_y = get_xy(datasets_dir, dataset_name, fold_id, split="train")
    test_x, test_y = get_xy(datasets_dir, dataset_name, fold_id, split="test")
    create_and_test_pipeline(f"{dataset_name}_{fold_id}", train_x, train_y, test_x, test_y, with_surrogate=True)

def cross_validation_fedot(
        datasets_dir: "os.Pathlike",
        dataset_name: "str",
        fold_id: "int",
):
    train_x, train_y = get_xy(datasets_dir, dataset_name, fold_id, split="train")
    tets_x, test_y = get_xy(datasets_dir, dataset_name, fold_id, split="train")
    create_and_test_pipeline(f"{dataset_name}_{fold_id}", train_x, train_y, tets_x, test_y, with_surrogate=False)

# if __name__ == "__main__":
#     dataset_name = "australian"
#     fold_id = 0
#     model_name = "feda07b8-4f4a-41ea-8a12-a0ef4089a4c2"
#     study_id = 0
#     study_dir = "/home/cherniak/itmo_job/graphs_with_hyperparameters"
#     pipeline_dir = "/home/cherniak/itmo_job/GAMLET/data/knowledge_base_1_v2/datasets"
#     datasets_dir = "/home/cherniak/itmo_job/datasets_folds"
#     main(dataset_name, fold_id, model_name, study_id, study_dir, pipeline_dir, datasets_dir)

if __name__ == "__main__":
    dataset_name = "albert"
    fold_id = 8
    datasets_dir = "/home/cherniak/itmo_job/datasets_folds"
    print(dataset_name, fold_id)
    try:
        cross_validation_fedot(datasets_dir, dataset_name, fold_id)
    except:
        print("=" * 50)
    try:
        cross_validation_fedot_with_surrogate(datasets_dir, dataset_name, fold_id)
    except:
        print("=" * 50)