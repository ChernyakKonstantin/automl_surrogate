"""Draft"""

from functools import partial

from fedot.api.main import Fedot
from golem.core.optimisers.meta.surrogate_optimizer import SurrogateEachNgenOptimizer as SurrogateEachNgenOptimizer_
import numpy as np
from fedot.core.pipelines.adapters import PipelineAdapter
from automl_surrogate.data.fedot_pipeline_features_extractor import FEDOTPipelineFeaturesExtractor2
from automl_surrogate.data.data_types import HeterogeneousBatch
import pathlib
import timeit
from datetime import datetime
from typing import Optional, Tuple
import os
from golem.core.adapter import BaseOptimizationAdapter
from golem.core.log import Log
from golem.core.optimisers.genetic.evaluation import OptionalEvalResult, DelegateEvaluator, SequentialDispatcher
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.meta.surrogate_model import SurrogateModel, RandomValuesSurrogateModel
from golem.core.optimisers.objective.objective import to_fitness, GraphFunction
from golem.core.optimisers.opt_history_objects.individual import GraphEvalResult
from golem.core.optimisers.populational_optimizer import EvaluationAttemptsError, _try_unfit_graph
from golem.core.optimisers.meta.surrogate_evaluator import SurrogateDispatcher as SurrogateDispatcher_
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from automl_surrogate.data import HeteroPipelineAndDatasetFeaturesDataset
from automl_surrogate.models import FusionRankNet
import torch
import yaml
from automl_surrogate.data.fedot_pipeline_features_extractor import FEDOTPipelineFeaturesExtractor2
import pandas as pd


class SurrogateDispatcher(SurrogateDispatcher_):
    """Evaluates objective function with surrogate model.
        Usage: call `dispatch(objective_function)` to get evaluation function.
        Additionally, we need to pass surrogate_model object
    """

    def __init__(self,
                 adapter: BaseOptimizationAdapter,
                 n_jobs: int = 1,
                 graph_cleanup_fn: Optional[GraphFunction] = None,
                 delegate_evaluator: Optional[DelegateEvaluator] = None,
                 surrogate_model: SurrogateModel = RandomValuesSurrogateModel()):
        super().__init__(adapter, n_jobs, graph_cleanup_fn, delegate_evaluator)
        self._n_jobs = 1
        self.surrogate_model = surrogate_model

    def evaluate_population(self, individuals: "PopulationT") -> "Optional[PopulationT]":
        individuals_to_evaluate, individuals_to_skip = self.split_individuals_to_evaluate(individuals)
        graphs = [ind.graph for ind in individuals_to_evaluate]
        uids = [ind.uid for ind in individuals_to_evaluate]
        evaluation_results = self.evaluate_batch(graphs, uids)
        # evaluation_results = [self.evaluate_single(ind.graph, ind.uid) for ind in individuals_to_evaluate]
        individuals_evaluated = self.apply_evaluation_results(individuals_to_evaluate, evaluation_results)
        evaluated_population = individuals_evaluated + individuals_to_skip or None
        return evaluated_population

    def evaluate_batch(self, graphs: "List[OptGraph]", uids: "List[str]") -> "List[GraphEvalResult]":
        start_time = timeit.default_timer()
        predictions = self.surrogate_model(graphs, objective=self._objective_eval)
        fitnesses = [to_fitness(p) for p in predictions]
        end_time = timeit.default_timer()
        eval_reses = []
        for fitness, uid_of_individual, graph in zip(fitnesses, uids, graphs):
            eval_res = GraphEvalResult(
                uid_of_individual=uid_of_individual, fitness=fitness, graph=graph, metadata={
                    'computation_time_in_seconds': end_time - start_time,
                    'evaluation_time_iso': datetime.now().isoformat(),
                    'surrogate_evaluation': True
                }
            )
            eval_reses.append(eval_res)
        return eval_reses


class SurrogateEachNgenOptimizer(SurrogateEachNgenOptimizer_):
    def __init__(self,
                 objective: "Objective",
                 initial_graphs: "Sequence[OptGraph]",
                 requirements: "GraphRequirements",
                 graph_generation_params: "GraphGenerationParams",
                 graph_optimizer_params: "GPAlgorithmParameters",
                 surrogate_model=RandomValuesSurrogateModel(),
                 surrogate_each_n_gen=5
                 ):
        super().__init__(objective, initial_graphs, requirements, graph_generation_params, graph_optimizer_params, surrogate_model, surrogate_each_n_gen)
        self.surrogate_dispatcher = SurrogateDispatcher(adapter=graph_generation_params.adapter,
                                                        n_jobs=requirements.n_jobs,
                                                        graph_cleanup_fn=_try_unfit_graph,
                                                        delegate_evaluator=graph_generation_params.remote_evaluator,
                                                        surrogate_model=surrogate_model)

    def optimise(self, objective: "ObjectiveFunction") -> "Sequence[OptGraph]":
        # # eval_dispatcher defines how to evaluate objective on the whole population
        # evaluator = self.eval_dispatcher.dispatch(objective, self.timer)
        # surrogate_dispatcher defines how to evaluate objective with surrogate model
        surrogate_evaluator = self.surrogate_dispatcher.dispatch(objective, self.timer)

        with self.timer, self._progressbar:
            # self._initial_population(evaluator)
            self._initial_population(surrogate_evaluator)
            while not self.stop_optimization():
                try:
                    new_population = self._evolve_population(surrogate_evaluator)
                    # if self.generations.generation_num % self.surrogate_each_n_gen == 0:
                    #     new_population = self._evolve_population(surrogate_evaluator)
                    # else:
                    #     new_population = self._evolve_population(evaluator)
                except EvaluationAttemptsError as ex:
                    self.log.warning(f'Composition process was stopped due to: {ex}')
                    return [ind.graph for ind in self.best_individuals]
                # Adding of new population to history
                self._update_population(new_population)
        self._update_population(self.best_individuals, 'final_choices')
        return [ind.graph for ind in self.best_individuals]

class SurrogatePipeline:
    def __init__(self):
        self.adapter = PipelineAdapter()
        self.pipe_ext = FEDOTPipelineFeaturesExtractor2(operation_encoding="ordinal")

        dataset_root = "/home/cherniak/itmo_job/GAMLET/data/meta_features_and_fedot_pipelines"
        self.ds = HeteroPipelineAndDatasetFeaturesDataset(
            root=dataset_root,
            task_pipe_comb_file="test_task_pipe_comb.json",
            meta_features_file="openml.csv",  #pymfe
            encode_type="ordinal",
            pipelines_per_step=35,
            normalize_meta_features="minmax",
            pipelines_dir="pipelines_as_json",
            # metric_sign="+"
        )

        dataset_meta_feats_all = pd.read_csv("/home/cherniak/itmo_job/surrogate/scripts/evaluate_on_test_datasets/openml_meta_feats.csv", index_col=0)
        self.dataset_meta_feats_all = pd.DataFrame(
            data=self.ds.scaler.transform(dataset_meta_feats_all.to_numpy()),
            index=dataset_meta_feats_all.index,
            columns=dataset_meta_feats_all.columns,
        )

        root = "/home/cherniak/itmo_job/surrogate/experiment_logs/meta_features_and_fedot_pipelines_(type_and_hparams)/train_node_embedder_from_scratch/simple_graph_encoder/rank_over_10/minmax_metafeats/openml/fusion_ranknet/version_0"
        ckpt_name = "epoch=2-step=17000"

        config_path = os.path.join(root, "config.yml")
        ckpt_path = os.path.join(root, "checkpoints", f"{ckpt_name}.ckpt")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = FusionRankNet(**{k: v for k,v in config["model"].items() if k != "class"})
        model.load_state_dict(ckpt["state_dict"])
        model = model.eval()
        self.model = model

    def set_dataset(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.dataset_meta_feats = torch.FloatTensor(self.dataset_meta_feats_all.loc[dataset_id].to_numpy()).reshape(1,-1)

    def __call__(self, graphs: "List[OptGraph]", *args, **kwargs) -> "List[float]":
        jsons_strs = [self.adapter._restore(g).save()[0] for g in graphs]
        pipes = [self.pipe_ext(g) for g in jsons_strs]
        batch = [HeterogeneousBatch.from_heterogeneous_data_list([p]) for p in pipes]
        with torch.no_grad():
            scores = self.model.forward(batch, self.dataset_meta_feats)
        return scores.reshape(-1,1).numpy().tolist()
        # return (-1 * scores).reshape(-1,1).numpy().tolist()


def create_and_test_pipeline(
        dataset_name: str,
        x: "np.ndarray",
        y: "np.ndarray",
        x_test: "np.ndarray",
        y_test: "np.ndarray",
        with_surrogate: "bool" = True,
    ):
    if with_surrogate:
        print("Creating surrogate")
        sur_pipe = SurrogatePipeline()
        sur_pipe.set_dataset(dataset_name)
    else:
        print("Usual FEDOT")

    # create FEDOT with SurrogateEachNgenOptimizer
    print("Creating FEDOT")
    kwargs = dict(
        problem='classification',
        timeout=5,
        num_of_generations=5,
        n_jobs=1,
        parallelization_mode="sequential",
        metric=ClassificationMetricsEnum.ROCAUC,
        with_tuning=False,
        # with_tuning=True,
        cv_folds=None,
        # validation_blocks=2,
        preset='best_quality',
        logging_level=10,
    )
    if with_surrogate:
        kwargs["optimizer"] = partial(SurrogateEachNgenOptimizer, surrogate_model=sur_pipe, surrogate_each_n_gen=1)

    model = Fedot(**kwargs, use_pipelines_cache=False, use_preprocessing_cache=False)
    print("Fitting FEDOT")
    pipeline = model.fit(features=x, target=y)
    print("Predicting FEDOT")
    model.predict(x_test)
    print(model.get_metrics(target=y_test))
