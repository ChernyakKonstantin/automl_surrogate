from typing import Any, Dict, List, Tuple, Sequence
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torch_geometric.data import Batch
import torch.optim as optim
from automl_surrogate.data.heterogeneous import HeterogeneousBatch
from automl_surrogate.layers.encoders import GraphTransformer, SimpleGNNEncoder
from automl_surrogate.models.heterogeneous.node_embedder import build_node_embedder
import automl_surrogate.losses as losses_module
import automl_surrogate.metrics as metrics_module
import torch.nn.functional as F
from typing import Iterable

class HeteroPipelineRegressionSurrogateModel(LightningModule):
    # Same hypothesis as in https://arxiv.org/pdf/1912.05891.pdf
    def __init__(
        self,
        model_parameters: Dict[str, Any],
        loss_fn: str,
        validation_metrics: List[str],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_name = loss_fn
        self.validation_metrics = validation_metrics

        self.node_embedder = build_node_embedder(model_parameters["node_embedder"])
        self.pipeline_encoder = self.build_pipeline_encoder(
            self.node_embedder.out_dim,
            model_parameters["pipeline_encoder"],
        )
        self.linear = nn.Linear(self.pipeline_encoder.out_dim, 1)

    @staticmethod
    def build_pipeline_encoder(in_dim: int, config: Dict[str, Any]) -> nn.Module:
        config["in_size"] = in_dim
        if config["type"] == "simple_graph_encoder":
            return SimpleGNNEncoder(**{k: v for k, v in config.items() if k != "type"})
        elif config["type"] == "graph_transformer":
            return GraphTransformer(**{k: v for k, v in config.items() if k != "type"})

    def homogenize_pipelines(self, heterogen_pipelines: Sequence[HeterogeneousBatch]) -> List[Batch]:
        node_embeddings = [self.node_embedder(p) for p in heterogen_pipelines]
        out_dim = self.node_embedder.out_dim
        return [p.to_pyg_batch(out_dim, emb) for p, emb in zip(heterogen_pipelines, node_embeddings)]

    def encode_pipelines(self, heterogen_pipelines: Sequence[HeterogeneousBatch]) -> Tensor:
        homogen_pipelines = self.homogenize_pipelines(heterogen_pipelines)
        # [N, BATCH, HIDDEN] -> [BATCH, N, HIDDEN]
        pipelines_embeddings = torch.stack([self.pipeline_encoder(x) for x in homogen_pipelines]).permute(1,0,2)
        return pipelines_embeddings

    def forward(self, heterogen_pipelines: Sequence[HeterogeneousBatch]) -> Tensor:
        # [BATCH, N, HIDDEN_1]
        pipelines_embeddings = self.encode_pipelines(heterogen_pipelines)
        # [BATCH, N]
        scores = self.linear(pipelines_embeddings).squeeze(2)
        return scores

    def training_step(
        self,
        batch: Tuple[Sequence[HeterogeneousBatch], Tensor],
        *args,
        **kwargs,
    ) -> Tensor:
        # For train sequnce length is 2.
        heterogen_pipelines, y = batch
        scores = self.forward(heterogen_pipelines)
        difference = scores[:, 0] - scores[:, 1]
        gt_label = (y[:, 0] > y[:, 1]).to(torch.float32)
        loss = F.binary_cross_entropy(torch.sigmoid(difference), gt_label)
        self.log("train_loss", loss)
        return loss

    def evaluation_step(
        self,
        prefix: str,
        batch: Tuple[Sequence[HeterogeneousBatch], Tensor],
    ):
        # For train sequnce length is abitrary.
        heterogen_pipelines, y = batch
        with torch.no_grad():
            scores = self.forward(heterogen_pipelines)

        if "ndcg" in self.validation_metrics:
            metric_fn = getattr(metrics_module, "ndcg")
            self.log(f"{prefix}_ndcg", metric_fn(y.cpu(), scores.cpu()))
        if "precision" in self.validation_metrics:
            metric_fn = getattr(metrics_module, "precision")
            self.log(f"{prefix}_precision", metric_fn(y, scores))
        if "kendalltau" in self.validation_metrics:
            metric_fn = getattr(metrics_module, "kendalltau")
            self.log(f"{prefix}_kendalltau", metric_fn(y.cpu(), scores.cpu()))

    def validation_step(
        self,
        batch: Tuple[Sequence[HeterogeneousBatch], Tensor],
        *args,
        **kwargs,
    ):
        self.evaluation_step("val", batch)

    def test_step(
        self,
        batch: Tuple[Sequence[HeterogeneousBatch], Tensor],
        *args,
        **kwargs,
    ):
        self.evaluation_step("test", batch)

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def transfer_batch_to_device(self, batch: Sequence, device: torch.device, dataloader_idx: int) -> Sequence:
        def transfer_heterogeneous_batch(heterogen_batch: HeterogeneousBatch) -> HeterogeneousBatch:
            heterogen_batch.batch = heterogen_batch.batch.to(device)
            heterogen_batch.ptr = heterogen_batch.ptr.to(device)
            heterogen_batch.edge_index = heterogen_batch.edge_index.to(device)
            heterogen_batch.node_idxes_per_type = {k: v.to(device) for k, v in heterogen_batch.node_idxes_per_type.items()}
            heterogen_batch.hparams = {k: v.to(device) for k, v in heterogen_batch.hparams.items()}
            heterogen_batch.encoded_type = {k: v.to(device) for k, v in heterogen_batch.encoded_type.items()}
            return heterogen_batch

        res = []
        for e in batch:
            if isinstance(e, Tensor):
                res.append(e.to(device))
            elif isinstance(e, HeterogeneousBatch):
                e = transfer_heterogeneous_batch(e)
                res.append(e)
            elif isinstance(e, Iterable) and isinstance(e[0], HeterogeneousBatch):
                res.append([transfer_heterogeneous_batch(h) for h in e])
            else:
                raise TypeError(f"Uknown type f{type(e)}")
        return res
