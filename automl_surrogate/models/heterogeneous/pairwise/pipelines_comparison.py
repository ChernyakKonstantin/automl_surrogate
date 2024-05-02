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

class HeteroPipelineComparisonSurrogateModel(LightningModule):
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
        self.comparator = self.build_comparator(model_parameters["comparator"])

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

    def build_comparator(self, model_parameters: Dict[str, Any]) -> nn.Module:
        class ConcatComparator(nn.Module):
            def __init__(self, in_dim: int):
                super().__init__()
                self.linear = nn.Linear(in_dim, 1)

            def forward(self, pipe1_emb: Tensor, pipe2_emb: Tensor) -> Tensor:
                x = torch.hstack([pipe1_emb, pipe2_emb])
                logit = self.linear(x)
                return torch.sigmoid(logit)

        class CrossAttentionComparator(nn.Module):
            def __init__():
                super().__init__()
                raise NotImplementedError()

        if model_parameters["type"] == "concat":
            return ConcatComparator(self.pipeline_encoder.out_dim * 2)
        elif model_parameters["type"] == "cross-attention":
            return CrossAttentionComparator(self.pipeline_encoder.out_dim * 2)
        else:
            raise ValueError(f"Unknown comparator type {model_parameters['comparator_type']}")

    def encode_pipelines(self, heterogen_pipelines: Sequence[HeterogeneousBatch]) -> Tensor:
        homogen_pipelines = self.homogenize_pipelines(heterogen_pipelines)
        # [N, BATCH, HIDDEN] -> [BATCH, N, HIDDEN]
        pipelines_embeddings = torch.stack([self.pipeline_encoder(x) for x in homogen_pipelines]).permute(1,0,2)
        return pipelines_embeddings

    def training_step(
        self,
        batch: Tuple[Sequence[HeterogeneousBatch], Tensor],
        *args,
        **kwargs,
    ) -> Tensor:
        # For train sequnce length is 2.
        heterogen_pipelines, y = batch
        # [BATCH, 2, HIDDEN]
        pipelines_embeddings = self.encode_pipelines(heterogen_pipelines)
        # [BATCH, 1]
        score = self.comparator(pipelines_embeddings[:, 0, :], pipelines_embeddings[:, 1, :]).squeeze(1)
        score_reversed = self.comparator(pipelines_embeddings[:, 1, :], pipelines_embeddings[:, 0, :]).squeeze(1)
        gt_label = (y[:, 0] > y[:, 1]).to(torch.float32)
        gt_label_reversed = (y[:, 1] > y[:, 0]).to(torch.float32)
        loss = F.binary_cross_entropy(score, gt_label)
        loss_reversed = F.binary_cross_entropy(score_reversed, gt_label_reversed)
        loss = 0.5 * loss + 0.5 * loss_reversed
        self.log("train_loss", loss)
        return loss

    def bubble_argsort(self, pipelines_embeddings: Tensor) -> Tensor:
        # Batch is 1. pipelines_embeddings shape is [BATCH, N, HIDDEN]
        batch_size = pipelines_embeddings.shape[0]
        n = pipelines_embeddings.shape[1]
        indices = torch.LongTensor([list(range(n))] * batch_size).to(self.device)
        for i in range(n-1):
            swapped = False
            for j in range(n-i-1):
                to_swap = self.comparator(pipelines_embeddings[:, j], pipelines_embeddings[:, j+1]).squeeze(1) > 0.5
                swapped = to_swap.any().item()
                if not swapped:
                    continue
                pipelines_embeddings[to_swap, j], pipelines_embeddings[to_swap, j+1] = pipelines_embeddings[to_swap, j+1], pipelines_embeddings[to_swap, j]
                indices[to_swap, j], indices[to_swap, j+1] = indices[to_swap, j+1], indices[to_swap, j]
            if not swapped:
                break
        return indices

    def evaluation_step(
        self,
        prefix: str,
        batch: Tuple[Sequence[HeterogeneousBatch], Tensor],
    ):
        # For train sequnce length is abitrary.
        heterogen_pipelines, y = batch
        with torch.no_grad():
            # [BATCH, N, HIDDEN]
            pipelines_embeddings = self.encode_pipelines(heterogen_pipelines)
            sorted_indices = self.bubble_argsort(pipelines_embeddings)
            # batch_size = pipelines_embeddings.shape[0]
            seq_len = pipelines_embeddings.shape[1]
            scores = []
            for seq_idxes in sorted_indices:
                seq_scores = torch.empty_like(seq_idxes, dtype=torch.float32)
                seq_scores[seq_idxes] = torch.linspace(0, 1, seq_len, device=seq_scores.device)
                scores.append(seq_scores)
            scores = torch.stack(scores)

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
