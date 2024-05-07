from automl_surrogate.models.heterogeneous.listwise.pipelines_ranking import HeteroPipelineRankingSurrogateModel
from typing import Any, Dict, List, Tuple, Sequence, Optional, Union, Callable, Iterable
from torch import Tensor
from automl_surrogate.data.heterogeneous import HeterogeneousBatch
import torch.nn as nn
import torch
import automl_surrogate.metrics as metrics_module
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_activation_fn
from pytorch_lightning import LightningModule
from automl_surrogate.models.heterogeneous.node_embedder import build_node_embedder

class CrossAttentionTransformerEncoder(nn.TransformerEncoder):

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        for mod in self.layers:
            output = mod(q, kv)

        if self.norm is not None:
            output = self.norm(output)

        return output

class CrossAttentionTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model: int, kdim: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(nn.TransformerEncoderLayer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                               kdim=kdim, vdim=kdim, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.activation = activation

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        x = self.norm1(q + self._ca_block(q, kv))
        x = self.norm2(x + self._ff_block(x))
        return x

    # cross-attention block
    def _ca_block(self, query: Tensor, kv: Tensor) -> Tensor:
        # [Batch, M] -> [Batch, 1, M]
        kv = kv.unsqueeze(1)
        x = self.cross_attn(query, kv, kv, need_weights=False)[0]
        return self.dropout1(x)

def build_dataset_encoder(config: Dict[str, Any]) -> nn.Module:
    dataset_encoder = nn.Sequential(
        nn.Linear(config["in_size"], config["hidden_dim"]),
        nn.ReLU(),
        nn.Linear(config["hidden_dim"], config["hidden_dim"]),
        nn.ReLU(),
    )
    dataset_encoder.out_dim = config["hidden_dim"]
    return dataset_encoder

class DataHeteroPipelineRankingSurrogateModel(HeteroPipelineRankingSurrogateModel):
    def __init__(
        self,
        model_parameters: Dict[str, Any],
        loss_fn: str,
        validation_metrics: List[str],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super(HeteroPipelineRankingSurrogateModel, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_name = loss_fn
        self.validation_metrics = validation_metrics

        self.node_embedder = build_node_embedder(model_parameters["node_embedder"])
        self.pipeline_encoder = self.build_pipeline_encoder(
            self.node_embedder.out_dim,
            model_parameters["pipeline_encoder"],
        )
        self.dataset_encoder = build_dataset_encoder(model_parameters["dataset_encoder"])
        self.mhca_block = self.build_mhca_block(
            self.pipeline_encoder.out_dim,
            self.dataset_encoder.out_dim,
            model_parameters["mhca_block"],
        )
        self.mhsa_block = self.build_mhsa_block(
            self.mhca_block.out_dim,
            model_parameters["mhsa_block"],
        )
        self.linear = nn.Linear(self.mhsa_block.out_dim, 1)

    @staticmethod
    def build_mhca_block(qdim: int, kdim: int, config: Dict[str, Any]) -> nn.TransformerEncoder:
        transformer_layer = CrossAttentionTransformerEncoderLayer(
            d_model=qdim,
            kdim=kdim,
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            batch_first=True,
        )
        mhca_block = CrossAttentionTransformerEncoder(
            transformer_layer,
            num_layers=config["num_layers"],
        )
        mhca_block.out_dim = qdim
        return mhca_block

    def forward(self, heterogen_pipelines: Sequence[HeterogeneousBatch], dataset: Tensor) -> Tensor:
        pipelines_embeddings = self.encode_pipelines(heterogen_pipelines)
        dataset_embeddings = self.dataset_encoder(dataset)
        pipelines_embeddings = self.mhca_block(pipelines_embeddings, dataset_embeddings)
        mhsa_output = self.mhsa_block(pipelines_embeddings)
        scores = self.linear(mhsa_output).squeeze(2)
        # [BATCH, N]
        return scores

    def training_step(
        self,
        batch: Tuple[Sequence[HeterogeneousBatch], Tensor],
        *args,
        **kwargs,
    ) -> Tensor:
        heterogen_pipelines, dataset, y = batch
        scores = self.forward(heterogen_pipelines, dataset)
        if self.loss_name == "kl_div":
            loss = F.kl_div(
                torch.log_softmax(scores, dim=1),
                torch.log_softmax(y, dim=1),
                log_target=True,
            )
        else:
            loss = self.loss_fn(scores, y)
        self.log("train_loss", loss)
        return loss

    def evaluation_step(
        self,
        prefix: str,
        batch: Tuple[Sequence[HeterogeneousBatch], Tensor],
    ):
        heterogen_pipelines, dataset, y = batch
        with torch.no_grad():
            scores = self.forward(heterogen_pipelines, dataset)

        if "ndcg" in self.validation_metrics:
            metric_fn = getattr(metrics_module, "ndcg")
            self.log(f"{prefix}_ndcg", metric_fn(y.cpu(), scores.cpu()))
        if "precision" in self.validation_metrics:
            metric_fn = getattr(metrics_module, "precision")
            self.log(f"{prefix}_precision", metric_fn(y, scores))
        if "kendalltau" in self.validation_metrics:
            metric_fn = getattr(metrics_module, "kendalltau")
            self.log(f"{prefix}_kendalltau", metric_fn(y.cpu(), scores.cpu()))
