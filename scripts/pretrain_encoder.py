import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from automl_surrogate.data import HeteroPipelineAndDatasetFeaturesDataset
from automl_surrogate.layers.encoders.simple_graph_encoder import SimpleGNNEncoder
from torch_geometric.data.database import SQLiteDatabase
from automl_surrogate.data.data_types import HeterogeneousData, HeterogeneousBatch
from torch_geometric.data import Data, Batch
from pytorch_lightning import LightningModule, Trainer
import torch.optim as optim
from torch_geometric.utils import to_dense_adj
from automl_surrogate.data.fedot_pipeline_features_extractor import FEDOTPipelineFeaturesExtractor2
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from automl_surrogate.models.node_embedder import NodeEmbedder
from automl_surrogate.models.embedding_joiner import CatEmbeddingJoiner
from automl_surrogate.models.hyperparams_embedder import HyperparametersEmbedder, PretrainedHyperparametersEmbedder
from automl_surrogate.models.name_embedder import NameEmbedder
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nested import nested_tensor

class PipelineDataset(Dataset):
    def __init__(self, task_pipe_comb_file: str):
        self.root = "/home/cherniak/itmo_job/GAMLET/data/meta_features_and_fedot_pipelines"
        task_pipe_comb = pd.read_json(os.path.join(self.root, task_pipe_comb_file), orient="table")
        task_pipe_comb = task_pipe_comb.reset_index(drop=False)
        task_pipe_comb = task_pipe_comb[task_pipe_comb.dataset_id != "mfeat-factors_5"]
        self.pipeline_ids = task_pipe_comb.pipeline_id.to_list()
        self.db = SQLiteDatabase("/home/cherniak/itmo_job/surrogate/pipelines.db", "pipelines")
        self.n_pipelines = len(self.pipeline_ids)
        self.pipeline_extractor = FEDOTPipelineFeaturesExtractor2("ordinal")

    def __len__(self):
        return self.n_pipelines

    def __getitem__(self, index) -> "HeterogeneousData":
        pipeline_id = self.pipeline_ids[index]
        graph = self.db.get(pipeline_id)
        return self.pipeline_extractor(graph)

    def collate_fn(self, batch: "Sequence[HeterogeneousData]") -> "HeterogeneousBatch":
        return HeterogeneousBatch.from_heterogeneous_data_list(batch)

class NodeDecoder(nn.Module):
    def __init__(
            self,
            in_dim: int,
            n_types: int,
            op_hparams_dim: int,
        ):
        super().__init__()
        self.type_classifier = nn.Linear(in_dim, n_types, bias=False)
        self.hparams_decoder = nn.Linear(in_dim, op_hparams_dim)

    def forward(self, x: "Tensor") -> "Tuple[Tensor, Tensor]":
        type_logits = self.type_classifier(x)
        hparams_emb = self.hparams_decoder(x)
        return type_logits, hparams_emb


class PipelineAE(LightningModule):
    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        name_emb = NameEmbedder(out_dim=2)
        hparams_emb = PretrainedHyperparametersEmbedder(
            autoencoder_ckpt_path="/home/cherniak/itmo_job/surrogate/experiment_logs/embed_hyperparameters/to_8_with_learnables/mse_loss__no_dropout/checkpoints/epoch=9-step=980.ckpt",
            out_dim=8,
            trainable=False,
        )
        joiner = CatEmbeddingJoiner(name_emb.out_dim, hparams_emb.out_dim)
        self.embedder = NodeEmbedder(
            op_hyperparams_embedder=hparams_emb,
            op_name_embedder=name_emb,
            emebedding_joiner=joiner,
        )
        self.encoder = tgnn.models.GCN(
            in_channels=self.embedder.out_dim,
            hidden_channels=64,
            num_layers=2,
        )
        self.encoder.out_dim = 64
        self.edge_decoder = tgnn.models.InnerProductDecoder()
        self.node_decoder = NodeDecoder(in_dim=self.encoder.out_dim, n_types=75+1, op_hparams_dim=hparams_emb.out_dim)


    def embed_input(self, batch: "HeterogeneousBatch") -> "Tensor":
        node_idxes_per_type = batch.node_idxes_per_type
        x_hparams = torch.empty(batch.num_nodes, self.embedder.op_hyperparams_embedder.out_dim).to(self.device)
        for node_type, idxes in node_idxes_per_type.items():
            unbind = batch.hparams.unbind()
            hparams = torch.vstack([unbind[i] for i in idxes])
            # Frozen hparams embedder
            with torch.no_grad():
                hparams_emb = self.embedder.op_hyperparams_embedder({node_type: hparams})[node_type]
            x_hparams[idxes] = hparams_emb
        x_type = self.embedder.op_name_embedder(batch.encoded_type)
        input_embeddings = torch.hstack([x_type, x_hparams])
        return input_embeddings

    @staticmethod
    def to_data_list(batch: "HeterogeneousBatch") -> "HeterogeneousData":
        res = []
        for i in range(len(batch.edge_split)-1):
            start, stop = batch.edge_split[i], batch.edge_split[i+1]
            edge_index = batch.edge_index[:, start: stop] - batch.nodes_before[i]
            nodes_idxes = torch.where(batch.batch == i)[0]
            node_embeddings = batch.node_embeddings[nodes_idxes]
            d = HeterogeneousData(
                edge_index=edge_index,
                num_nodes=len(nodes_idxes),
                encoded_type=torch.LongTensor([1]),  # Fake
                node_idxes_per_type={},
            )
            d.node_embeddings = node_embeddings
            res.append(d)
        return res

    def forward(self, batch: "HeterogeneousBatch") -> "Tuple[Tensor, Tensor, Tensor]":
        input_embeddings = self.embed_input(batch)
        batch.input_embeddings = input_embeddings

        node_embeddings = self.encoder(batch.input_embeddings, batch.edge_index)
        batch.node_embeddings = node_embeddings

        type_logits, hparams_emb = self.node_decoder(node_embeddings)
        type_loss = F.cross_entropy(type_logits, batch.encoded_type)
        hparams_loss = F.mse_loss(hparams_emb, batch.input_embeddings[:, 2:])  # `name_emb dim:`

        samples = self.to_data_list(batch)

        edge_losses = []
        for sample in samples:
            adj_logits = self.edge_decoder.forward_all(sample.node_embeddings, sigmoid=False)
            adj = to_dense_adj(sample.edge_index)
            edge_loss = F.binary_cross_entropy_with_logits(adj_logits, adj.squeeze(0))
            edge_losses.append(edge_loss)
        edge_loss = sum(edge_losses) / len(edge_losses)
        return type_loss, hparams_loss, edge_loss

    def training_step(self, batch: "HeterogeneousBatch", *args, **kwargs) -> "Tensor":
        type_loss, hparams_loss, edge_loss = self.forward(batch)
        loss = type_loss + hparams_loss + edge_loss
        self.log("train_type_loss", type_loss.detach().cpu().item())
        self.log("train_hparams_loss", hparams_loss.detach().cpu().item())
        self.log("train_edge_loss", edge_loss.detach().cpu().item())
        self.log("train_loss", loss.detach().cpu().item())
        return loss

    def validation_step(self, batch: "Batch", *args, **kwargs):
        type_loss, hparams_loss, edge_loss = self.forward(batch)
        loss = type_loss + hparams_loss + edge_loss
        self.log("val_type_loss", type_loss.cpu().item())
        self.log("val_hparams_loss", hparams_loss.cpu().item())
        self.log("val_edge_loss", edge_loss.cpu().item())
        self.log("val_loss", loss.cpu().item())

    def test_step(self, batch: "Batch", *args, **kwargs):
        type_loss, hparams_loss, edge_loss = self.forward(batch)
        loss = type_loss + hparams_loss + edge_loss
        self.log("test_type_loss", type_loss.cpu().item())
        self.log("test_hparams_loss", hparams_loss.cpu().item())
        self.log("test_edge_loss", edge_loss.cpu().item())
        self.log("test_loss", loss.cpu().item())

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def transfer_batch_to_device(self, batch: "Sequence", device: torch.device, dataloader_idx: int) -> "Sequence":
        def transfer_heterogeneous_batch(heterogen_batch: HeterogeneousBatch) -> HeterogeneousBatch:
            heterogen_batch.batch = heterogen_batch.batch.to(device)
            heterogen_batch.ptr = heterogen_batch.ptr.to(device)
            heterogen_batch.edge_index = heterogen_batch.edge_index.to(device)
            heterogen_batch.node_idxes_per_type = {k: v.to(device) for k, v in heterogen_batch.node_idxes_per_type.items()}
            heterogen_batch.hparams = nested_tensor([heterogen_batch.hparams[i] for i in range(heterogen_batch.hparams.size(0))], device=device)
            heterogen_batch.encoded_type = heterogen_batch.encoded_type.to(device)
            return heterogen_batch

        return transfer_heterogeneous_batch(batch)

if __name__ == "__main__":
    train_ds = PipelineDataset("train_task_pipe_comb.json")
    train_loader = DataLoader(train_ds, shuffle=True, num_workers=0, batch_size=256, collate_fn=train_ds.collate_fn)

    val_ds = PipelineDataset("test_task_pipe_comb.json")
    val_loader = DataLoader(val_ds, shuffle=False, num_workers=0, batch_size=256, collate_fn=val_ds.collate_fn)

    logger = TensorBoardLogger(save_dir="/home/cherniak/itmo_job/surrogate/experiment_logs/pretrain_encoder/frozen_hparams")

    model_checkpoint_callback = ModelCheckpoint(save_last=True, every_n_train_steps=1000)  # save_top_k=3, monitor="val_loss", mode="min", every_n_epochs=1,

    model = PipelineAE()

    trainer = Trainer(
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        max_epochs=100,
        accelerator="cuda",
        devices="auto",
        logger=logger,
        callbacks=[model_checkpoint_callback],
        gradient_clip_val=0.5,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)