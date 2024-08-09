from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from gnn_model.model import GNN

import torch
from torch import nn
import torch.nn.functional as F
import sklearn.metrics as sm
from torchmetrics.classification import F1Score, Recall, Precision

class LPModel(pl.LightningModule):
    def __init__(self, args):
        super(LPModel, self).__init__()
        self.gnn = GNN(args.in_dim, args.out_dim, args.n_heads, args.n_layers, args.dropout, args.conv_name)
        self.lin_layer = nn.Linear(self.gnn.n_hid, 1)
        self.lr = args.lr
        self.f1 = F1Score(task='binary')
        self.recall = Recall(task='binary')
        self.precision = Precision(task='binary')


    def forward(self, x, edge_index):
        h = self.gnn(x, edge_index)
        h_src, h_dst = h[edge_index[0, :]], h[edge_index[1, :]]
        src_dst_mult = h_src * h_dst
        return self.lin_layer(src_dst_mult)
    
    def _step(self, batch: torch.Tensor, phase: str='train') -> torch.Tensor: 
        yhat_edge = self(batch.x, batch.edge_label_index).squeeze() 
        y = batch.edge_label 
        loss = F.binary_cross_entropy_with_logits(input=yhat_edge, target=y) 
        f1 = self.f1(preds=yhat_edge, target=y) 
        prec = self.precision(preds=yhat_edge, target=y) 
        recall = self.recall(preds=yhat_edge, target=y) 
 
        # Watch for logging here - we need to provide batch_size, as (at the time of this implementation) 
        # PL cannot understand the batch size. 
        self.log(f"{phase}_f1", f1, batch_size=batch.edge_label_index.shape[1]) 
        self.log(f"{phase}_loss", loss, batch_size=batch.edge_label_index.shape[1]) 
        self.log(f"{phase}_precision", prec, batch_size=batch.edge_label_index.shape[1]) 
        self.log(f"{phase}_recall", recall, batch_size=batch.edge_label_index.shape[1]) 

        return loss 

    def training_step(self, batch, batch_idx): 
        return self._step(batch) 
 
    def validation_step(self, batch, batch_idx): 
        return self._step(batch, "val") 
 
    def test_step(self, batch, batch_idx): 
        return self._step(batch, "test") 
 
    def predict_step(self, batch): 
        x, edge_index = batch 
        return self(x, edge_index) 
 
    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(), lr=self.lr)