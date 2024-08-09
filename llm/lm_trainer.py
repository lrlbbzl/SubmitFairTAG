import gc
import logging
import os
import os.path as osp

import evaluate
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import KLDivLoss
import torch.nn.functional as F
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import subgraph
from torchmetrics.functional import retrieval_reciprocal_rank as mrr
from transformers import EarlyStoppingCallback
from transformers import Trainer as HugTrainer
from transformers import TrainingArguments
from .lm_modeling import load_model

from config import args


if args.add_kl:
    oracle_model = load_model(args, ty='oracle')

class InnerTrainer(HugTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensitive_weight = 2.0
        self.non_sensitive_weight = 1.0
        self.gamma = 2.0
        self.compute_steps = 0.0

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        input_ids, attention_mask = inputs.pop("input_ids"), inputs.pop("attention_mask")
        if 'is_heterogeneous' in inputs.keys():
            is_heter = inputs.pop('is_heterogeneous')
        pred = model(input_ids, attention_mask)
        pred = pred.squeeze(-1)
        loss = F.binary_cross_entropy(pred, labels,)

        if args.add_kl:
            with torch.no_grad():
                oracle_input_ids, oracle_attention_mask = inputs.pop('oracle_input_ids'), inputs.pop('oracle_attention_mask')
                retain_outputs = oracle_model(oracle_input_ids, oracle_attention_mask)
                retain_logits = torch.cat([1 - retain_outputs, retain_outputs], dim=-1)
            retain_pred = model(oracle_input_ids, oracle_attention_mask)
            pred_logits = torch.cat([1 - retain_pred, retain_pred], dim=-1)
            log_retain_pred = F.log_softmax(pred_logits, dim=-1)
            retain_loss = F.kl_div(log_retain_pred, retain_logits, reduction='batchmean', )
            loss += retain_loss
        if return_outputs:
            pred = {'pred' : pred}
        return (loss, pred) if return_outputs else loss