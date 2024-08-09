import gc
import logging
import os
import os.path as osp
import json

import evaluate
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import subgraph, negative_sampling
from torchmetrics.functional import retrieval_reciprocal_rank as mrr
from transformers import EarlyStoppingCallback
from transformers import Trainer as HugTrainer
from transformers import TrainingArguments
from transformers import AutoModel, AutoTokenizer

from peft import PeftModel



lm1 = AutoModel.from_pretrained('/root/autodl-tmp/models/AI-ModelScope/bert-base-uncased')
model1 = PeftModel.from_pretrained(lm1, '/root/autodl-tmp/FairLLM4Graph/checkpoints/cora/bert-base-uncased_filter_data1_beta0.3/save_model')


lm2 = AutoModel.from_pretrained('/root/autodl-tmp/models/AI-ModelScope/bert-base-uncased')
model2 = PeftModel.from_pretrained(lm2, '/root/autodl-tmp/FairLLM4Graph/checkpoints/cora/bert-base-uncased_po/save_model')


k1, k2 = {}, {}
for name, param in model1.named_parameters():
    # if 'lora' in name.lower():
    k1.update({name : param})
for name, param in model2.named_parameters():
    # if 'lora' in name.lower():
    k2.update({name : param})

p = []
for k, v in k1.items():
    if False in v == k2[k]:
        print(k)

import pdb; pdb.set_trace()
