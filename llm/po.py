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
from .dataset import LPDataset, EncodeDataset, PoDataset
from .lm_modeling import LP_model, lp_compute_metrics, load_model
from transformers import Trainer
from .lm_trainer import InnerTrainer
from peft import PeftConfig, PeftModel
from torch.utils.data import Dataset
from .pooling import MeanPooling, MaxPooling
from tqdm import tqdm
from .po_trainer import PoTrainer

from config import logger

Pooler = {
    'mean' : MeanPooling(),
    'max' : MaxPooling()
}


def po_lm(args, data, text):
    t = [b for a, b in text.items()]
    if args.plm_name == 'bert-base-uncased':
        tokenizer = AutoTokenizer.from_pretrained(args.plm_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    logger.info('Use model : {}'.format(args.plm_name))
    inputs = tokenizer(t,
                    truncation=True, 
                    add_special_tokens=True,
                    max_length=256,
                    padding='max_length',
                    return_tensors='pt',)
    p = osp.join(osp.join(args.input_dir, args.dataset), 'po_data.pt')
    po_edge_idx = torch.load(p)
    train_dataset = PoDataset(inputs['input_ids'], 
                        inputs['attention_mask'], 
                        po_edge_idx['po_edge'])

    new_model = load_model(args, ty='ref')

    training_args = TrainingArguments(
        per_device_train_batch_size=args.po_sm_batch_size,
        gradient_accumulation_steps=args.po_batch_size // args.po_sm_batch_size,
        output_dir=args.model_path,
        learning_rate=args.po_lr,
        per_device_eval_batch_size=args.po_batch_size,
        num_train_epochs=args.po_epoch,
        weight_decay=0.01,
        do_eval = False,
        remove_unused_columns=False,
        logging_dir='./train.log',
        logging_steps=args.logging_steps,
    )
    trainer = PoTrainer(
        model=new_model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=lp_compute_metrics
    )

    trainer.train()
    lm = new_model.model
    lm.save_pretrained(osp.join(args.model_path, 'save_model'))