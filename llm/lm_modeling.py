from config import logger

import logging
import os.path as osp

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_config, get_peft_model
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers import logging as transformers_logging
from .pooling import MaxPooling, MeanPooling
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from typing import Optional, Dict, Callable

from transformers.trainer import EvalPrediction
from transformers import AutoTokenizer, AutoModel

hidden_size_map = {
    'bert-base-uncased' : 768,
    'Llama-2-7b-ms' : 4096
}

def lp_compute_metrics(preds: Optional[Callable[[EvalPrediction], Dict]]):
    pred, label = preds.predictions, preds.label_ids
    pred = pred.reshape(pred.shape[0])
    label = label.reshape(label.shape[0])
    pred, label = torch.from_numpy(pred), torch.from_numpy(label)
    pred = (pred >= 0.5).long()
    precision, recall, f1, _ = precision_recall_fscore_support(label, pred, average='binary')
    acc = accuracy_score(label, pred)
    return {
        'acc': acc, 'f1' : f1
    }

def load_model(args, ty):
    assert ty in ['oracle', 'ref']
    p = {
        'oracle' : args.oracle_model_path,
        'ref' : args.ref_model_path
    } 
    path = p[ty]

    if args.plm_name == 'bert-base-uncased':
        lm = AutoModel.from_pretrained(args.plm_path)
        tokenizer = AutoTokenizer.from_pretrained(args.plm_path)
    else:
        lm = AutoModel.from_pretrained(args.plm_path, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(args.plm_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

    lp_model = LP_model(args)
    if args.use_peft:
        peft_model = PeftModel.from_pretrained(lm, path)
        if ty == 'oracle':
            lp_model.model = peft_model.model # BertModel
        elif ty == 'ref':
            lp_model.model.load_adapter(path, adapter_name='default')
    elif args.use_full:
        lp_model.model = AutoModel.from_pretrained(path)
    else:
        lp_model.model = lm
    
    lp_model = lp_model.to('cuda')
    return lp_model


class LinkPredHead(nn.Module):
    def __init__(self, hidden_size, header_dropout_prob):
        super(LinkPredHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        dropout = header_dropout_prob if header_dropout_prob is not None else header_dropout_prob
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, x_i, x_j):
        x = x_i * x_j
        x = self.dense(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return torch.sigmoid(x)


class LP_model(nn.Module):
    def __init__(self, args):
        super(LP_model, self).__init__()
        self.plm_path = args.plm_path
        if args.pooling == 'mean':
            self.pooling = MeanPooling()
        elif args.pooling == 'max':
            self.pooling = MaxPooling()
        assert osp.exists(self.plm_path)
        logger.info("Load model from {}".format(self.plm_path))
        
        if args.plm_name == 'bert-base-uncased':
            self.model = AutoModel.from_pretrained(args.plm_path)
        else:
            self.model = AutoModel.from_pretrained(args.plm_path,
                                                torch_dtype=torch.bfloat16,
                                                device_map='auto')

        if args.mode in ['ft_lm', 'po']:
            if args.use_peft:
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    inference_mode=False,
                    r=args.peft_r,
                    lora_alpha=args.peft_lora_alpha,
                    lora_dropout=args.peft_lora_dropout
                )
                self.model = PeftModel(self.model, lora_config)
                self.model.print_trainable_parameters()
            ### else full, use total model parameters


        lp_config = {
            'hidden_size' : hidden_size_map[args.plm_name],
            'header_dropout_prob' : 0.2
        }
        self.lp_head = LinkPredHead(lp_config['hidden_size'], lp_config['header_dropout_prob'])
        self.nm = 0


    def forward(self, input_ids, attention_mask, labels=None):
        """
        input_ids, attention_mask:
            (bs, 2, seq_len)
        """
        bs, num_samples, input_size = input_ids.shape
        input_ids, attention_mask = input_ids.view(-1, input_size), attention_mask.view(-1, input_size)
        output = self.model(input_ids, attention_mask)
        output = self.pooling(output.last_hidden_state, attention_mask)
        output = output.view(bs, num_samples, -1)
        hidden_size = output.shape[-1]
        pred = self.lp_head(output[:, 0, :], output[:, 1, :])
        return pred