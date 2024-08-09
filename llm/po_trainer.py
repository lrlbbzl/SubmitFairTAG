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
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from .lm_modeling import load_model

from config import args

if args.mode == 'po':
    ref_model = load_model(args, ty='ref')

class PoTrainer(HugTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensitive_weight = 2.0
        self.non_sensitive_weight = 1.0
        self.gamma = 2.0
        self.compute_steps = 0.0

    # def compute_loss(self, model, inputs, return_outputs=False):
        
    #     pre_input_ids, pre_attention_mask = inputs.pop('pre_input_ids'), inputs.pop('pre_attention_mask')
    #     aft_input_ids, aft_attention_mask = inputs.pop('aft_input_ids'), inputs.pop('aft_attention_mask')

    #     win_pred = model(pre_input_ids, pre_attention_mask)
    #     lose_pred = model(aft_input_ids, aft_attention_mask)


    #     with torch.no_grad():
    #         win_ref = ref_model(pre_input_ids, pre_attention_mask)
    #         lose_ref = ref_model(aft_input_ids, aft_attention_mask)

    #     # # import pdb; pdb.set_trace()
    #     # use loss instead of logits
    #     labels_heter = torch.tensor([1] * pre_input_ids.size(0), dtype=torch.float)
    #     labels_homo = torch.tensor([0] * aft_input_ids.size(0), dtype=torch.float)
    #     fc = nn.BCEWithLogitsLoss(reduction='none')

    #     # win_pred_loss = -1 * fc(win_pred, labels_heter.unsqueeze(-1).cuda(),)
    #     # lose_pred_loss = -1 * fc(lose_pred.squeeze(-1), labels_homo.cuda(),)
    #     # win_ref_loss = -1 * fc(win_ref.squeeze(-1), labels_heter.cuda(),)
    #     # lose_ref_loss = -1 * fc(lose_ref.squeeze(-1), labels_homo.cuda(),)

    #     # theta_logratio = win_pred.log() - lose_pred.log()
    #     # ref_logratio = win_ref.log() - lose_ref.log()

    #     theta_logratio = win_pred - lose_pred
    #     ref_logratio = win_ref - lose_ref

    #     loss = -F.logsigmoid(args.po_beta * (theta_logratio - ref_logratio)).mean()
    #     _lambda = 10
    #     # stimulate larger win
    #     loss += _lambda * torch.max(torch.zeros_like(win_ref), win_ref - win_pred).mean()
    #     # depress less lose
    #     # loss -= _lambda * torch.min(torch.zeros_like(win_ref), lose_pred.log() - lose_ref.log()).mean()

    #     if return_outputs:
    #         pred = {'pred' : win_pred.squeeze(-1) }
    #     return (loss, pred) if return_outputs else loss


    def compute_loss(self, model, inputs, return_outputs=False):
        
        pre_input_ids, pre_attention_mask = inputs.pop('pre_input_ids'), inputs.pop('pre_attention_mask')
        aft_input_ids, aft_attention_mask = inputs.pop('aft_input_ids'), inputs.pop('aft_attention_mask')

        ty = inputs.pop('types')
        pos_idx, neg_idx = torch.nonzero(ty == 0).squeeze(1), torch.nonzero(ty == 1).squeeze(1)
        _lambda = 10

        if pos_idx.size(0) > 0:
            pos_heter_input_ids, pos_heter_attention_mask = pre_input_ids[pos_idx], pre_attention_mask[pos_idx]
            pos_homo_input_ids, pos_homo_attention_mask = aft_input_ids[pos_idx], aft_attention_mask[pos_idx]
            with torch.no_grad():
                win_ref = ref_model(pos_heter_input_ids, pos_heter_attention_mask)
                lose_ref = ref_model(pos_homo_input_ids, pos_homo_attention_mask)
            win_pred = model(pos_heter_input_ids, pos_heter_attention_mask)
            lose_pred = model(pos_homo_input_ids, pos_homo_attention_mask)

            theta_logratio = win_pred - lose_pred
            ref_logratio = win_ref - lose_ref

            loss = -F.logsigmoid(args.po_beta * (theta_logratio - ref_logratio)).mean()
            # stimulate larger win
            loss += _lambda * torch.max(torch.zeros_like(win_ref), win_ref - win_pred).mean()
        # depress less lose
        # loss -= _lambda * torch.min(torch.zeros_like(win_ref), lose_pred.log() - lose_ref.log()).mean()

        # part 2, lower score is a win
        if neg_idx.size(0) > 0:
            neg_homo_input_ids, neg_homo_attention_mask = pre_input_ids[neg_idx], pre_attention_mask[neg_idx]
            neg_heter_input_ids, neg_heter_attention_mask = aft_input_ids[neg_idx], aft_attention_mask[neg_idx]
            with torch.no_grad():
                win_ref = ref_model(neg_homo_input_ids, neg_homo_attention_mask)
                lose_ref = ref_model(neg_heter_input_ids, neg_heter_attention_mask)
            win_pred = model(neg_homo_input_ids, neg_homo_attention_mask)
            lose_pred = model(neg_heter_input_ids, neg_heter_attention_mask)
            theta_logratio = win_pred - lose_pred
            ref_logratio = win_ref - lose_ref
            if pos_idx.size(0) > 0:
                loss += F.sigmoid(args.po_beta * (theta_logratio - ref_logratio)).mean()
            else:
                loss = F.sigmoid(args.po_beta * (theta_logratio - ref_logratio)).mean()
            loss += _lambda * torch.max(torch.zeros_like(win_ref), win_pred - win_ref).mean()


        if return_outputs:
            pred = {'pred' : win_pred.squeeze(-1) }
        return (loss, pred) if return_outputs else loss