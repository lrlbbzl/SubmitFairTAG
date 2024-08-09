import torch
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)
import numpy as np
from tqdm import tqdm
from itertools import combinations_with_replacement
from torch import nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from config import logger
from llm.dataset import LPDataset
from torch.utils.data import DataLoader, TensorDataset
from llm.lm_modeling import load_model

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, sensitive_weight=1.0, non_sensitive_weight=1.0):
        super(CustomCrossEntropyLoss, self).__init__()
        self.sensitive_weight = sensitive_weight
        self.non_sensitive_weight = non_sensitive_weight

    def forward(self, input, target, sensitive_attr):
        """
        input: (bs,)
        target: (bs,)
        """
        log_softmax = F.log_softmax(input, dim=1)
        target_one_hot = F.one_hot(target, num_classes=input.shape[1])
        loss = torch.sum(-1 * target_one_hot * log_softmax, dim=1)

        # 根据敏感属性标签设置权重
        weights = torch.where(sensitive_attr == 1, torch.tensor(self.sensitive_weight), torch.tensor(self.non_sensitive_weight))
        loss = loss * weights.float()
        
        return torch.mean(loss)

def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float)
    link_labels[: pos_edge_index.size(1)] = 1.0
    return link_labels


def fair_metrics(gt, y, group):
    metrics_dict = {
        "DPd": demographic_parity_difference(gt, y, sensitive_features=group),
        "EOd": equalized_odds_difference(gt, y, sensitive_features=group),
    }
    return metrics_dict

def prediction_fairness(test_edge_idx, test_edge_labels, te_y, group):
    te_dyadic_src = group[test_edge_idx[0]]
    te_dyadic_dst = group[test_edge_idx[1]]

    # SUBGROUP DYADIC
    u = list(combinations_with_replacement(np.unique(group), r=2))

    te_sub_diatic = []
    for i, j in zip(te_dyadic_src, te_dyadic_dst):
        for k, v in enumerate(u):
            if (i, j) == v or (j, i) == v:
                te_sub_diatic.append(k)
                break
    te_sub_diatic = np.asarray(te_sub_diatic)
    # MIXED DYADIC 
    
    te_mixed_dyadic = te_dyadic_src != te_dyadic_dst
    # GROUP DYADIC
    te_gd_dict = fair_metrics(
        np.concatenate([test_edge_labels, test_edge_labels], axis=0),
        np.concatenate([te_y, te_y], axis=0),
        np.concatenate([te_dyadic_src, te_dyadic_dst], axis=0),
    )

    te_md_dict = fair_metrics(test_edge_labels, te_y, te_mixed_dyadic)

    te_sd_dict = fair_metrics(test_edge_labels, te_y, te_sub_diatic)

    fair_list = [
        te_md_dict["DPd"],
        te_md_dict["EOd"],
        te_gd_dict["DPd"],
        te_gd_dict["EOd"],
        te_sd_dict["DPd"],
        te_sd_dict["EOd"],
    ]

    return fair_list



def generate_results(args, text):
    t = [b for a, b in text.items()]
    if args.plm_name == 'bert-base-uncased':
        tokenizer = AutoTokenizer.from_pretrained(args.plm_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    inputs = tokenizer(t,
                    truncation=True, 
                    add_special_tokens=True,
                    max_length=256,
                    padding='max_length',
                    return_tensors='pt',)
    pre_data = torch.load('results.pt')
    true_l, false_l = torch.nonzero(pre_data['labels'] == 1).squeeze(), torch.nonzero(pre_data['labels'] == 0).squeeze()
    pos_edge, neg_edge = pre_data['node_idx'][true_l].transpose(0, 1), pre_data['node_idx'][false_l].transpose(0, 1)
    is_heter = torch.cat([pre_data['is_heter'][true_l], pre_data['is_heter'][false_l]])
    train_dataset = LPDataset(inputs['input_ids'],
                            inputs['attention_mask'],
                            pos_edge,
                            neg_edge)
    train_dataloader = DataLoader(train_dataset, batch_size=args.infer_batch_size, shuffle=False, num_workers=4)
    scores, labels, node_idx = [], [], []
    model = load_model(args, ty='ref')

    with torch.no_grad():
        for step, data in tqdm(enumerate(train_dataloader)):
            score = model(data['input_ids'].cuda(), data['attention_mask'].cuda())
            scores.append(score.squeeze(-1))
            labels.append(data['label'])
            node_idx.append(data['node_idx'])
        scores = torch.cat(scores, dim=0)
        labels = torch.cat(labels, dim=0)
        node_idx = torch.cat(node_idx, dim=0)
        mp = {
            'scores' : scores,
            'labels' : labels,
            'is_heter' : is_heter,
            'node_idx' : node_idx
        }
        torch.save(mp, 'results_ref.pt')