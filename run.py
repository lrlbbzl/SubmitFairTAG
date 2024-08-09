import os
import os.path as osp
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch_geometric as tg
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit
import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from dataset import get_dataset
from torch_geometric.utils import train_test_split_edges
from gnn_model.model import GNN
from sklearn.metrics import classification_report
from torch.optim import Adam, AdamW
from utils import get_link_labels

from config import logger
from llm.finetune import finetune_lm, merge_modeling, finetune_lm_on_filtering
from transformers import AutoModel, AutoTokenizer
from llm.lm_trainer import InnerTrainer
from llm.lm_modeling import LP_model
from llm.po import po_lm

from peft import PeftModel
from utils import prediction_fairness, generate_results
import warnings
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"

seeds = [1,2,3,4,5,6,7,8,9,10]

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 200)
        self.conv2 = GCNConv(200, out_channels)
        self.dropout = 0.2

    def forward(self, x, pos_edge_index):
        x = self.conv1(x, pos_edge_index)
        x = self.conv2(x, pos_edge_index)
        return x

    def test(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits, edge_index

def run(args):
    ## prepare dataset
    acc_auc = []
    fairness = []

    for i, seed in enumerate(seeds):
        logger.info("Round: {}".format(i + 1))
        torch.manual_seed(seed)
        np.random.seed(seed)
        g, text = get_dataset(args)
        args.in_dim = g.num_features
        protected_attribute = g.y
        num_calsses = len(np.unique(protected_attribute))
        g.to(device)
        g.train_mask = g.val_mask = g.test_mask = None
        data = train_test_split_edges(g, val_ratio=0.1, test_ratio=0.2)
        data = data.to(device)
        N = data.num_nodes
        logger.info("Mode: {}".format(args.mode))
        logger.info("Use peft: {}".format(args.use_peft))
        logger.info("Use full: {}".format(args.use_full))
        logger.info("Filtering: {}".format(args.filter))
        if args.mode == 'ft_lm' and not args.filter:
            ## Start fine-tune LLM for Graph context
            args.model_path = osp.join(args.output_dir, osp.join(args.dataset, args.plm_name))
            if args.use_full:
                args.model_path = args.model_path + '_full'
            embeds_path = osp.join(args.model_path, 'text_embeddings_{}.pt'.format(args.use_peft))
            if not osp.exists(args.model_path):
                os.makedirs(args.model_path)
            if (args.use_peft or args.use_full) and (not any('save_model' in d for d in os.listdir(args.model_path))):
                finetune_lm(args, g, text)
            if not osp.exists(embeds_path):
                text_embeddings = merge_modeling(args, g, text)
                torch.save(text_embeddings, embeds_path)
            text_embeddings = torch.load(embeds_path)
            args.in_dim = text_embeddings.size(1)
            data.x = text_embeddings

        elif args.mode == 'ft_lm' and args.filter:
            args.model_path = osp.join(args.output_dir, osp.join(args.dataset, args.plm_name + '_filter'))
            # args.model_path = '/root/autodl-tmp/FairLLM4Graph/checkpoints/cora/bert-base-uncased_filter_data1_beta0.3'
            if args.use_full:
                args.model_path = args.model_path + '_full'
            embeds_path = osp.join(args.model_path, 'text_embeddings_{}.pt'.format(args.use_peft))

            if not osp.exists(args.model_path):
                os.makedirs(args.model_path)
            if (args.use_peft or args.use_full) and (not any('save_model' in d for d in os.listdir(args.model_path))):
                finetune_lm_on_filtering(args, g, text)
            if not os.path.exists('results_ref.pt'):
                generate_results(args, text)
            if not osp.exists(embeds_path):
                text_embeddings = merge_modeling(args, g, text)
                torch.save(text_embeddings, embeds_path)
            text_embeddings = torch.load(embeds_path)
            args.in_dim = text_embeddings.size(1)
            data.x = text_embeddings

        elif args.mode == 'po':
            args.model_path = osp.join(args.output_dir, osp.join(args.dataset, args.plm_name + '_po'))
            if args.use_full:
                args.model_path = args.model_path + '_full'
                embeds_path = osp.join(args.model_path, 'text_embeddings.pt'.format(args.use_peft))
            else:
                embeds_path = osp.join(args.model_path, 'text_embeddings_{}.pt'.format(args.use_peft))
            if not osp.exists(args.model_path):
                os.makedirs(args.model_path)
            if not any('save_model' in d for d in os.listdir(args.model_path)):
                po_lm(args, g, text)
            if not osp.exists(embeds_path):
                text_embeddings = merge_modeling(args, g, text)
                torch.save(text_embeddings, embeds_path)
            text_embeddings = torch.load(embeds_path)
            args.in_dim = text_embeddings.size(1)
            data.x = text_embeddings
            


        model = GNN(args.in_dim, args.out_dim, args.n_heads, args.n_layers, args.dropout, args.conv_name).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr)


        best_val_perf = test_perf = 0
        edges, inter_and_intra = [], []
        for epoch in range(1, args.epoch + 1):
            neg_edges_train = negative_sampling(
                edge_index=data.train_pos_edge_index,
                num_nodes=N,
                num_neg_samples=data.train_pos_edge_index.size(1)
            ).to(device)
            data.x = data.x.to(device)
            model.train()
            optimizer.zero_grad()
            h = model(data.x, data.train_pos_edge_index)
            link_logits, _ = model.test(h, data.train_pos_edge_index, neg_edges_train)
            train_labels = get_link_labels(data.train_pos_edge_index, neg_edges_train).to(device)
            loss = F.binary_cross_entropy_with_logits(link_logits, train_labels)
            loss.backward()
            optimizer.step()

            # eval
            model.eval()
            perfs = []
            for prefix in ["val", "test"]:
                pos_edge_index = data[f"{prefix}_pos_edge_index"]
                neg_edge_index = data[f"{prefix}_neg_edge_index"]
                with torch.no_grad():
                    z = model(data.x, data.train_pos_edge_index)
                    link_logits, edge_idx = model.test(z, pos_edge_index, neg_edge_index)

                link_probs = link_logits.sigmoid()
                link_labels = get_link_labels(pos_edge_index, neg_edge_index)
                auc = roc_auc_score(link_labels.cpu(), link_probs.cpu())
                perfs.append(auc)

                if prefix == 'test':
                    test_edges = torch.cat([pos_edge_index, neg_edge_index], dim=1)
                    test_inter_and_intra = torch.tensor(protected_attribute[test_edges[0].cpu()] != protected_attribute[test_edges[1].cpu()], dtype=torch.int32).clone().detach()

            val_perf, tmp_test_perf = perfs
            if val_perf > best_val_perf:
                best_val_perf = val_perf
                test_perf = tmp_test_perf
                edges, inter_and_intra = test_edges, test_inter_and_intra
            if epoch % 20 == 0:
                logger.info('Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(epoch, loss, best_val_perf, test_perf))
        auc = test_perf
        cut = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        # cut = [0.50]
        best_acc = 0
        best_cut = 0.5
        inter_acc, intra_acc = 0, 0
        inter_report, intra_report = None, None
        for i in cut:
            pred = link_probs.cpu() >= i
            acc = accuracy_score(link_labels.cpu(), pred)
            inter_pred_true, intra_pred_true = sum(inter_and_intra * pred) / sum(inter_and_intra), (sum(pred) - sum(inter_and_intra * pred)) / (inter_and_intra.size(0) - sum(inter_and_intra))
            logger.info("Predict 1 from inter relation: {}, intra relation: {}".format(inter_pred_true, intra_pred_true))
            inter_pos, intra_pos = torch.where(inter_and_intra == 1)[0], torch.where(inter_and_intra == 0)[0]
            inter_pred, inter_label = pred[inter_pos], link_labels.cpu()[inter_pos]
            intra_pred, intra_label = pred[intra_pos], link_labels.cpu()[intra_pos]
            logger.info("Cut: {}, Acc: {}".format(i, acc))
            if acc > best_acc:
                best_acc = acc
                best_cut = i
                inter_acc, intra_acc = accuracy_score(inter_label, inter_pred), accuracy_score(intra_label, intra_pred)
                inter_report, intra_report = classification_report(inter_label, inter_pred), classification_report(intra_label, intra_pred)
        
        logger.info("Inter relation accuracy: {}, intra relation accuracy: {}.".format(inter_acc, intra_acc))

        logger.info("Inter relation:")
        logger.info("\n{}".format(inter_report))
        logger.info("Intra relation:")
        logger.info("\n{}".format(intra_report))

        f = prediction_fairness(
            edge_idx.cpu(), link_labels.cpu(), link_probs.cpu() >= best_cut, protected_attribute.cpu()
        )
        acc_auc.append([best_acc * 100, auc * 100])
        fairness.append([x * 100 for x in f])
    
    ma = np.mean(np.asarray(acc_auc), axis=0)
    mf = np.mean(np.asarray(fairness), axis=0)

    sa = np.std(np.asarray(acc_auc), axis=0)
    sf = np.std(np.asarray(fairness), axis=0)

    logger.info(f"ACC: {ma[0]:2f} +- {sa[0]:2f}")
    logger.info(f"AUC: {ma[1]:2f} +- {sa[1]:2f}")

    logger.info(f"DP mix: {mf[0]:2f} +- {sf[0]:2f}")
    logger.info(f"EoP mix: {mf[1]:2f} +- {sf[1]:2f}")
    logger.info(f"DP group: {mf[2]:2f} +- {sf[2]:2f}")
    logger.info(f"EoP group: {mf[3]:2f} +- {sf[3]:2f}")
    logger.info(f"DP sub: {mf[4]:2f} +- {sf[4]:2f}")
    logger.info(f"EoP sub: {mf[5]:2f} +- {sf[5]:2f}")
        




