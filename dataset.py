import json
import torch
import os.path as osp

def get_dataset(args):
    if args.dataset == 'cora':
        p = osp.join(args.input_dir, args.dataset)
        g, text = torch.load(osp.join(p, 'g.pt')), json.load(open(osp.join(p, 'text.json'), 'r'))
        return g, text

    elif args.dataset == 'pubmed':
        p = osp.join(args.input_dir, args.dataset)
        g, text = torch.load(osp.join(p, 'g.pt')), json.load(open(osp.join(p, 'text.json'), 'r'))
        return g, text
    
    elif args.dataset == 'citeseer':
        p = osp.join(args.input_dir, args.dataset)
        g = torch.load(osp.join(p, 'g.pt'))
        text = {i : t for i, t in enumerate(g.raw_texts)}
        return g, text