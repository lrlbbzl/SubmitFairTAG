import json
import numpy as np
import torch
from torch_geometric.data import Data
source = '/root/autodl-tmp/FairLLM4Graph/data/pubmed/data/Pubmed-Diabetes.DIRECTED.cites.tab'
la = '/root/autodl-tmp/FairLLM4Graph/data/pubmed/data/Pubmed-Diabetes.NODE.paper.tab'
mp = dict()
edges = []
lines = open(source).readlines()[2:]
for line in lines:
    line = line.strip().split('\t')
    a, b = line[1], line[3]
    a = a[a.find('paper:') + len('paper:'):]
    b = b[b.find('paper:') + len('paper:'):]
    if a not in mp:
        mp.update({a : len(mp)})
    if b not in mp:
        mp.update({b : len(mp)})
    edges.append([mp[a], mp[b]])
x = json.load(open('./pubmed.json', 'r'))
t = dict()
for d in x:
    if 'PMID' not in d:
        t_id = '17874530'
        t.update({t_id : 'Title: -- Abstract: --'})
        continue
    t.update({d['PMID'] : 'Title: {} Abstract: {}'.format(d['TI'], d['AB'])})

las = {}
lines = open(la, 'r').readlines()[2:]
for line in lines:
    line = line.strip().split('\t')
    a, b = line[0], line[1]
    b = int(b[b.find('label=') + len('label=') : ])
    las.update({mp[a] : b})
las = np.array([las[i] for i in range(len(las))])

t = {mp[a] : b for a, b in t.items()}
json.dump(t, open('text.json', 'w'))
json.dump(mp, open('mp.json', 'w'))
edges = np.array(edges)
edges = torch.from_numpy(edges).transpose(0, 1)


n_nodes, n_features = 19717, 500
data_X = np.zeros((n_nodes, n_features), dtype='float32')
feature_to_index = {}
## geenrate feature
with open('./data/Pubmed-Diabetes.NODE.paper.tab', 'r') as node_file:
    # first two lines are headers
    node_file.readline()
    node_file.readline()

    k = 0

    for i, line in enumerate(node_file.readlines()):
        items = line.strip().split('\t')

        paper_id = items[0]

        # f1=val1 \t f2=val2 \t ... \t fn=valn summary=...
        features = items[2:-1]
        for feature in features:
            parts = feature.split('=')
            fname = parts[0]
            fvalue = float(parts[1])

            if fname not in feature_to_index:
                feature_to_index[fname] = k
                k += 1

            data_X[mp[paper_id], feature_to_index[fname]] = fvalue

data_X = torch.from_numpy(data_X)
print(data_X.dtype)
g = Data(x=data_X, edge_index=edges)
g['y'] = torch.from_numpy(las).data

torch.save(g, 'g.pt')