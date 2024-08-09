import torch
import numpy as np

x = torch.load('./data/cora/filter_data1_beta0.3.pt')
y = torch.load('results.pt')


mp = {}
st = set()

for i in range(len(y['scores'])):
    score, label, is_heter, node_idx = y['scores'][i], y['labels'][i], y['is_heter'][i], y['node_idx'][i]
    a, b = node_idx.numpy().tolist()
    if a not in mp:
        mp.update({a : dict()})
        mp[a].update({1 : dict()})
        mp[a].update({0 : dict()})
    if is_heter and label == 1:
        mp[a][1].update({b : score.cpu().numpy().tolist()})
    if not is_heter and label == 0:
        mp[a][0].update({b : score.cpu().numpy().tolist()})

new_mp = {}
for k, v in mp.items():
    if len(v[0]) > 0 and len(v[1]) > 0:
        new_mp.update({k : v})

ans = []
for k, v in new_mp.items():
    heter_e, homo_e = v[1], v[0]
    heter = sorted(heter_e.items(), key=lambda x : x[1])
    node1 = heter[0][0]
    homo = sorted(homo_e.items(), key=lambda x : x[1], reverse=True)
    node2 = homo[0][0]
    ans.append([k, node1, node2])


res = {'po_edge' : torch.from_numpy(np.array(ans)).transpose(0, 1)} # (3, N)
torch.save(res, 'po_data.pt')
# _mp = {}
# for i in range(len(y['scores'])):
#     score, label, is_heter, node_idx = y['scores'][i], y['labels'][i], y['is_heter'][i], y['node_idx'][i]
#     b, a = node_idx.numpy().tolist()
#     if a not in _mp:
#         _mp.update({a : dict()})
#         _mp[a].update({1 : dict()})
#         _mp[a].update({0 : dict()})
#     if is_heter and label == 1:
#         _mp[a][1].update({b : score})
#     if not is_heter and label == 0:
#         _mp[a][0].update({b : score})

# _new_mp = {}
# for k, v in _mp.items():
#     if len(v[0]) > 0 and len(v[1]) > 0:
#         _new_mp.update({k : v})

# node_idx = y['node_idx'].numpy().tolist()
# now = torch.cat([x['pos_edge'], x['neg_edge']], dim=-1).transpose(0, 1).numpy().tolist()

# node_idx = [tuple(x) for x in node_idx]
# now = [tuple(x) for x in now]

# b = [(x in node_idx) for x in now]

# st = list(set(node_idx) - set(now))
# print(len(st))

# l = len(now)
# select = np.random.choice(range(len(st)), l, replace=False)
# select_edge = [st[i] for i in select]

# select_edge = torch.tensor(select_edge)
# select_edge = select_edge.transpose(0, 1)
# x.update({'oracle_edge' : select_edge})

# # import pdb; pdb.set_trace()
# torch.save(x, 'filter_data1_beta0.3.pt')