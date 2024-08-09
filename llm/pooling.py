import torch
from torch import nn

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        mask_expand = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeds = torch.sum(last_hidden_state * mask_expand, dim=1)
        sum_mask = torch.sum(mask_expand, dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeds = sum_embeds / sum_mask
        return mean_embeds
    
class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        mask_expand = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_embeds = last_hidden_state * mask_expand
        max_embeds = torch.max(masked_embeds, dim=1)[0]
        return max_embeds