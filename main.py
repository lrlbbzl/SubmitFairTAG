import torch
import numpy as np
import os
import json
from torch_geometric.loader import NeighborLoader, HGTLoader
import torch_geometric.transforms as T

from config import logger, args
from run import run

if __name__ == '__main__':

    run(args)
    # model = 
    # for subgraph in train_loader()
    # for e in range(args.epoch):
    #     for subgraph in train_loader:

