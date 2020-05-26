from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN


from pygcn.utils import normalize
from pygcn.utils import  sparse_mx_to_torch_sparse_tensor



np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


###load data



adj, features, labels, idx_train, idx_val, idx_test = load_data()

model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)