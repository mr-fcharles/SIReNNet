# %%
import warnings
from numba.core.errors import NumbaDeprecationWarning

warnings.filterwarnings('ignore', category=NumbaDeprecationWarning)

# %%
from SIReNet.epidemics_graph import EpidemicsGraph

prova = EpidemicsGraph(pop_size=10000)
prova.sampler_initializer()

# %%

prova.create_families()

# %%

prova.adjacency


# %%

prova.add_links(get_richer_step=1)

# %%
prova.build_nx_graph()
prova.degree_distribution()

# %%

prova.compute_common_neighbors()

# %%

prova.start_infection(contagion_probability=0.05)

# %%

prova.initialize_individual_factors()

# %%

prova.infect_over_time

# %%
for i in range(10):
    prova.propagate_infection(mu=2)

#prova.propagation_stats()


# %%

prova.export_graph()


# %%

# Graph neural networks
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN


from pygcn.utils import normalize
from pygcn.utils import  sparse_mx_to_torch_sparse_tensor

from SIReNet.utils import  IndexSampler
import pandas as pd
#%%


np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

adj = sp.load_npz('adjacency.npz')
adj = normalize(adj)

features = pd.read_csv('node_features.csv')
indexes = features.index.values
labels = features['labels'].values

print(np.sum(labels)/labels.shape[0])

features = features.drop(['labels'],axis=1)
feature_transformer =  StandardScaler()
features = feature_transformer.fit_transform(features)

features = torch.FloatTensor(features)
labels = torch.LongTensor(labels)
adj = sparse_mx_to_torch_sparse_tensor(adj)


print(len(indexes))

idx_sampler = IndexSampler(indexes)

idx_train = torch.LongTensor(idx_sampler.sample(n_samples=2000))


idx_val = torch.LongTensor(idx_sampler.sample(n_samples=1000))

idx_test = torch.LongTensor(idx_sampler.sample_remaining())

#%%
print(adj.shape)
print(features.shape)
print(labels.shape)
print(idx_train.shape)
print(idx_val.shape)
print(idx_test.shape)

#%%
model = GCN(nfeat=features.shape[1],
            nhid=16,
            nclass=labels.max().item() + 1,
            dropout=0.5)
optimizer = optim.Adam(model.parameters(),
                       lr=0.01, weight_decay=5e-4)

model.cuda()
features = features.cuda()
adj = adj.cuda()
labels = labels.cuda()
idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()



#%%

fastmode = False

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(200):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

