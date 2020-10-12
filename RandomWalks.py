import dgl
import torch
import os
import numpy as np
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
import torch.nn as nn
import pkbar
import argparse
import dgl.function as fn

device = "cuda"
ratings = 5

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']

def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.number_of_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage = nn.RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()
    def forward(self, g, neg_g, x, etype):
        h = self.sage(g, x)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)


def load_graph(DATA_PATH , fname):
    print(os.path.join(DATA_PATH, fname))
    G = dgl.load_graphs(os.path.join(DATA_PATH, fname))[0][0]
    return G


def load_data(DATA_PATH):

    train_G = load_graph(DATA_PATH , "ua_train.graph")
    test_G = load_graph(DATA_PATH, "ua_test.graph")

    return train_G , test_G

def gen_embeddings(hetero_graph):
    def compute_loss(pos_score, neg_score):
        # Margin loss
        n_edges = pos_score.shape[0]
        return (1 - neg_score.view(n_edges, -1) + pos_score.unsqueeze(1)).clamp(min=0).mean()

    k = 5
    model = Model(10, 20, 5, hetero_graph.etypes)
    user_feats = hetero_graph.nodes['user'].data['feature']
    item_feats = hetero_graph.nodes['item'].data['feature']
    node_features = {'user': user_feats, 'item': item_feats}
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(200):
        negative_graph = construct_negative_graph(hetero_graph, k, ('user', 'rate', 'movie'))
        pos_score, neg_score = model(hetero_graph, negative_graph, node_features, ('user', 'rate', 'movie'))
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())

def run(DATA_PATH):

    train_G , test_G  = load_data(DATA_PATH)

    graphs = []

    for rating in range(1 , ratings + 1):
        sub = dgl.edge_type_subgraph(train_G , [ ("user", str(rating) + "u", "movie") ])
        sub = dgl.to_homogeneous(sub)
        sub = dgl.to_bidirected(sub)
        graphs.append(sub)

    for graph in graphs:
        embeddings = gen_embeddings(graph)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    DATA_PATH = "./data/ml-100k_processednew/"

    args = parser.parse_args()

    run(DATA_PATH)
