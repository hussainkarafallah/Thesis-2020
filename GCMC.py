import dgl
import torch
import os
import numpy as np
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
import torch.nn as nn
import pkbar
import argparse

device = "cuda"


def get_SAGEConv( in_feats , hid_feats , agg = 'mean' , activation = F.relu , bias = True , norm = None):
    return dglnn.conv.SAGEConv( in_feats , hid_feats , agg , activation= activation , bias=bias, norm=norm)

def get_GATConv(in_feats, hid_feats, num_heads = 1, feat_drop=0.4, attn_drop=0.4, activation=F.relu):
    return dglnn.conv.GATConv(in_feats , hid_feats , num_heads=num_heads , feat_drop=feat_drop , attn_drop=attn_drop
                              , activation = activation)
class MovieLensNetwork(torch.nn.Module):

    def __init__(self, in_feats, hid_feats, conv_depth , ufeats, mfeats, ratings , main_layer = 'SAGE'):

        super(MovieLensNetwork, self).__init__()

        assert (in_feats == (ufeats.shape[1], mfeats.shape[1]))

        self.userfeatures = nn.Parameter(ufeats)
        self.moviefeatures = nn.Parameter(mfeats)
        self.feats = nn.ParameterDict({"user": self.userfeatures, "movie": self.moviefeatures})
        self.conv_depth = conv_depth
        self.main_layer = main_layer

        print(self.userfeatures.shape)
        print(self.moviefeatures.shape)

        self.ratings = ratings

        self.userfdim, self.moviefdim = in_feats

        self.hid_feats = hid_feats

        self.conv_dictionaries = nn.ModuleList([nn.ModuleDict() for _ in range(self.conv_depth)])


        #conv1dict, conv2dict = nn.ModuleDict(), nn.ModuleDict()

        print(self.main_layer)

        for (idx , curconv_dict) in enumerate(self.conv_dictionaries):

            norm = F.normalize

            if idx == 0:
                in_feats1 = (self.userfdim, self.moviefdim)
                in_feats2 = (self.moviefdim, self.userfdim)
            else:
                in_feats1 = (self.ratings * self.hid_feats)
                in_feats2 = (self.ratings * self.hid_feats)

            if idx == len(self.conv_dictionaries) - 1:
                norm = None


            for i in range(1, self.ratings + 1):
                if self.main_layer == 'SAGE':
                    curconv_dict[str(i) + "u"] = get_SAGEConv(in_feats1 , self.hid_feats , 'mean' , activation=F.relu,
                                                      bias = True , norm = norm)
                    curconv_dict[str(i) + "m"] = get_SAGEConv(in_feats2, self.hid_feats, 'mean', activation=F.relu,
                                                      bias=True, norm=norm)
                if self.main_layer == 'GAT':
                    curconv_dict[str(i) + "u"] = get_GATConv(in_feats1, self.hid_feats,  activation=F.relu)

                    curconv_dict[str(i) + "m"] = get_GATConv(in_feats2, self.hid_feats,  activation=F.relu)



        self.convs = nn.ModuleList([dglnn.HeteroGraphConv(convdict, aggregate='stack') for convdict in self.conv_dictionaries])

        self.decoders = nn.ParameterDict({
            str(i): nn.Parameter(
                torch.randn( ( self.hid_feats * self.ratings , self.hid_feats * self.ratings  ), dtype=torch.float32) \
                , requires_grad=True) for i in range(1, self.ratings + 1)

        })

        for key in self.decoders.keys():
            nn.init.xavier_uniform_(self.decoders[key])

    @classmethod
    def relax(cls, res):
        for rel in res.keys():
            sh = res[rel].shape
            res[rel] = torch.reshape(res[rel], (sh[0], -1,))
        return res

    def forward(self, G):

        assert(len(self.convs) > 0)
        # x = x.to(self.device)

        # convolution over the 1st layer
        res = self.convs[0](G, (self.feats, self.feats))
        self.relax(res)

        for i in range(1 , len(self.convs)):
            res = self.convs[i](G, (res, res))
            self.relax(res)

        probs_tensors = []

        for rating in range(1, self.ratings + 1):
            u, v = G.edges(etype=str(rating) + "u")
            u, v = u.long(), v.long()

            users_embeddings = torch.index_select(res["user"], index=u, dim=0)
            movies_embeddings = torch.index_select(res["movie"], index=v, dim=0)

            n = users_embeddings.shape[0]
            d = users_embeddings.shape[1]

            dots_tensors = [
                torch.bmm( (users_embeddings @ decoder).view((n, 1, -1)), movies_embeddings.view(n, d, -1)).view((n, -1))
                for _, decoder in self.decoders.items()]
            dots = torch.stack(dots_tensors, dim=1).squeeze()
            probs = F.log_softmax(dots, dim=1)
            probs_tensors.append(probs)

        return probs_tensors

mainLayer = 'SAGE'
conv_depth = 1

def load_graph(DATA_PATH , fname):
    print(os.path.join(DATA_PATH, fname))
    G = dgl.load_graphs(os.path.join(DATA_PATH, fname))[0][0]
    return G

def load_data(DATA_PATH):

    users_feats = np.load(os.path.join(DATA_PATH, "users.npy"))
    users_feats = torch.from_numpy(users_feats).float()
    movies_feats = np.load(os.path.join(DATA_PATH, "movies.npy"))
    movies_feats = torch.from_numpy(movies_feats).float()

    train_G = load_graph(DATA_PATH , "ua_train.graph")
    test_G = load_graph(DATA_PATH, "ua_test.graph")

    return train_G , test_G , users_feats , movies_feats

def calc_metrics(probs):

    loss, acc, rmse, full_cnt = 0, 0, 0, 0

    criterion = nn.NLLLoss()

    for cls, probtensor in enumerate(probs):
        ypred = torch.argmax(probtensor, dim=1).view(-1)
        correct = torch.sum(torch.eq(ypred, cls)).item()
        ytrue = torch.full(size=(probtensor.shape[0],), fill_value=cls, dtype=torch.int64).to(device)

        loss += criterion(probtensor, ytrue)
        acc += correct
        rmse += ((ytrue - ypred) ** 2).sum()

        full_cnt += ypred.shape[0]

    acc /= full_cnt
    rmse = rmse.float() / full_cnt
    rmse = round(rmse.item() ** 0.5, 2)

    return loss , acc , rmse

def go_train(model , train_G , test_G):

    train_G = train_G.to(torch.device(device))
    test_G = test_G.to(torch.device(device))

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    epochs = 2000

    best = 1e20

    kbar = pkbar.Kbar(target=epochs, width=20)

    for epoch in range(epochs):

        model.train()
        optimizer.zero_grad()

        probs = model(train_G)
        loss , acc , tr_rmse = calc_metrics(probs)
        loss.backward()
        optimizer.step()
        loss /= len(probs)

        model.eval()
        probs2 = model(test_G)
        _ , _ , val_rmse = calc_metrics(probs2)

        if val_rmse < best:
            best = val_rmse
            patience = 0
        else:
            if patience > 10:
                break

        kbar.add(1, values=[("loss", loss), ("acc", acc), ("tr_rmse", tr_rmse) , ("val_rmse" , val_rmse)])


def run(DATA_PATH):
    train_G , test_G , users_feats , movies_feats = load_data(DATA_PATH)

    ufeats = users_feats.shape[1]
    vfeats = movies_feats.shape[1]

    print("mensss")
    print(mainLayer)
    print(conv_depth)

    model = MovieLensNetwork((ufeats, vfeats), hid_feats=75 , conv_depth= conv_depth, ufeats=users_feats, mfeats=movies_feats,
                             ratings=5 , main_layer= mainLayer)

    #return
    go_train(model , train_G , test_G)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--layer" , type = str , action = 'store' ,
        help = "The layer type you want to use inside the model 'GCN','SAGE','GAT'")
    parser.add_argument("--depth", type = int , action= 'store' ,
        help = "The depth of the graph neural network (>2 isn't ideal at all)")

    DATA_PATH = "./data/ml-100k_processednew/"

    args = parser.parse_args()

    if args.layer:
        assert args.layer in ['SAGE' , 'GAT']
        mainLayer = args.layer

    if args.depth:
        assert args.depth > 0
        conv_depth = args.depth

    run(DATA_PATH)


