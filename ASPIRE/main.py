

from typing import Dict, Tuple

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
import freerec

from augment_graph import get_graph
from module import MainModel, CoachForModel, add_filter_args

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3)

cfg.add_argument("--freedom_mod_k", type=int, default=0)
cfg.add_argument("--freedom_soc_k", type=float, default=0.0)

cfg.add_argument("--mod_kappa", type=int, default=0)
cfg.add_argument("--soc_kappa", type=float, default=0)
cfg.add_argument("--shift_alpha", type=float, default=0.0) # always 0.0
cfg.add_argument("--shift_beta", type=float, default=1.0) # always 1.0
cfg.add_argument("--weight4mAdj", type=float, default=0.1, help="weight for fusing vAdj and tAd")
cfg.add_argument("--noise_scale", type=float, default=0.0) # for robustness evaluation

cfg.add_argument("--vfile", type=str, default="visual_modality.pkl", help="the file of visual modality features")
cfg.add_argument("--tfile", type=str, default="textual_modality.pkl", help="the file of textual modality features")

# ['none', 'text_all-MiniLM-L6-v2.pkl', 'text_SFR-Embedding-Mistral.pkl', 'text_Qwen2.5-7B.pkl']
cfg.add_argument("--text-comparison", type=str, default="none")
cfg.add_argument("--text-style", type=str, choices=['stair', 'alpharec'], default="stair")

cfg.add_argument("--loss-type", type=str, choices=['bpr', 'bce', 'ce'], default="bpr")

add_filter_args(cfg)

cfg.set_defaults(
    description="ASPIRE",
    root="../../data",
    dataset='Amazon2014Baby_550_MMRec',
    epochs=1000,
    batch_size=2048,
    optimizer='adam',
    lr=1e-4,
    weight_decay=0.0,
    seed=2026
)
cfg.compile()


class SideInfoGraph(nn.Module):

    def __init__(
        self,
        num_users: int,
        num_items: int,
        cfg,
        dataset: freerec.data.datasets.base.RecDataSet,
        graph_model
    ) -> None:
        super().__init__()
        assert cfg.mod_kappa == 0 and cfg.soc_kappa == 0.0
        assert cfg.freedom_mod_k == 0 or cfg.freedom_soc_k == 0.0

        self.num_users = num_users
        self.num_items = num_items

        if cfg.freedom_mod_k > 0:
            self.load_feats(dataset.path)
        elif cfg.freedom_soc_k > 0:
            self.load_soc_graph(dataset.path)
        else:
            raise NotImplementedError
        
        self.graph_model = graph_model
        self.graph_model.register_graph('side', cfg.num_layers, cfg.filter_init)

    def load_soc_graph(self, path):
        import pandas as pd
        social_edges = pd.read_csv(os.path.join(path, 'trust_mat.txt'), sep='\t', header=0)
        social_edges = torch.tensor(social_edges.to_numpy()).T
        edge_index = torch.unique(torch.cat([social_edges, torch.flip(social_edges, dims=[0])], dim=1), dim=1)
        edge_weight = torch.ones_like(edge_index[0]) * cfg.freedom_soc_k

        rows, cols = edge_index[0], edge_index[1]
        row_degree = 1.e-7 + scatter(edge_weight, rows, dim=0, dim_size=self.num_users)
        row_deg = row_degree.pow(-0.5)
        col_degree = 1.e-7 + scatter(edge_weight, cols, dim=0, dim_size=self.num_users)
        col_deg = col_degree.pow(-0.5)
        edge_weight = edge_weight * row_deg[rows] * col_deg[cols]

        eye_tensor = torch.tile(torch.arange(0, self.num_users), (2, 1))
        edge_index = torch.cat([edge_index, eye_tensor], dim=1)
        edge_weight = torch.cat([edge_weight, - torch.ones_like(eye_tensor[0]) * cfg.shift_alpha]) / cfg.shift_beta

        mAdj = torch.sparse_coo_tensor(
                edge_index, edge_weight.to(dtype=torch.float32),
                size=(self.num_users, self.num_users),
                device=cfg.device
            )
        self.register_buffer(
            'mAdj',
            mAdj.to_sparse_csr()
        )
    
    def load_feats(self, path: str):
        r"""
        Load v/t Features.

        Note: Following the offical implementation,
        they are stored as nn.Embedding and are trainable in default.
        I tried a frozen variant on Baby and found this operation makes no difference.
        """
        from freerec.utils import import_pickle
        if cfg.vfile:
            vFeats = import_pickle(
                os.path.join(path, cfg.vfile)
            )
            self.vFeats = nn.Embedding.from_pretrained(vFeats, freeze=False)
            vAdj = self.get_knn_graph(vFeats)

        if cfg.tfile:
            tFeats = import_pickle(
                os.path.join(path, cfg.tfile)
            )
            self.tFeats = nn.Embedding.from_pretrained(tFeats, freeze=False)
            tAdj = self.get_knn_graph(tFeats)

        if cfg.vfile and cfg.tfile:
            mAdj = vAdj * cfg.weight4mAdj + tAdj * (1 - cfg.weight4mAdj)
        elif cfg.vfile:
            mAdj = vAdj
        elif cfg.tfile:
            mAdj = tAdj
        else:
            raise NotImplementedError("At least visual or texual modality should be given ...")
        self.register_buffer(
            'mAdj',
            mAdj.to_sparse_csr()
        )

    def get_knn_graph(self, features: torch.Tensor):
        r"""
        Compute the kNN graph.

        Note: Following the offical implementation,
        this graph is not symmetric.
        """
        features = F.normalize(features, dim=-1) # (N, D)
        sim = features @ features.t() # (N, N)
        edge_index, _ = freerec.graph.get_knn_graph(
            sim, cfg.freedom_mod_k, symmetric=False
        )

        edge_weight = torch.ones_like(edge_index[0])
        rows, cols = edge_index[0], edge_index[1]
        row_degree = 1.e-7 + scatter(edge_weight, rows, dim=0, dim_size=self.num_items)
        deg_row_inv_sqrt = row_degree.pow(-0.5)
        col_degree = 1.e-7 + scatter(edge_weight, cols, dim=0, dim_size=self.num_items)
        deg_col_inv_sqrt = col_degree.pow(-0.5)
        edge_weight = edge_weight * deg_row_inv_sqrt[rows] * deg_col_inv_sqrt[cols]

        eye_tensor = torch.tile(torch.arange(0, self.num_items), (2, 1))
        edge_index = torch.cat([edge_index, eye_tensor], dim=1)
        edge_weight = torch.cat([edge_weight, - torch.ones_like(eye_tensor[0]) * cfg.shift_alpha]) / cfg.shift_beta

        return torch.sparse_coo_tensor(
            edge_index, edge_weight,
            size=(self.num_items, self.num_items)
        )
    

    def forward(self, sideEmbds: torch.Tensor):
        avgEmbds, _ = self.graph_model(sideEmbds, self.mAdj, 'side')
        return avgEmbds


class ASPIRE(MainModel):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet, cfg
    ) -> None:
        super().__init__(dataset, cfg)
        self.cfg = cfg

        if cfg.freedom_mod_k > 0 or cfg.freedom_soc_k > 0:
            self.side_graph = SideInfoGraph(
                self.User.count,
                self.Item.count,
                cfg=cfg,
                dataset=dataset,
                graph_model=self.graph_model,
            )

        self.num_layers = cfg.num_layers

        self.User.add_module(
            "embeddings", nn.Embedding(
                self.User.count, cfg.embedding_dim
            )
        )

        self.Item.add_module(
            "embeddings", nn.Embedding(
                self.Item.count, cfg.embedding_dim
            )
        )

        rec_adj, _, _ = get_graph(dataset, cfg, self.User.count, self.Item.count)
        self.register_buffer("Adj", rec_adj)
        self.graph_model.register_graph('key', cfg.num_layers, self.cfg.filter_init)

        if cfg.loss_type == 'bce':
            self.criterion = freerec.criterions.BCELoss4Logits(reduction='mean')
        elif cfg.loss_type == 'bpr':
            self.criterion = freerec.criterions.BPRLoss(reduction='mean')
        elif cfg.loss_type == 'ce':
            self.criterion = freerec.criterions.CrossEntropy4Logits(reduction='mean')
        else:
            self.criterion = freerec.criterions.BPRLoss(reduction='mean')

        self.reset_parameters()

    def whitening(self, feats: torch.Tensor):
        import math
        feats = feats - feats.mean(0, keepdim=True)
        feats, _, _ = torch.linalg.svd(feats, full_matrices=False)
        return feats[:, :cfg.embedding_dim] * math.sqrt(self.Item.count / cfg.embedding_dim)
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.e-4)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

        if 'text_comparison' in cfg.dataset and cfg.text_comparison != 'none':
            from freerec.utils import import_pickle
            self.item_llm_emb = import_pickle(
                os.path.join(self.dataset.path, cfg.text_comparison)
            ).to(cfg.device).float()
            init_emb = self.whitening(self.item_llm_emb)
            self.Item.embeddings.weight.data.copy_(init_emb)
            print(f'use {cfg.text_comparison} init item emb')

            edge_index = self.dataset.train().to_bigraph(edge_type='u2i')['u2i'].edge_index
            edge_index, edge_weight = freerec.graph.to_normalized(edge_index, normalization='left')
            R = torch.sparse_coo_tensor(
                edge_index, edge_weight, size=(self.User.count, self.Item.count), device=cfg.device
            ).to_sparse_csr()

            self.user_llm_emb = R @ self.item_llm_emb
            self.User.embeddings.weight.data.copy_(R @ init_emb)
            print(f'use {cfg.text_comparison} init user emb')

            # To keep the same parameter size
            mlp_hidden_dim = int((4096 + cfg.embedding_dim) * cfg.embedding_dim / (self.item_llm_emb.shape[-1] + cfg.embedding_dim))
            self.text_mlp = nn.Sequential(
                nn.Linear(self.item_llm_emb.shape[-1], mlp_hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(mlp_hidden_dim, cfg.embedding_dim)
            )
            # for m in self.text_mlp:
            #     if isinstance(m, nn.Linear):
            #         nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
            #         nn.init.zeros_(m.bias)

    def get_user_emb(self):
        if 'text_comparison' in cfg.dataset and cfg.text_comparison != 'none':
            if cfg.text_style == 'alpharec':
                return self.text_mlp(self.user_llm_emb)
            elif cfg.text_style == 'stair':
                return self.User.embeddings.weight
            else:
                raise NotImplementedError
        else:
            return self.User.embeddings.weight
    
    def get_item_emb(self):
        if 'text_comparison' in cfg.dataset and cfg.text_comparison != 'none':
            if cfg.text_style == 'alpharec':
                return self.text_mlp(self.item_llm_emb)
            elif cfg.text_style == 'stair':
                return self.Item.embeddings.weight
            else:
                raise NotImplementedError
        else:
            return self.Item.embeddings.weight
        
    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().choiced_user_ids_source(
        ).gen_train_sampling_pos_().gen_train_sampling_neg_(
            num_negatives=1
        ).batch_(batch_size).tensor_()
        
    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        allEmbds = torch.cat(
            (self.get_user_emb(), self.get_item_emb()), dim=0
        ) # (N, D)
        
        avgEmbds, _ = self.graph_model(allEmbds, self.Adj, 'key')

        userEmbds, itemEmbds = torch.split(
            avgEmbds, (self.User.count, self.Item.count)
        )
        #return userEmbds, itemEmbds

        if cfg.freedom_mod_k > 0:
            iiEmbds = self.side_graph(self.get_item_emb())
            ifinalEmbds = (itemEmbds + iiEmbds) / 2
            ufinalEmbds = userEmbds
        elif cfg.freedom_soc_k > 0:
            uuEmbds = self.side_graph(self.get_user_emb())
            ufinalEmbds = (userEmbds + uuEmbds) / 2
            ifinalEmbds = itemEmbds
        else:
            ufinalEmbds, ifinalEmbds = userEmbds, itemEmbds
        
        return ufinalEmbds, ifinalEmbds
    
    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        uallEmbds, iallEmbds = self.encode()
        users, positives, negatives = data[self.User], data[self.IPos], data[self.INeg]
        userEmbds = uallEmbds[users] # (B, 1, D)
        iposEmbds = iallEmbds[positives] # (B, 1, D)
        inegEmbds = iallEmbds[negatives] # (B, K, D)
        
        posLogits = torch.einsum("BKD,BKD->BK", userEmbds, iposEmbds)
        negLogits = torch.einsum("BKD,BKD->BK", userEmbds, inegEmbds)
        if cfg.loss_type == 'bpr':
            rec_loss = self.criterion(posLogits, negLogits)
        elif cfg.loss_type == 'bce':
            posLabels = torch.ones_like(posLogits)
            negLabels = torch.zeros_like(negLogits)
            rec_loss = self.criterion(posLogits, posLabels) + \
                self.criterion(negLogits, negLabels)
        elif cfg.loss_type == 'ce':
            logits = torch.einsum("MD,ND->MN", userEmbds.squeeze(), iallEmbds) # (M, N)
            rec_loss = self.criterion(logits, positives.squeeze())
        else:
            raise NotImplementedError

        return rec_loss

    def reset_ranking_buffers(self):
        """This method will be executed before evaluation."""
        userEmbds, itemEmbds = self.encode()
        self.ranking_buffer = dict()
        self.ranking_buffer[self.User] = userEmbds.detach().clone()
        self.ranking_buffer[self.Item] = itemEmbds.detach().clone()

    def recommend_from_full(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.ranking_buffer[self.User][data[self.User]] # (B, 1, D)
        itemEmbds = self.ranking_buffer[self.Item]
        return torch.einsum("BKD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.ranking_buffer[self.User][data[self.User]] # (B, 1, D)
        itemEmbds = self.ranking_buffer[self.Item][data[self.IUnseen]] # (B, 101, D)
        return torch.einsum("BKD,BKD->BK", userEmbds, itemEmbds)
    

def main():

    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = ASPIRE(dataset, cfg)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForModel(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg
    )
    coach.fit()


if __name__ == "__main__":
    main()
