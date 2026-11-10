
from typing import Dict, Tuple
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch.utils.data import DataLoader
import freerec
from freerec.data.tags import USER, ITEM, TIMESTAMP, ID
from freerec.utils import infoLogger

from graph_upper_grad import GraphUpperGrad
from sampler import ASPIREValidSampler
    

def add_filter_args(cfg: freerec.parser.Parser):
    cfg.add_argument("--val-data-ratio", type=float, default=0.5)
    cfg.add_argument("--temperature", type=float, default=1.0)
    cfg.add_argument("--prefilter-norm", type=int, default=1) # 1 for true, 0 for false
    cfg.add_argument("--filter-init", type=str, choices=['full_spec', 'avg_pool', 'linear', 'mid_pass', 'high_pass', 'prior_full_spec', 'prior_high_pass'], default='full_spec') 
        # full_spec: y = 1; avg_pool: y = (1 + x + x^2 + ...) / L; linear: y = x; mid_pass: y = 1 - x^2; high_pass: y = 1 - 2x + x^2
    cfg.add_argument("--val-sampling", type=str, choices=['pair', 'ui'], default='ui')
    cfg.add_argument("--graph-w-dim", type=int, default=0)
    cfg.add_argument("--graph-w-unique-wd", type=int, default=0) # 1 for true, 0 for false

    cfg.add_argument("--graph-loss-type", type=str, choices=['ce', 'sample_ce', 'bpr', 'bce', 'mse'], default='ce')
    cfg.add_argument("--num-neg", type=int, default=1)
    cfg.add_argument("--graph-loss-weight", type=float, default=1.0)
    cfg.add_argument("--graph-optimizer", type=str, choices=['adam', 'sgd'], default='sgd')
    cfg.add_argument("--graph-lr", type=float, default=1.e-2)
    cfg.add_argument("--graph-momentum", type=float, default=0.1)
    cfg.add_argument("--graph-weight-decay", type=float, default=0.0)
    cfg.add_argument("--graph-warmup", type=int, default=0)
    cfg.add_argument("--graph-interval", type=int, default=5)

    cfg.add_argument("--use-implicit-grad", type=int, default=0) # 1 for true, 0 for false
    cfg.add_argument("--graph-implicit-approx", type=str, choices=['neumann', 'darts'], default='neumann')
    cfg.add_argument("--graph-implicit-lr", type=float, default=1.e-2)
    cfg.add_argument("--graph-max-gradnorm", type=float, default=1.)
    cfg.add_argument("--graph-darts-eps", type=float, default=1.e-2)

class Common_Graph(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.w_dim = cfg.graph_w_dim
        assert cfg.graph_w_dim <= cfg.embedding_dim

        self.graph_coefs = nn.ParameterDict()
        self.graph_linears = nn.ModuleDict()

    def get_init_filter(self, num_layers, init_method: str):
        init = torch.zeros(num_layers + 1, 1, 1)

        if init_method == 'full_spec':
            init[0] = 1.0
        elif init_method == 'avg_pool':
            init = torch.ones(num_layers + 1, 1, 1) / (num_layers + 1)
        elif init_method == 'linear':
            init[1] = 1.0
        elif init_method == 'mid_pass':
            init[0] = 1.0
            init[2] = -1.0
        elif init_method == 'high_pass':
            init[0] = 1.0
            init[1] = -2.0
            init[2] = 1.0
        elif init_method == 'prior_full_spec':
            assert num_layers == 3
            if "Amazon2014Baby_550_MMRec.yaml" in self.cfg.config:
                init = torch.tensor([1.209106, -0.326407, 1.812408, 1.526129]).view(-1, 1, 1)
            elif "Ciao_330_MMRec.yaml" in self.cfg.config:
                init = torch.tensor([0.473673, -0.530628, 3.413559, 1.690168]).view(-1, 1, 1)
            else:
                raise ValueError(f"Unknown config for prior_param initialization: {self.cfg.config}")
        elif init_method == 'prior_high_pass':
            assert num_layers == 3
            if "Amazon2014Baby_550_MMRec.yaml" in self.cfg.config:
                init = torch.tensor([1.467383, -0.931164, 1.892319, 1.854603]).view(-1, 1, 1)
            elif "Ciao_330_MMRec.yaml" in self.cfg.config:
                init = torch.tensor([-0.503082, 0.564295, -3.273188, -1.789663]).view(-1, 1, 1)
            else:
                raise ValueError(f"Unknown config for prior_param initialization: {self.cfg.config}")
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
        
        return init
    
    def register_graph(self, name: str, num_layers: int, init_method: str):
        if name in self.graph_coefs:
            raise ValueError(f"Graph '{name}' already registered.")
        assert num_layers > 0

        init = self.get_init_filter(num_layers, init_method)
        coef = nn.Parameter(init)
        self.graph_coefs[name] = coef

        linears = nn.ModuleList([
            nn.Linear(self.w_dim, self.w_dim, bias=False)
            for _ in range(num_layers)
        ]) if self.w_dim > 0 else nn.ModuleList()
        self.graph_linears[name] = linears

    def conv_op(self, X, graph, coefs: torch.Tensor, linears):
        allEmbds = [X]
        L = coefs.shape[0] - 1
        
        for i in range(L):
            X = graph @ allEmbds[-1]

            if self.w_dim > 0:
                x_front = X[:, :self.w_dim]
                x_back = X[:, self.w_dim:]

                x_front = linears[i](x_front)
                X = torch.cat([x_front, x_back], dim=1)

            allEmbds.append(X)

        weightedEmbds = torch.stack(allEmbds, dim=0) * coefs
        return weightedEmbds.sum(dim=0), allEmbds

    def forward(self, X_in, graph, graph_key: str):
        if graph_key not in self.graph_coefs:
            raise KeyError(
                f"Graph '{graph_key}' not registered. "
                f"Available: {list(self.graph_coefs.keys())}"
            )
        coefs = self.graph_coefs[graph_key]
        linears = self.graph_linears[graph_key]

        if self.cfg.prefilter_norm:
            X_in = F.normalize(X_in, dim=-1)
        out, allEmbds = self.conv_op(X_in, graph, coefs, linears)
        return out, allEmbds

    def append_graph_coefs_to_csv(self, epoch: int, csv_path: str):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.exists(csv_path)

        start_index = 1
        if file_exists:
            with open(csv_path, "r", encoding="utf-8") as f:
                row_count = sum(1 for _ in f)
                start_index = row_count 

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            for key, param in self.graph_coefs.items():
                coef = param.detach().cpu().view(-1).numpy()
                coef_list = coef.tolist()

                if not file_exists:
                    header = ["index", "epoch", "key"] + [f"l{i}coef" for i in range(len(coef_list))]
                    writer.writerow(header)
                    file_exists = True 

                row = [start_index, epoch, key] + coef_list
                writer.writerow(row)

                start_index += 1

    def dump_graph_coefs_and_funcs(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        coef_txt_path = os.path.join(save_dir, "coefs.txt")

        x = np.linspace(-1.25, 1.25, 400)
        lines = []

        for key, param in self.graph_coefs.items():
            coef = param.detach().cpu().view(-1).numpy()
            coef_list = coef.tolist()

            line = f"{key}: [{', '.join(f'{c:.6f}' for c in coef_list)}]"
            lines.append(line)

            # f(x) = c0 + c1 x + c2 x^2 + ...
            y = np.zeros_like(x)
            for i, c in enumerate(coef_list):
                y += c * (x ** i)

            plt.figure(figsize=(6, 4))

            plt.plot(x, y, linewidth=2.5, color="#1f77b4", label="f(x)")
            plt.axhline(0, linewidth=1.2, color="#888888", linestyle="--", alpha=0.8)
            plt.axvline(0, linewidth=1.2, color="#888888", linestyle="--", alpha=0.8)

            plt.title(f"Polynomial for {key}")
            plt.xlabel("x")
            plt.ylabel("f(x)")

            plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)

            fig_path = os.path.join(save_dir, f"coef_func_{key}.png")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()

        with open(coef_txt_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")


class MainModel(freerec.models.GenRecArch):
    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet, cfg
    ) -> None:
        super().__init__(dataset)
        self.cfg = cfg

        self.graph_model = Common_Graph(cfg)
        self.auxiliary_model = None

        self._train_data = dataset.train().to_bigraph((USER, ID), (ITEM, ID), edge_type='U2I')['U2I'].edge_index.to(cfg.device)

        full_val_data = dataset.valid().to_bigraph((USER, ID), (ITEM, ID), edge_type='U2I')['U2I'].edge_index.to(cfg.device)
        perm = torch.randperm(full_val_data.shape[1], device=cfg.device)
        keep = perm[: int(full_val_data.shape[1] * cfg.val_data_ratio)]
        self._val_data = full_val_data[:, keep]

        if self.cfg.val_sampling == 'pair':
            _val_probs = torch.ones_like(self._val_data[0])
        elif self.cfg.val_sampling == 'ui':
            val_data_users = self._val_data[0]
            counts = torch.bincount(val_data_users)
            _val_probs = 1.0 / counts[val_data_users].float()
        else:
            raise NotImplementedError
            
        self._val_sampling_probs = _val_probs / _val_probs.sum()

        self.criterion = freerec.criterions.BPRLoss(reduction='mean')
        if self.cfg.graph_loss_type == 'bpr' or self.cfg.graph_loss_type == 'bce':
            assert self.cfg.num_neg == 1

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().choiced_user_ids_source(
        ).gen_train_sampling_pos_().gen_train_sampling_neg_(
            num_negatives=1
        ).batch_(batch_size).tensor_()
    
    def sure_validpipe(self, ranking: str = 'full', batch_size: int = 512):
        return self.dataset.valid().ordered_user_ids_source(
        ).aspire_valid_sampling_(
            ranking, val_data=self._val_data
        ).batch_(batch_size).tensor_()

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

    def normalize_graph(self, edge_index: torch.Tensor, edge_weight: torch.Tensor, row_gamma = 0.5, col_gamma = 0.5):
        row, col = edge_index[0], edge_index[1] # symmetrical full graph
        #edge_weight = torch.ones_like(row)
        row_sum = 1.e-7 + scatter(edge_weight, row, dim=0, dim_size=self.User.count + self.Item.count)
        col_sum = 1.e-7 + scatter(edge_weight, col, dim=0, dim_size=self.User.count + self.Item.count)
        row_inv_sqrt = row_sum.pow(-row_gamma)
        col_inv_sqrt = col_sum.pow(-col_gamma)
        norm_weight = row_inv_sqrt[row] * col_inv_sqrt[col]

        # # Shift item indices to [num_users, num_users + num_items)
        # ui_row = row
        # ui_col = col + self.User.count

        # # Add symmetric edges: U → I and I → U
        # all_row = torch.cat([ui_row, ui_col])
        # all_col = torch.cat([ui_col, ui_row])
        # all_weight = torch.cat([norm_weight, norm_weight])

        all_row, all_col, all_weight = row, col, norm_weight

        return torch.sparse_coo_tensor(
                    indices=torch.stack([all_row, all_col]), values=all_weight, device=self.cfg.device,
                    size=(self.User.count + self.Item.count, self.User.count + self.Item.count)
                )#.to_sparse_csr()
    
    def _valid_sampling(self):
        sampling = torch.multinomial(
            self._val_sampling_probs,
            self.cfg.batch_size,
            replacement=True
        )
        users, positives = self._val_data[:, sampling]

        negatives = torch.randint(
            0,
            self.Item.count,
            (self.cfg.batch_size, self.cfg.num_neg),
            device=positives.device
        )
        return users, positives, negatives

    
    def graph_loss(self):
        '''
        return:
         - auxiliary (valid) set loss over the main task
        '''
        userEmbds, itemEmbds = self.encode()[:2]
        users, positives, negatives = self._valid_sampling()

        if self.cfg.graph_loss_type.lower() == 'ce':
            logits = userEmbds[users] @ itemEmbds.T / self.cfg.temperature
            valid_loss = F.cross_entropy(logits, positives)

        elif self.cfg.graph_loss_type.lower() == 'sample_ce':
            pos_scores = (userEmbds[users] * itemEmbds[positives]).sum(dim=1, keepdim=True)   # [B, 1]
            neg_scores = torch.einsum('bd,bkd->bk', userEmbds[users], itemEmbds[negatives])   # [B, K]
            logits = torch.cat([pos_scores, neg_scores], dim=1)
            logits = logits / self.cfg.temperature
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            valid_loss = F.cross_entropy(logits, labels)

        elif self.cfg.graph_loss_type.lower() == 'bpr':
            pos_scores = (userEmbds[users] * itemEmbds[positives]).sum(dim=1, keepdim=True)
            neg_scores = torch.einsum('bd,bkd->bk', userEmbds[users], itemEmbds[negatives])
            valid_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

        elif self.cfg.graph_loss_type.lower() == 'bce':
            pos_scores = (userEmbds[users] * itemEmbds[positives]).sum(dim=1, keepdim=True)
            neg_scores = torch.einsum('bd,bkd->bk', userEmbds[users], itemEmbds[negatives])
            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)
            valid_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels) + \
                         F.binary_cross_entropy_with_logits(neg_scores, neg_labels)
            
        elif self.cfg.graph_loss_type.lower() == 'mse':
            valid_loss = F.mse_loss(userEmbds[users], itemEmbds[positives])

        else:
            raise NotImplementedError

        return valid_loss * self.cfg.graph_loss_weight
    
    def encode(self):
        '''
        return:
         - user embedding for main task (emb loss) & auxiliary task (graph loss)
         - item embedding for main task (emb loss) & auxiliary task (graph loss)
         - user feature for auxiliary model
         - item feature for auxiliary model
        '''
        raise NotImplementedError()
    
    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        '''
        return:
         - embedding loss + regularization loss
        '''
        raise NotImplementedError()

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
    

class CoachForModel(freerec.launcher.Coach):

    def set_optimizer(self):
        graph_params = list(self.model.graph_model.parameters())
        excluded = set(map(id, graph_params))
        main_params = [p for p in self.model.parameters() if id(p) not in excluded]

        self.graph_params_for_aux = [p for p in graph_params if p is not None and p.numel() > 0 and p.requires_grad]
        self.main_params_for_aux = [p for p in main_params if p is not None and p.numel() > 0 and p.requires_grad]

        if self.cfg.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                main_params,
                lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        else:
            raise NotImplementedError(
                f"Unexpected optimizer {self.cfg.optimizer} ..."
            )

        if self.cfg.graph_w_unique_wd: # set graph linear weight_decay
            graph_linear_params = []
            for linears in self.model.graph_model.graph_linears.values():
                for layer in linears:
                    graph_linear_params += list(layer.parameters())

            graph_linear_ids = set(map(id, graph_linear_params))
            graph_other_params = [p for p in graph_params if id(p) not in graph_linear_ids]

            param_groups = []
            if len(graph_other_params) > 0:
                param_groups.append({
                    "params": graph_other_params,
                    "weight_decay": 0.0
                })
            if len(graph_linear_params) > 0:
                param_groups.append({
                    "params": graph_linear_params,
                    "weight_decay": self.cfg.graph_weight_decay
                })
        else:
            param_groups = graph_params
                    
        if self.cfg.graph_optimizer.lower() == 'adam':
            self.graph_optimizer = torch.optim.Adam(
                param_groups,
                lr=self.cfg.graph_lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.graph_weight_decay
            )
        elif self.cfg.graph_optimizer.lower() == 'sgd':
            self.graph_optimizer = torch.optim.SGD(
                param_groups,
                lr=self.cfg.graph_lr,
                weight_decay=self.cfg.graph_weight_decay,
                momentum=self.cfg.graph_momentum
            )
        else:
            raise NotImplementedError(
                f"Unexpected optimizer {self.cfg.optimizer} ..."
            )
        self.graph_upper_grad = GraphUpperGrad(self.cfg)
        
    def train_per_epoch(self, epoch: int):
        step = 0
        for data in self.dataloader:
            step += 1
            data = self.dict_to_device(data)
            
            if epoch >= self.cfg.graph_warmup and step % self.cfg.graph_interval == 0:
                graph_grads = self.graph_upper_grad.compute(
                    model=self.model,
                    data=data,
                    main_params=self.main_params_for_aux,
                    graph_params=self.graph_params_for_aux
                )

                self.graph_optimizer.zero_grad()
                for param, grad in zip(self.graph_params_for_aux, graph_grads):
                    param.grad = grad
                self.graph_optimizer.step()

            main_loss = self.model.fit(data)

            self.optimizer.zero_grad()
            main_loss.backward()
            self.optimizer.step()

            self.monitor(
                main_loss.item(), 
                n=len(data[self.User]), reduction="mean", 
                mode='train', pool=['LOSS']
            )

    def evaluate(self, epoch: int, step: int = -1, mode: str = 'valid'):
        if epoch == self.cfg.epochs:
            save_path = os.path.join(self.cfg.LOG_PATH, 'last_filter')
            self.model.graph_model.dump_graph_coefs_and_funcs(save_path)    

        save_hist_csv = os.path.join(self.cfg.LOG_PATH, 'filter_hist.csv')
        self.model.graph_model.append_graph_coefs_to_csv(epoch, save_hist_csv)  

        self.get_res_sys_arch().reset_ranking_buffers()
        for data in self.dataloader:
            bsz = data[self.Size]
            data = self.dict_to_device(data)

            if self.cfg.ranking == 'full':
                scores = self.model(data, ranking='full')
                if self.remove_seen:
                    seen = self.Item.to_csr(data[self.ISeen]).to(self.device).to_dense().bool()
                    scores[seen] = -1e23
                targets = self.Item.to_csr(data[self.IUnseen]).to(self.device).to_dense()
            else:
                raise NotImplementedError(
                    f"`ranking` should be 'full' or 'pool' but {self.cfg.ranking} received ..."
                )

            self.monitor(
                scores, targets,
                n=bsz, reduction="mean", mode=mode,
                pool=['HITRATE', 'PRECISION', 'RECALL', 'NDCG', 'MRR']
            )

    def check_best(self, epoch: int, step: int = -1) -> None:
        """Update best value."""
        if self.meter4best.active:
            best_ = self.meter4best.which_is_better(self._best)
            if best_ != self._best:
                save_path = os.path.join(self.cfg.LOG_PATH, 'best_filter')
                self.model.graph_model.dump_graph_coefs_and_funcs(save_path)
        
        super().check_best(epoch, step)
