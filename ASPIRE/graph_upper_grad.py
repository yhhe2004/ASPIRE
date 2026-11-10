import torch


class GraphUpperGrad:
    def __init__(self, cfg):
        self.cfg = cfg

    def compute(self, model, data, main_params, graph_params):
        if self.cfg.graph_implicit_approx == 'neumann':
            return self._neumann(model, data, main_params, graph_params)
        if self.cfg.graph_implicit_approx == 'darts':
            return self._darts(model, data, main_params, graph_params)
        raise ValueError(f"Unknown graph_implicit_approx: {self.cfg.graph_implicit_approx}")

    @staticmethod
    def _materialize(grads, params):
        return [
            torch.zeros_like(param) if grad is None else grad
            for grad, param in zip(grads, params)
        ]

    @staticmethod
    def _add_grads(left, right):
        return [lgrad + rgrad for lgrad, rgrad in zip(left, right)]

    @staticmethod
    def _add_to_params(params, updates, scale):
        with torch.no_grad():
            for param, update in zip(params, updates):
                if update is not None:
                    param.add_(update, alpha=scale)

    @staticmethod
    def _grad_norm(grads):
        total = None
        for grad in grads:
            if grad is None:
                continue
            value = grad.detach().pow(2).sum()
            total = value if total is None else total + value
        if total is None:
            return None
        return total.sqrt()

    def _upper_loss(self, model):
        return model.graph_loss()

    def _grads(self, loss, params, retain_graph=False, create_graph=False):
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=True
        )
        return self._materialize(grads, params)

    def _clip_implicit(self, grads):
        max_norm = self.cfg.graph_max_gradnorm
        if max_norm is None or max_norm <= 0:
            return grads

        norm = self._grad_norm(grads)
        if norm is None or norm.item() <= max_norm:
            return grads

        scale = max_norm / (norm.item() + 1.e-6)
        return [grad * scale for grad in grads]

    def _neumann(self, model, data, main_params, graph_params):
        upper_loss = self._upper_loss(model)
        explicit = self._grads(
            upper_loss,
            graph_params,
            retain_graph=bool(self.cfg.use_implicit_grad)
        )
        if not self.cfg.use_implicit_grad:
            return explicit

        train_loss = model.fit(data)
        implicit = self._neumann_implicit(
            upper_loss,
            train_loss,
            main_params,
            graph_params
        )
        return self._add_grads(explicit, self._clip_implicit(implicit))

    def _neumann_implicit(self, upper_loss, train_loss, main_params, graph_params):
        dupper_dmain = torch.autograd.grad(
            upper_loss,
            main_params,
            retain_graph=True,
            allow_unused=True
        )
        dtrain_dmain = torch.autograd.grad(
            train_loss,
            main_params,
            create_graph=True,
            allow_unused=True
        )

        active = [
            (param, torch.zeros_like(param) if upper_grad is None else upper_grad, train_grad)
            for param, upper_grad, train_grad in zip(main_params, dupper_dmain, dtrain_dmain)
            if train_grad is not None
        ]
        if not active:
            return [torch.zeros_like(param) for param in graph_params]

        active_main, active_upper, active_train = map(list, zip(*active))
        inverse_hvp = self._approx_inverse_hvp(active_upper, active_train, active_main)
        cross_grads = torch.autograd.grad(
            active_train,
            graph_params,
            grad_outputs=inverse_hvp,
            allow_unused=True
        )
        cross_grads = self._materialize(cross_grads, graph_params)
        return [-grad for grad in cross_grads]

    def _approx_inverse_hvp(self, dupper_dmain, dtrain_dmain, main_params):
        p = list(dupper_dmain)
        v = list(dupper_dmain)

        for _ in range(3):
            hvp = torch.autograd.grad(
                dtrain_dmain,
                main_params,
                grad_outputs=v,
                retain_graph=True,
                allow_unused=True
            )
            hvp = self._materialize(hvp, main_params)
            hvp = [grad * self.cfg.graph_implicit_lr for grad in hvp]
            v = [curr_v - curr_hvp for curr_v, curr_hvp in zip(v, hvp)]
            p = [curr_p + curr_v for curr_p, curr_v in zip(p, v)]

        return p

    def _darts(self, model, data, main_params, graph_params):
        train_loss = model.fit(data)
        train_main_grads = self._grads(train_loss, main_params)

        self._add_to_params(main_params, train_main_grads, -self.cfg.graph_implicit_lr)
        upper_loss = self._upper_loss(model)
        explicit = self._grads(
            upper_loss,
            graph_params,
            retain_graph=bool(self.cfg.use_implicit_grad)
        )
        if not self.cfg.use_implicit_grad:
            self._add_to_params(main_params, train_main_grads, self.cfg.graph_implicit_lr)
            return explicit

        upper_main_grads = self._grads(upper_loss, main_params)
        self._add_to_params(main_params, train_main_grads, self.cfg.graph_implicit_lr)

        implicit = self._darts_implicit(model, data, main_params, upper_main_grads, graph_params)
        return self._add_grads(explicit, self._clip_implicit(implicit))

    def _darts_implicit(self, model, data, main_params, upper_main_grads, graph_params):
        norm = self._grad_norm(upper_main_grads)
        if norm is None or norm.item() == 0:
            return [torch.zeros_like(param) for param in graph_params]

        eps = self.cfg.graph_darts_eps / norm.item()

        self._add_to_params(main_params, upper_main_grads, eps)
        loss_pos = model.fit(data)
        grads_pos = self._grads(loss_pos, graph_params)
        self._add_to_params(main_params, upper_main_grads, -eps)

        self._add_to_params(main_params, upper_main_grads, -eps)
        loss_neg = model.fit(data)
        grads_neg = self._grads(loss_neg, graph_params)
        self._add_to_params(main_params, upper_main_grads, eps)

        return [
            -self.cfg.graph_implicit_lr * (pos - neg) / (2 * eps)
            for pos, neg in zip(grads_pos, grads_neg)
        ]
