import copy

import torch
from cyy_naive_pytorch_lib.trainer import Trainer
from torch.optim.sgd import SGD

from sign_sgd_server import SignSGDServer
from worker import Worker


class SignSGDWorker(Worker):
    def __init__(self, trainer: Trainer, server: SignSGDServer):
        assert isinstance(trainer.get_optimizer(), SGD)
        super().__init__(trainer, server)

    def train(self, device):
        self.trainer.train(
            device=device, optimizer_step_callbacks=[self.__get_gredient]
        )

    @torch.no_grad()
    def __get_gredient(self, optimizer, **kwargs):
        device = kwargs.get("device")
        gradient = list()
        for group in optimizer.param_groups:
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if momentum != 0:
                    param_state = optimizer.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                d_p = torch.sign(d_p).detach().cpu()
                gradient.append(d_p)
        self.server.add_gradient(gradient)
        gradient = copy.copy(self.server.get_gradient())
        for group in optimizer.param_groups:
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = gradient.pop(0).to(device)

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                p.add_(d_p, alpha=-group["lr"])
        assert not gradient
