import copy

import torch
import numpy as np
from cyy_naive_pytorch_lib.trainer import Trainer
from torch.optim.sgd import SGD

from sketch_server import SketchsgdServer
from csvec import CSVec

import sys 
sys.path.append("..")
from worker import Worker


class SketchsgdWorker(Worker):
    def __init__(self, trainer: Trainer, server: SketchsgdServer):
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

            # mask
            sketchMask = []
            D = 0
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if momentum != 0:
                    param_state = optimizer.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # d_p = torch.sign(d_p).detach().cpu()
                # d_p = function_gradient(sketch_gradients=d_p).detach().cpu()
                # sketch 
                sketchParamsLargerThan = 0
                p.do_sketching = p.numel() >= sketchParamsLargerThan
                
                if p.requires_grad:
                    size = np.prod(p.data.shape)
                    # todo: do_sketching
                    if p.do_sketching:
                        sketchMask.append(torch.ones(size))
                    else:
                        sketchMask.append(torch.zeros(size))
            D += size
            sketchMask = torch.cat(sketchMask).bool().to(device)
            d_p = CSVec(d=sketchMask.sum().item(),
                            c=20, r=5, device=device, numBlocks=1)
            print('dp', d_p)
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

def function_gradient(q_gradients):
    level = 8   # set level
    q_gradients = q_gradients.cpu()
    norm = q_gradients.norm()
    level_float = level * np.abs(q_gradients) / norm
    previous_level = np.floor(level_float)
    is_next_level = torch.rand(*q_gradients.shape) < (level_float - previous_level)
    new_level = previous_level + is_next_level
    tensor_compressed = np.sign(q_gradients) * norm * new_level / level

    return tensor_compressed