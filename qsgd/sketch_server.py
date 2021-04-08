from typing import List

import torch
import numpy as np
from cyy_naive_lib.data_structure.task_queue import RepeatedResult
from cyy_naive_lib.data_structure.thread_task_queue import ThreadTaskQueue

import sys
sys.path.append("..")
from server import Server


class SketchsgdServer(Server):
    def __init__(self, worker_number: int):
        super().__init__()
        self.worker_number = worker_number
        self.sketch_gradients: list = []
        self.gradients_queue = ThreadTaskQueue(
            worker_fun=self.__worker, worker_num=1)

    def stop(self):
        self.gradients_queue.stop()

    def add_gradient(self, sketch_gradients: List[torch.Tensor]):
        self.gradients_queue.add_task(sketch_gradients)

    def get_gradient(self) -> List[torch.Tensor]:
        return self.gradients_queue.get_result()


    def __worker(self, sketch_gradients: torch.Tensor, extra_args):
        self.sketch_gradients.append(sketch_gradients)
        if len(self.sketch_gradients) != self.worker_number:
            return None
        total_sketch_gradients = [sum(i) for i in zip(*self.sketch_gradients)]
        # for idx, grad in enumerate(total_sketch_gradients):
        #     total_sketch_gradients[idx] = torch.sign(sketch_gradients)

        self.sketch_gradients = []
        return RepeatedResult(data=total_sketch_gradients, num=self.worker_number)

