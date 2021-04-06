from cyy_naive_lib.data_structure.thread_task_queue import ThreadTaskQueue
from cyy_naive_pytorch_lib.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue


class Server:
    def __init__(self, tester, worker_number, multi_process: bool):
        self.__tester = tester
        self.__worker_num = worker_number
        if multi_process:
            self.__worker_data_queue = TorchProcessTaskQueue(
                worker_fun=self._process_worker_data
            )
        else:
            self.__worker_data_queue = ThreadTaskQueue(
                worker_fun=self._process_worker_data
            )

    @property
    def tester(self):
        return self.__tester

    @property
    def worker_number(self):
        return self.__worker_num

    @property
    def worker_data_queue(self):
        return self.__worker_data_queue

    def stop(self):
        self.worker_data_queue.stop()

    def _process_worker_data(self, *args, **kwargs):
        pass
