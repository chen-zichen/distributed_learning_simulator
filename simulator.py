import argparse
import copy
import datetime
import os

from cyy_naive_lib.data_structure.process_pool import ProcessPool
from cyy_naive_lib.data_structure.thread_pool import ThreadPool
from cyy_naive_lib.log import get_logger, set_file_handler
from cyy_naive_pytorch_lib.dataset import DatasetUtil
from cyy_naive_pytorch_lib.device import get_cuda_devices
from cyy_naive_pytorch_lib.ml_type import MachineLearningPhase

from config import get_config
from factory import get_server, get_worker

if __name__ == "__main__":
    config = get_config()
    # Let python initialize pool
    ProcessPool().exec(lambda: "1")

    set_file_handler(
        os.path.join(
            "log",
            config.distributed_algorithm,
            config.dataset_name,
            config.model_name,
            "{date:%Y-%m-%d_%H:%M:%S}.log".format(date=datetime.datetime.now()),
        )
    )

    trainer = config.create_trainer()
    training_datasets = DatasetUtil(trainer.dataset).iid_split(
        [1] * config.worker_number
    )
    tester = trainer.get_inferencer(phase=MachineLearningPhase.Test)
    server = get_server(
        config.distributed_algorithm,
        worker_number=config.worker_number,
        tester=tester,
    )

    devices = get_cuda_devices()
    worker_pool = ThreadPool()

    for worker_id in range(config.worker_number):
        get_logger().info("create worker %s", worker_id)
        worker_trainer = copy.deepcopy(trainer)

        worker_trainer.dataset_collection.transform_dataset(
            MachineLearningPhase.Training,
            lambda old_dataset: training_datasets[worker_id],
        )
        worker = get_worker(
            config.distributed_algorithm,
            trainer=worker_trainer,
            server=server,
            round=config.round,
            worker_id=worker_id,
        )
        worker_pool.exec(worker.train, device=devices[worker_id % len(devices)])
    get_logger().info("begin training")
    worker_pool.stop()
    get_logger().info("end training")
    server.stop()
