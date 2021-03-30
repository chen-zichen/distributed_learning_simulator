import argparse

from cyy_naive_pytorch_lib.default_config import DefaultConfig


class ExperimentConfig(DefaultConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.distributed_algorithm = None
        self.worker_number = None
        self.round = None

    def load_args(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument("--distributed_algorithm", type=str, required=True)
        parser.add_argument("--worker_number", type=int, required=True)
        parser.add_argument("--round", type=int, required=True)
        super().load_args(parser=parser)


def get_config(parser=None) -> ExperimentConfig:
    config = ExperimentConfig()
    config.load_args(parser=parser)
    return config
