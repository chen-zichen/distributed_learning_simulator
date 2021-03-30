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
        super().load_args(parser=parser)


def get_config(parser=None) -> ExperimentConfig:
    config = ExperimentConfig()
    config.load_args(parser=parser)
    return config
