from sign_sgd_server import SignSGDServer
from sign_sgd_worker import SignSGDWorker


def get_server(algorithm, **kwargs):
    if algorithm == "sign_SGD":
        return SignSGDServer(**kwargs)
    raise RuntimeError("unknown algorithm:" + algorithm)


def get_worker(algorithm, **kwargs):
    if algorithm == "sign_SGD":
        return SignSGDWorker(**kwargs)
    raise RuntimeError("unknown algorithm:" + algorithm)
