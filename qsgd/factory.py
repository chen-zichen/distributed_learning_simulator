# from fed_quant_server import FedQuantServer
# from fed_quant_worker import FedQuantWorker
# from sign_sgd_server import SignSGDServer
# from sign_sgd_worker import SignSGDWorker
from qsgd_server import QsgdServer
from qsgd_worker import QsgdWorker
from sketch_server import SketchsgdServer
from sketch_worker import SketchsgdWorker


def get_server(algorithm, **kwargs):
    if algorithm == "sign_SGD":
        return SignSGDServer(**kwargs)
    if algorithm == "fed_quant":
        return FedQuantServer(**kwargs)
    if algorithm == "qsgd":
        return QsgdServer(**kwargs)
    if algorithm == "sketch":
        return SketchsgdServer(**kwargs)

    raise RuntimeError("unknown algorithm:" + algorithm)


def get_worker(algorithm, **kwargs):
    if algorithm == "sign_SGD":
        return SignSGDWorker(**kwargs)
    if algorithm == "fed_quant":
        return FedQuantWorker(**kwargs)
    if algorithm == "qsgd":
        return QsgdWorker(**kwargs)
    if algorithm == "sketch":
        return SketchsgdWorker(**kwargs)
    raise RuntimeError("unknown algorithm:" + algorithm)
