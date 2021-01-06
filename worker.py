import abc


class Worker(abc.ABC):
    @abc.abstractmethod
    def train(self):
        pass
