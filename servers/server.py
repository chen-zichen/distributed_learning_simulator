class Server:
    def __init__(self, tester, worker_num):
        self.__tester = tester
        self.__worker_num = worker_num

    @property
    def tester(self):
        return self.__tester

    @property
    def worker_num(self):
        return self.__worker_num
