from cyy_naive_lib.log import get_logger
from cyy_naive_pytorch_lib.tensor import (concat_dict_values,
                                          get_data_serialization_size,
                                          load_dict_values)

from fed_server import FedServer


class MultiRoundShapleyValueServer(FedServer):
    def _process_aggregated_parameter(self, aggregated_parameter: dict):
        pass


    def compute_shapley_value(self,idxs,**kwargs):
        round=self.round
        V_S_t=kwargs['V_func']

        util={}
        powerset=list(powersettool(idxs))
        for S in powerset:
            util[S]=V_S_t(t=round,S=S)

        #for print only
        self.full_set=powerset[-1]

        self.SV_t[t]=self.shapley_value(util,idxs)

        self.Ut[t]=copy.deepcopy(util)

        self.print_results(t)

        return self.SV_t[t]
